"""Validation orchestration.

Applies the hard rules (which quarantine rows) and the soft rules (which produce
auditable issue counts), returning a structured result. Rows are *quarantined*,
never silently dropped, and each failing row is annotated with its reason code
and evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.reason_codes import ReasonCode
from src.validation import rules

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame  # includes reason_code + evidence columns
    issue_counts: dict[str, int] = field(default_factory=dict)
    gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    staleness: pd.DataFrame = field(default_factory=pd.DataFrame)
    schema_ok: bool = True


# Hard rules evaluated row-wise. Order matters only for which reason is assigned
# first; each row keeps the first failing reason for a clean audit trail.
def _apply_hard_rules(
    df: pd.DataFrame, known_station_ids: set[str], cfg: Config
) -> pd.Series:
    valid_params = set(cfg.pollutants.keys())
    checks: list[tuple[ReasonCode, pd.Series]] = [
        (ReasonCode.BROKEN_STATION_REF, rules.broken_station_ref(df, known_station_ids)),
        (ReasonCode.INVALID_PARAMETER, rules.invalid_parameter(df, valid_params)),
        (ReasonCode.INVALID_TIMESTAMP, rules.invalid_timestamp(df)),
        (ReasonCode.MISSING_VALUE, rules.missing_value(df)),
        (ReasonCode.UNIT_MISMATCH, rules.unit_mismatch(df)),
        (ReasonCode.DOMAIN_VIOLATION, rules.domain_violation(df, cfg.pollutants)),
        (ReasonCode.DUPLICATE_RECORD, rules.duplicate_record(df)),
    ]
    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    for code, mask in checks:
        assign = mask & reason.isna()
        reason.loc[assign] = code.value
    return reason


def run_validation(
    df: pd.DataFrame,
    known_station_ids: set[str],
    cfg: Config,
    run_time_utc: pd.Timestamp,
) -> ValidationResult:
    if df.empty:
        return ValidationResult(valid=df.copy(), quarantined=df.copy())

    missing_cols = rules.REQUIRED_AIR_COLUMNS - set(df.columns)
    if missing_cols:
        logger.error("Schema mismatch, missing columns: %s", missing_cols)
        quar = df.copy()
        quar["reason_code"] = ReasonCode.SCHEMA_MISMATCH.value
        quar["evidence"] = f"missing_columns={sorted(missing_cols)}"
        return ValidationResult(valid=df.iloc[0:0].copy(), quarantined=quar, schema_ok=False)

    reason = _apply_hard_rules(df, known_station_ids, cfg)
    fail_mask = reason.notna()

    valid = df.loc[~fail_mask].copy()
    quarantined = df.loc[fail_mask].copy()
    quarantined["reason_code"] = reason.loc[fail_mask].values
    quarantined["evidence"] = quarantined.apply(
        lambda r: f"parameter={r['parameter']}, value={r['value']}, unit={r['unit']}", axis=1
    )

    # Soft rules: computed on the *valid* set, reported as counts (not quarantined).
    stale_after = cfg.quality["thresholds"]["freshness_zero_hours"]
    gap_hours = cfg.quality["thresholds"]["max_gap_hours"]
    const_run = cfg.quality["thresholds"]["persistent_constant_run"]

    non_mono = rules.non_monotonic_timestamps(valid) if not valid.empty else pd.Series(dtype=bool)
    const = rules.persistent_constant(valid, const_run) if not valid.empty else pd.Series(dtype=bool)
    gaps = rules.time_gaps(valid, gap_hours) if not valid.empty else pd.DataFrame()
    staleness = rules.stale_data(valid, run_time_utc, stale_after) if not valid.empty else pd.DataFrame()

    issue_counts = {code: int((reason == code).sum()) for code in reason.dropna().unique()}
    issue_counts[ReasonCode.NON_MONOTONIC.value] = int(non_mono.sum()) if len(non_mono) else 0
    issue_counts[ReasonCode.PERSISTENT_CONSTANT.value] = int(const.sum()) if len(const) else 0
    issue_counts[ReasonCode.TIME_GAP.value] = int(gaps["n_gaps"].sum()) if not gaps.empty else 0
    issue_counts[ReasonCode.STALE_DATA.value] = (
        int(staleness["is_stale"].sum()) if not staleness.empty else 0
    )

    logger.info(
        "Validation: %d valid, %d quarantined, issues=%s",
        len(valid), len(quarantined), issue_counts,
    )
    return ValidationResult(
        valid=valid,
        quarantined=quarantined,
        issue_counts=issue_counts,
        gaps=gaps,
        staleness=staleness,
    )
