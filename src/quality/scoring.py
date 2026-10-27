"""Data Quality Score (single, transparent, weighted).

Q = wC*C + wV*V + wU*U + wT*T + wF*F + wP*P,  with sum(w) = 1, each in [0,100].

  C Completeness        observed valid hours / expected hours
  V Validity            share of received rows that parsed/typed/schema-passed
  U Uniqueness          1 - duplicate_rows / received
  T Temporal consistency 1 - (gaps + non-monotonic) / expected intervals
  F Freshness           decay on age of the station's latest observation
  P Plausibility        1 - (domain + stuck-sensor rows) / received

The weights (config/quality_rules.yaml) are a defensible default for hourly
environmental data, NOT a universal standard. Every component is auditable: the
tidy output stores each component and the overall score per (station, date), and
the reason-code counts explain exactly which rows drove each deduction.

"Station reliability" is this score interpreted across time (a station whose
daily Q stays high is reliable), not a separate score.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.logging_utils import get_logger
from src.common.reason_codes import ReasonCode
from src.validation import rules

logger = get_logger(__name__)

_VALIDITY_CODES = {
    ReasonCode.INVALID_TIMESTAMP.value,
    ReasonCode.UNIT_MISMATCH.value,
    ReasonCode.INVALID_PARAMETER.value,
    ReasonCode.SCHEMA_MISMATCH.value,
    ReasonCode.BROKEN_STATION_REF.value,
    ReasonCode.MISSING_VALUE.value,
}


def _freshness_score(age_hours: float, full_h: float, zero_h: float) -> float:
    """Linear decay: 100 until full_h, down to 0 at zero_h."""
    if age_hours <= full_h:
        return 100.0
    if age_hours >= zero_h:
        return 0.0
    return 100.0 * (zero_h - age_hours) / (zero_h - full_h)


def _status(score: float, pass_min: float, warn_min: float) -> str:
    if score >= pass_min:
        return "PASS"
    if score >= warn_min:
        return "WARN"
    return "FAIL"


def compute_quality_scores(
    valid_df: pd.DataFrame,
    quar_df: pd.DataFrame,
    cfg: Config,
    run_id: str,
    run_time_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Return a tidy quality_metrics frame (one row per station/date/metric)."""
    weights = cfg.quality["weights"]
    th = cfg.quality["thresholds"]
    rule_version = cfg.quality["rule_version"]

    if valid_df.empty and quar_df.empty:
        return pd.DataFrame()

    vd = valid_df.copy()
    vd["_dt"] = pd.to_datetime(vd["timestamp_utc"], utc=True, errors="coerce")
    vd["date"] = vd["_dt"].dt.strftime("%Y-%m-%d")

    qd = quar_df.copy()
    if not qd.empty:
        qd["_dt"] = pd.to_datetime(qd["timestamp_utc"], utc=True, errors="coerce")
        qd["date"] = qd["_dt"].dt.strftime("%Y-%m-%d")
        # Attribute unparseable-timestamp rows to the station's earliest valid day.
        first_day = vd.groupby("station_id")["date"].min().to_dict()
        qd["date"] = qd.apply(
            lambda r: r["date"] if pd.notna(r["date"]) else first_day.get(r["station_id"]),
            axis=1,
        )

    # Station-level freshness (latest obs age at run time).
    latest = vd.groupby("station_id")["_dt"].max()
    fresh_by_station = {
        sid: _freshness_score(
            (run_time_utc - ts).total_seconds() / 3600.0,
            th["freshness_full_hours"], th["freshness_zero_hours"],
        )
        for sid, ts in latest.items()
    }

    # Distinct parameters per station (for expected-slot calculation).
    params_by_station = vd.groupby("station_id")["parameter"].nunique().to_dict()
    const_run = th["persistent_constant_run"]
    max_gap = th["max_gap_hours"]

    records: list[dict] = []
    station_dates = vd[["station_id", "date"]].drop_duplicates()

    for _, row in station_dates.iterrows():
        sid, date = row["station_id"], row["date"]
        day_valid = vd[(vd["station_id"] == sid) & (vd["date"] == date)]
        day_quar = (
            qd[(qd["station_id"] == sid) & (qd["date"] == date)]
            if not qd.empty else qd
        )
        n_valid = len(day_valid)
        n_quar = len(day_quar)
        received = n_valid + n_quar
        n_params = max(1, params_by_station.get(sid, 1))
        expected = 24 * n_params

        # Completeness.
        C = min(100.0, 100.0 * n_valid / expected)

        # Validity & Uniqueness from quarantine reason codes.
        if n_quar:
            n_invalid = int(day_quar["reason_code"].isin(_VALIDITY_CODES).sum())
            n_dupe = int((day_quar["reason_code"] == ReasonCode.DUPLICATE_RECORD.value).sum())
            n_domain = int((day_quar["reason_code"] == ReasonCode.DOMAIN_VIOLATION.value).sum())
        else:
            n_invalid = n_dupe = n_domain = 0
        V = 100.0 * (received - n_invalid) / received if received else 100.0
        U = 100.0 * (received - n_dupe) / received if received else 100.0

        # Temporal consistency: gaps + non-monotonic within the day.
        gaps_df = rules.time_gaps(day_valid, max_gap)
        n_gaps = int(gaps_df["n_gaps"].sum()) if not gaps_df.empty else 0
        nonmono = rules.non_monotonic_timestamps(day_valid)
        n_nonmono = int(nonmono.sum()) if len(nonmono) else 0
        T = max(0.0, 100.0 * (1 - (n_gaps + n_nonmono) / expected))

        # Plausibility: domain violations (quarantined) + stuck-sensor runs (valid).
        stuck = rules.persistent_constant(day_valid, const_run)
        n_stuck = int(stuck.sum()) if len(stuck) else 0
        P = max(0.0, 100.0 * (1 - (n_domain + n_stuck) / received)) if received else 100.0

        F = fresh_by_station.get(sid, 0.0)

        overall = (
            weights["completeness"] * C
            + weights["validity"] * V
            + weights["uniqueness"] * U
            + weights["temporal_consistency"] * T
            + weights["freshness"] * F
            + weights["plausibility"] * P
        )

        components = {
            "completeness": C, "validity": V, "uniqueness": U,
            "temporal_consistency": T, "freshness": F, "plausibility": P,
            "overall": overall,
        }
        for metric, value in components.items():
            records.append(
                {
                    "station_id": sid,
                    "date": date,
                    "metric_name": metric,
                    "metric_value": round(value, 2),
                    "threshold": th["pass_min"] if metric == "overall" else None,
                    "status": _status(value, th["pass_min"], th["warn_min"])
                    if metric == "overall" else None,
                    "rule_version": rule_version,
                    "pipeline_run_id": run_id,
                }
            )

    out = pd.DataFrame(records)
    logger.info("Computed quality scores for %d station-days", len(station_dates))
    return out
