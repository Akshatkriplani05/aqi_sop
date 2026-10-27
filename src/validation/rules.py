"""Individual, independently testable data-quality rules.

Each rule returns a boolean pandas Series aligned to the input index where True
marks a *failing* row (hard rules) or a flagged row (soft rules). Keeping rules
pure and small makes them unit-testable and composable in the validator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_AIR_COLUMNS = {
    "station_id",
    "timestamp_utc",
    "parameter",
    "value",
    "unit",
}

# Units we know how to interpret per parameter. Anything else -> UNIT_MISMATCH.
ACCEPTED_UNITS = {"ug/m3", "µg/m³", "ppm", "ppb", "mg/m3"}


def missing_value(df: pd.DataFrame) -> pd.Series:
    return df["value"].isna()


def invalid_timestamp(df: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return parsed.isna()


def duplicate_record(df: pd.DataFrame) -> pd.Series:
    """Duplicate on the natural key (station_id, timestamp_utc, parameter).

    Marks all but the first occurrence as failing.
    """
    return df.duplicated(subset=["station_id", "timestamp_utc", "parameter"], keep="first")


def invalid_parameter(df: pd.DataFrame, valid_parameters: set[str]) -> pd.Series:
    return ~df["parameter"].str.lower().isin(valid_parameters)


def unit_mismatch(df: pd.DataFrame) -> pd.Series:
    return ~df["unit"].astype(str).isin(ACCEPTED_UNITS)


def broken_station_ref(df: pd.DataFrame, known_station_ids: set[str]) -> pd.Series:
    return ~df["station_id"].isin(known_station_ids)


def domain_violation(df: pd.DataFrame, pollutant_cfg: dict[str, Any]) -> pd.Series:
    """Flag values outside the configured physical/domain range for the pollutant."""
    out = pd.Series(False, index=df.index)
    for param, meta in pollutant_cfg.items():
        mask = df["parameter"].str.lower() == param
        vals = pd.to_numeric(df.loc[mask, "value"], errors="coerce")
        bad = (vals < meta["min"]) | (vals > meta["max"])
        out.loc[mask] = bad.fillna(False)
    return out


def non_monotonic_timestamps(df: pd.DataFrame) -> pd.Series:
    """Within each (station, parameter), flag rows where time goes backwards."""
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    tmp = df.assign(_ts=ts)
    flag = pd.Series(False, index=df.index)
    for _, idx in tmp.groupby(["station_id", "parameter"]).groups.items():
        sub = tmp.loc[idx].sort_index()
        decreasing = sub["_ts"].diff() < pd.Timedelta(0)
        flag.loc[sub.index] = decreasing.fillna(False)
    return flag


def persistent_constant(df: pd.DataFrame, run_length: int) -> pd.Series:
    """Flag runs of >= run_length identical consecutive values (stuck sensor)."""
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    tmp = df.assign(_ts=ts)
    flag = pd.Series(False, index=df.index)
    for _, idx in tmp.groupby(["station_id", "parameter"]).groups.items():
        sub = tmp.loc[idx].sort_values("_ts")
        vals = pd.to_numeric(sub["value"], errors="coerce")
        # Group identical consecutive values.
        run_id = (vals != vals.shift()).cumsum()
        run_sizes = run_id.map(run_id.value_counts())
        stuck = (run_sizes >= run_length) & vals.notna()
        flag.loc[sub.index] = stuck.values
    return flag


def time_gaps(df: pd.DataFrame, max_gap_hours: float) -> pd.DataFrame:
    """Return a per-(station,parameter) summary of gaps exceeding max_gap_hours.

    Reported (not quarantined) because a gap concerns *missing* rows, not a bad row.
    """
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    tmp = df.assign(_ts=ts).dropna(subset=["_ts"])
    records = []
    for (station, param), sub in tmp.groupby(["station_id", "parameter"]):
        s = sub.sort_values("_ts")["_ts"]
        gaps = s.diff().dt.total_seconds() / 3600.0
        n_gaps = int((gaps > max_gap_hours).sum())
        records.append(
            {
                "station_id": station,
                "parameter": param,
                "n_gaps": n_gaps,
                "max_gap_hours": float(gaps.max()) if len(gaps.dropna()) else 0.0,
            }
        )
    return pd.DataFrame(records)


def stale_data(df: pd.DataFrame, run_time_utc: pd.Timestamp, stale_after_hours: float) -> pd.DataFrame:
    """Per-(station,parameter) staleness: hours between latest obs and run time."""
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    tmp = df.assign(_ts=ts).dropna(subset=["_ts"])
    records = []
    for (station, param), sub in tmp.groupby(["station_id", "parameter"]):
        latest = sub["_ts"].max()
        age_h = (run_time_utc - latest).total_seconds() / 3600.0
        records.append(
            {
                "station_id": station,
                "parameter": param,
                "latest_ts": latest.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "age_hours": float(age_h),
                "is_stale": bool(age_h > stale_after_hours),
            }
        )
    return pd.DataFrame(records)
