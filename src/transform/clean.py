"""Cleaning & standardization.

Takes validated raw rows and produces the canonical clean form:
  * timestamps floored to the hour in strict UTC ISO8601 ('...Z'),
  * concentrations normalized to a single unit (ug/m3),
  * duplicates on the natural key collapsed deterministically (mean),
  * numeric types coerced.

Unit conversion for gases uses the ideal-gas relation at standard conditions
(25 C, 1 atm): ug/m3 = ppb * MW / 24.45. This is an approximation (true
conversion depends on ambient T and P), documented as a known limitation.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# Molar masses (g/mol) for gaseous pollutants, for ppb/ppm -> ug/m3 conversion.
_MOLAR_MASS = {"no2": 46.0055, "so2": 64.066, "co": 28.01, "o3": 48.00}
_MOLAR_VOLUME_25C = 24.45  # L/mol at 25 C, 1 atm


def _to_ugm3(value: float, unit: str, parameter: str) -> float:
    u = str(unit).strip().lower().replace("µ", "u").replace("³", "3").replace("μ", "u")
    if value is None or pd.isna(value):
        return value
    if u in ("ug/m3", "ug/m^3"):
        return value
    if u == "mg/m3":
        return value * 1000.0
    mw = _MOLAR_MASS.get(parameter.lower())
    if mw is None:
        return value  # PM in volumetric units is not meaningful; leave as-is.
    if u == "ppm":
        return value * 1000.0 * mw / _MOLAR_VOLUME_25C
    if u == "ppb":
        return value * mw / _MOLAR_VOLUME_25C
    return value


def clean_air_measurements(df: pd.DataFrame, cfg: Config, run_id: str) -> pd.DataFrame:
    """Return the canonical clean air_measurements frame (idempotent-ready)."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["parameter"] = out["parameter"].str.lower()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # Normalize units -> ug/m3.
    out["value"] = out.apply(
        lambda r: _to_ugm3(r["value"], r["unit"], r["parameter"]), axis=1
    )
    out["unit"] = "ug/m3"

    # Canonical UTC timestamp floored to the hour.
    ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")
    out["timestamp_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.dropna(subset=["timestamp_utc"])

    # Collapse duplicates on the natural key by averaging (sub-hourly -> hourly).
    group_keys = ["station_id", "timestamp_utc", "parameter"]
    agg = (
        out.groupby(group_keys, as_index=False)
        .agg(
            value=("value", "mean"),
            unit=("unit", "first"),
            source=("source", "first"),
            ingestion_timestamp=("ingestion_timestamp", "max"),
        )
    )
    agg["pipeline_run_id"] = run_id
    logger.info("Cleaned air measurements: %d -> %d rows", len(df), len(agg))
    return agg


def clean_weather(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Standardize weather to canonical columns + hourly UTC, dedup on (city, ts)."""
    if df.empty:
        return df.copy()
    out = df.copy()
    ts = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.floor("h")
    out["timestamp_utc"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.dropna(subset=["timestamp_utc"])
    cols = [
        "city", "timestamp_utc", "temperature", "humidity", "precipitation",
        "pressure", "wind_speed", "wind_direction", "source",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols].drop_duplicates(subset=["city", "timestamp_utc"], keep="last")
    out["pipeline_run_id"] = run_id
    logger.info("Cleaned weather: %d rows", len(out))
    return out
