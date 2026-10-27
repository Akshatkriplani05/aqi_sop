"""Feature engineering.

Builds the reusable, per-(station, pollutant) statistical feature table. Rolling
statistics use a *time-based* window so irregular spacing does not distort them.
Lags are positional (hourly grain after cleaning). Wind direction is decomposed
into Cartesian components because direction is circular: 350 deg and 10 deg are
20 deg apart, but numerically ~340 apart, so averaging/regressing raw degrees is
meaningless. u = speed*cos(theta), v = speed*sin(theta) are linear and additive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.logging_utils import get_logger
from src.statistics import robust, shift, timeseries

logger = get_logger(__name__)


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    out["hour"] = ts.dt.hour
    out["day"] = ts.dt.day
    out["weekday"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["season"] = ts.dt.month % 12 // 3  # 0=winter,1=spring,2=summer,3=autumn
    return out


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Add Cartesian wind components u, v from wind_speed and wind_direction (deg)."""
    out = df.copy()
    theta = np.deg2rad(out["wind_direction"].astype(float))
    out["wind_u"] = out["wind_speed"].astype(float) * np.cos(theta)
    out["wind_v"] = out["wind_speed"].astype(float) * np.sin(theta)
    return out


def _features_for_group(sub: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute features for a single (station, pollutant) group sorted by time."""
    feat_cfg = cfg.features
    window = int(feat_cfg["rolling_window_hours"])
    sub = sub.sort_values("timestamp_utc").copy()
    idx = pd.to_datetime(sub["timestamp_utc"], utc=True)
    s = pd.Series(sub["value"].to_numpy(), index=idx)

    # Time-based rolling window (e.g. '24h').
    roll = timeseries.rolling_stats(s, window)  # index = datetime
    diffs = timeseries.differences(s)
    lag_df = timeseries.lags(s, feat_cfg["lags"])
    pr = timeseries.percentile_rank(s)
    ew = shift.ewma(s, feat_cfg["ewma_lambda"])
    cu = shift.cusum(s, feat_cfg["cusum_k_sigma"], feat_cfg["cusum_h_sigma"])
    hampel_flags, _ = robust.hampel_filter(
        s, cfg.statistics["hampel_window"], cfg.statistics["hampel_n_sigma"]
    )

    out = pd.DataFrame(
        {
            "station_id": sub["station_id"].to_numpy(),
            "timestamp_utc": sub["timestamp_utc"].to_numpy(),
            "pollutant": sub["parameter"].to_numpy(),
            "value": s.to_numpy(),
            "rolling_mean": roll["rolling_mean"].to_numpy(),
            "rolling_median": roll["rolling_median"].to_numpy(),
            "rolling_std": roll["rolling_std"].to_numpy(),
            "rolling_mad": roll["rolling_mad"].to_numpy(),
            "rolling_iqr": roll["rolling_iqr"].to_numpy(),
            "rolling_cv": roll["rolling_cv"].to_numpy(),
            "percentile_rank": pr.to_numpy(),
            "delta": diffs["delta"].to_numpy(),
            "pct_change": diffs["pct_change"].to_numpy(),
            "ewma": ew.to_numpy(),
            "cusum_pos": cu["cusum_pos"].to_numpy(),
            "cusum_neg": cu["cusum_neg"].to_numpy(),
            "anomaly_flag": hampel_flags.astype(int).to_numpy(),
            "anomaly_method": "HAMPEL",
        }
    )
    for k in feat_cfg["lags"]:
        out[f"lag_{k}"] = lag_df[f"lag_{k}"].to_numpy()
    return out


def compute_features(clean_air: pd.DataFrame, cfg: Config, run_id: str) -> pd.DataFrame:
    """Compute the features_hourly table from clean air measurements."""
    if clean_air.empty:
        return pd.DataFrame()
    frames = []
    for _, sub in clean_air.groupby(["station_id", "parameter"]):
        if len(sub) < 2:
            continue
        frames.append(_features_for_group(sub, cfg))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["pipeline_run_id"] = run_id

    ordered = [
        "station_id", "timestamp_utc", "pollutant", "value",
        "rolling_mean", "rolling_median", "rolling_std", "rolling_mad",
        "rolling_iqr", "rolling_cv", "percentile_rank",
        "lag_1", "lag_3", "lag_6", "lag_12", "lag_24",
        "delta", "pct_change", "ewma", "cusum_pos", "cusum_neg",
        "anomaly_flag", "anomaly_method", "pipeline_run_id",
    ]
    for c in ordered:
        if c not in out.columns:
            out[c] = np.nan
    logger.info("Computed %d feature rows", len(out))
    return out[ordered]


def detect_events(clean_air: pd.DataFrame, cfg: Config, run_id: str) -> pd.DataFrame:
    """Run CUSUM-based sustained-shift detection per (station, pollutant)."""
    if clean_air.empty:
        return pd.DataFrame()
    feat_cfg = cfg.features
    all_events: list[shift.Event] = []
    for (station, pollutant), sub in clean_air.groupby(["station_id", "parameter"]):
        if len(sub) < feat_cfg["cusum_min_persistence"] + 2:
            continue
        sub = sub.sort_values("timestamp_utc")
        s = pd.Series(sub["value"].to_numpy())
        cu = shift.cusum(s, feat_cfg["cusum_k_sigma"], feat_cfg["cusum_h_sigma"])
        events = shift.extract_events(
            sub["timestamp_utc"], s, cu, station, pollutant,
            feat_cfg["cusum_min_persistence"], feat_cfg["cusum_h_sigma"],
        )
        all_events.extend(events)
    return shift.events_to_frame(all_events, run_id)
