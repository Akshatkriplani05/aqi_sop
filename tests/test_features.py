"""Tests for feature engineering (lags, deltas, wind components, temporal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.transform.features import add_temporal_features, add_wind_components, compute_features


def _hourly_air(n=60, station="s1", pollutant="pm25"):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "station_id": station,
        "timestamp_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameter": pollutant,
        "value": np.linspace(10, 40, n),
    })


def test_wind_components():
    df = pd.DataFrame({"wind_speed": [10.0], "wind_direction": [0.0]})
    out = add_wind_components(df)
    assert abs(out.iloc[0]["wind_u"] - 10.0) < 1e-9  # cos(0)=1
    assert abs(out.iloc[0]["wind_v"] - 0.0) < 1e-9   # sin(0)=0


def test_temporal_features():
    df = pd.DataFrame({"timestamp_utc": ["2024-07-04T15:00:00Z"]})
    out = add_temporal_features(df)
    assert out.iloc[0]["hour"] == 15
    assert out.iloc[0]["month"] == 7
    assert out.iloc[0]["season"] == 2  # summer (Jun-Aug)


def test_compute_features_lags_and_delta(tmp_cfg):
    df = _hourly_air()
    feats = compute_features(df, tmp_cfg, "run1").sort_values("timestamp_utc").reset_index(drop=True)
    # lag_1 of row i equals value of row i-1.
    assert np.isnan(feats.loc[0, "lag_1"])
    assert abs(feats.loc[1, "lag_1"] - feats.loc[0, "value"]) < 1e-9
    # delta is the first difference.
    assert abs(feats.loc[1, "delta"] - (feats.loc[1, "value"] - feats.loc[0, "value"])) < 1e-9
    # schema columns present.
    for col in ["rolling_mean", "rolling_mad", "ewma", "cusum_pos", "anomaly_flag"]:
        assert col in feats.columns
