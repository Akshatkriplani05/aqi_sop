"""Tests for the Data Quality Score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.quality.scoring import compute_quality_scores


def test_weights_sum_to_one(tmp_cfg):
    assert abs(sum(tmp_cfg.quality["weights"].values()) - 1.0) < 1e-9


def test_perfect_day_scores_high(tmp_cfg):
    # A full, clean day of hourly data for one parameter.
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    valid = pd.DataFrame({
        "station_id": "s1",
        "timestamp_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameter": "pm25",
        "value": np.linspace(10, 20, 24),
    })
    run_time = pd.to_datetime(valid["timestamp_utc"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
    q = compute_quality_scores(valid, pd.DataFrame(), tmp_cfg, "run1", run_time)
    comp = q.set_index("metric_name")["metric_value"]
    assert comp["completeness"] == 100.0
    assert comp["validity"] == 100.0
    assert comp["uniqueness"] == 100.0
    assert comp["freshness"] == 100.0
    assert comp["overall"] >= 95.0


def test_quarantine_lowers_validity(tmp_cfg):
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    valid = pd.DataFrame({
        "station_id": "s1",
        "timestamp_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameter": "pm25",
        "value": np.linspace(10, 20, 24),
    })
    quar = pd.DataFrame({
        "station_id": ["s1"] * 4,
        "timestamp_utc": idx[:4].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameter": ["pm25"] * 4,
        "value": [np.nan] * 4,
        "reason_code": ["MISSING_VALUE"] * 4,
    })
    run_time = pd.to_datetime(valid["timestamp_utc"].iloc[-1], utc=True) + pd.Timedelta(hours=1)
    q = compute_quality_scores(valid, quar, tmp_cfg, "run1", run_time)
    comp = q.set_index("metric_name")["metric_value"]
    assert comp["validity"] < 100.0
