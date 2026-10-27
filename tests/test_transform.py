"""Tests for cleaning: unit normalization, UTC flooring, dedup aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.transform.clean import _to_ugm3, clean_air_measurements


def test_mgm3_to_ugm3():
    assert _to_ugm3(1.0, "mg/m3", "pm25") == 1000.0


def test_ppb_to_ugm3_no2():
    # NO2: ug/m3 = ppb * MW / 24.45 ; 1 ppb ~ 1.88 ug/m3.
    val = _to_ugm3(1.0, "ppb", "no2")
    assert abs(val - 46.0055 / 24.45) < 1e-6


def test_ugm3_passthrough_and_unicode():
    assert _to_ugm3(5.0, "µg/m³", "pm25") == 5.0


def test_clean_floors_to_hour_and_dedup(tmp_cfg):
    df = pd.DataFrame({
        "station_id": ["s1", "s1"],
        "timestamp_utc": ["2024-01-01T00:10:00Z", "2024-01-01T00:50:00Z"],
        "parameter": ["pm25", "pm25"],
        "value": [10.0, 20.0],
        "unit": ["ug/m3", "ug/m3"],
        "source": ["t", "t"],
        "ingestion_timestamp": ["2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z"],
    })
    out = clean_air_measurements(df, tmp_cfg, "run1")
    # Both sub-hour readings collapse to one hourly row, averaged.
    assert len(out) == 1
    assert out.iloc[0]["timestamp_utc"] == "2024-01-01T00:00:00Z"
    assert out.iloc[0]["value"] == 15.0
    assert out.iloc[0]["unit"] == "ug/m3"


def test_clean_handles_empty(tmp_cfg):
    out = clean_air_measurements(pd.DataFrame(), tmp_cfg, "run1")
    assert out.empty
