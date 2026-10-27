"""Shared pytest fixtures.

Builds an isolated Config pointing at a temporary directory and a single city so
pipeline/integration tests run fast and never touch the real data directory.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import get_config, Config


@pytest.fixture
def tmp_cfg(tmp_path) -> Config:
    base = get_config()
    raw = copy.deepcopy(base.raw)
    raw["paths"] = {
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "quarantine_dir": str(tmp_path / "quarantine"),
        "db_path": str(tmp_path / "processed" / "aqp.db"),
    }
    cities = copy.deepcopy(base.cities_raw)
    cities["cities"] = cities["cities"][:1]  # single city keeps tests quick
    cfg = Config(
        raw=raw,
        cities_raw=cities,
        quality_raw=copy.deepcopy(base.quality_raw),
        openaq_api_key=None,
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def sample_air() -> pd.DataFrame:
    """Small hand-built raw air frame with known defects for validation tests."""
    rows = [
        # valid rows
        {"station_id": "s1", "timestamp_utc": "2024-01-01T00:00:00Z", "parameter": "pm25",
         "value": 10.0, "unit": "ug/m3"},
        {"station_id": "s1", "timestamp_utc": "2024-01-01T01:00:00Z", "parameter": "pm25",
         "value": 12.0, "unit": "ug/m3"},
        # missing value
        {"station_id": "s1", "timestamp_utc": "2024-01-01T02:00:00Z", "parameter": "pm25",
         "value": np.nan, "unit": "ug/m3"},
        # domain violation (negative)
        {"station_id": "s1", "timestamp_utc": "2024-01-01T03:00:00Z", "parameter": "pm25",
         "value": -5.0, "unit": "ug/m3"},
        # invalid parameter
        {"station_id": "s1", "timestamp_utc": "2024-01-01T04:00:00Z", "parameter": "xyz",
         "value": 3.0, "unit": "ug/m3"},
        # duplicate of first row
        {"station_id": "s1", "timestamp_utc": "2024-01-01T00:00:00Z", "parameter": "pm25",
         "value": 10.0, "unit": "ug/m3"},
        # broken station ref
        {"station_id": "ghost", "timestamp_utc": "2024-01-01T00:00:00Z", "parameter": "pm25",
         "value": 11.0, "unit": "ug/m3"},
    ]
    return pd.DataFrame(rows)
