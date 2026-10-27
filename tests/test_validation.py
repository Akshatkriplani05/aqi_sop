"""Tests for the data-quality rules and validator."""
from __future__ import annotations

import pandas as pd

from src.common.reason_codes import ReasonCode
from src.validation import rules
from src.validation.validator import run_validation


def test_missing_value(sample_air):
    assert rules.missing_value(sample_air).sum() == 1


def test_duplicate_record(sample_air):
    # Exactly one of the two identical (s1, 00:00, pm25) rows is flagged.
    assert rules.duplicate_record(sample_air).sum() == 1


def test_invalid_parameter(sample_air):
    assert rules.invalid_parameter(sample_air, {"pm25", "pm10"}).sum() == 1


def test_domain_violation(sample_air):
    cfg = {"pm25": {"min": 0.0, "max": 1000.0}}
    assert rules.domain_violation(sample_air, cfg).sum() == 1


def test_invalid_timestamp():
    df = pd.DataFrame({"timestamp_utc": ["2024-01-01T00:00:00Z", "not-a-date"]})
    assert rules.invalid_timestamp(df).tolist() == [False, True]


def test_broken_station_ref(sample_air):
    assert rules.broken_station_ref(sample_air, {"s1"}).sum() == 1


def test_persistent_constant():
    df = pd.DataFrame({
        "station_id": ["s1"] * 6,
        "parameter": ["pm25"] * 6,
        "timestamp_utc": pd.date_range("2024-01-01", periods=6, freq="h").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "value": [7.0] * 6,
    })
    assert rules.persistent_constant(df, run_length=6).sum() == 6


def test_validator_quarantines_without_dropping(sample_air, tmp_cfg):
    run_time = pd.Timestamp.now(tz="UTC")
    res = run_validation(sample_air, {"s1"}, tmp_cfg, run_time)
    # No row is silently lost: valid + quarantined == input.
    assert len(res.valid) + len(res.quarantined) == len(sample_air)
    # Every quarantined row has a reason code.
    assert res.quarantined["reason_code"].notna().all()
    assert set(res.quarantined["reason_code"]).issubset({c.value for c in ReasonCode})
