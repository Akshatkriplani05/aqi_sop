"""Tests for EWMA, CUSUM, and event extraction (persistence)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics import shift


def test_ewma_smoothing_bounds():
    x = pd.Series([0.0] * 10 + [10.0] * 10)
    slow = shift.ewma(x, 0.05)
    fast = shift.ewma(x, 0.4)
    # Faster lambda reacts more to the step at the transition.
    assert fast.iloc[11] > slow.iloc[11]


def test_cusum_detects_sustained_shift():
    rng = np.random.default_rng(0)
    x = pd.Series(np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 50)]))
    cu = shift.cusum(x, k_sigma=0.5, h_sigma=5.0)
    assert cu["alarm_pos"].iloc[100:].any()


def test_single_spike_does_not_create_event():
    x = pd.Series([10.0] * 100)
    x.iloc[50] = 1000.0  # isolated spike, not sustained
    cu = shift.cusum(x, 0.5, 5.0)
    events = shift.extract_events(
        pd.date_range("2024-01-01", periods=100, freq="h").to_series().reset_index(drop=True),
        x, cu, "s1", "pm25", min_persistence=3, h_sigma=5.0,
    )
    # A one-hour spike must not be promoted to a sustained event.
    assert all(e.evidence["persistence_hours"] >= 3 for e in events)


def test_sustained_shift_creates_event():
    x = pd.Series(np.concatenate([np.full(60, 10.0), np.full(30, 60.0)]))
    cu = shift.cusum(x, 0.5, 5.0)
    ts = pd.date_range("2024-01-01", periods=90, freq="h").to_series().reset_index(drop=True)
    events = shift.extract_events(ts, x, cu, "s1", "pm25", 3, 5.0)
    assert len(events) >= 1
    assert events[0].event_type == "POSITIVE_SHIFT"
