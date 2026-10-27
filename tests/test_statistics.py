"""Tests for robust statistics, correlation, uncertainty, hypothesis tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics import correlation, hypothesis, robust, uncertainty


def test_iqr_flags_catches_outlier():
    x = pd.Series([10, 11, 12, 10, 11, 9, 10, 500])
    flags = robust.iqr_flags(x, 1.5)
    assert bool(flags.iloc[-1]) is True
    assert flags.iloc[:-1].sum() == 0


def test_mad_robust_z_resists_masking():
    # Mean/std would be inflated by the extreme value; MAD should still flag it.
    x = pd.Series([10, 11, 10, 12, 11, 10, 1000])
    assert robust.mad_flags(x, 3.5).iloc[-1]


def test_hampel_local_detection():
    x = pd.Series([10.0] * 20)
    x.iloc[10] = 200.0  # local spike
    flags, _ = robust.hampel_filter(x, window=7, n_sigma=3.0)
    assert bool(flags.iloc[10]) is True


def test_spearman_monotonic():
    x = pd.Series(np.arange(50, dtype=float))
    y = x ** 2  # monotonic but nonlinear -> Spearman ~ 1
    res = correlation.spearman(x, y)
    assert res["rho"] > 0.99


def test_partial_correlation_removes_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z + rng.normal(scale=0.1, size=500)
    y = z + rng.normal(scale=0.1, size=500)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    marginal = correlation.spearman(df["x"], df["y"])["rho"]
    partial = correlation.partial_correlation(df, "x", "y", ["z"])["partial_r"]
    # x and y are correlated only through z; controlling for z collapses it.
    assert marginal > 0.8
    assert abs(partial) < 0.3


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(50, 5, 1000))
    res = uncertainty.bootstrap_statistic(x, np.mean, 1000, 0.95)
    assert res["ci_low"] < 50 < res["ci_high"]


def test_mann_whitney_detects_shift():
    a = pd.Series(np.arange(0, 100, dtype=float))
    b = pd.Series(np.arange(50, 150, dtype=float))
    res = hypothesis.mann_whitney(a, b)
    assert res["p_value"] < 0.05


def test_kruskal_and_posthoc():
    groups = {
        "a": pd.Series(np.random.default_rng(2).normal(10, 1, 100)),
        "b": pd.Series(np.random.default_rng(3).normal(20, 1, 100)),
        "c": pd.Series(np.random.default_rng(4).normal(30, 1, 100)),
    }
    kw = hypothesis.kruskal_wallis(groups)
    assert kw["p_value"] < 0.05
    ph = hypothesis.posthoc_pairwise(groups)
    assert ph["significant"].all()
    assert "p_holm" in ph.columns
