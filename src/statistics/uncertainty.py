"""Bootstrap confidence intervals.

The nonparametric bootstrap resamples the observed data with replacement to
approximate the sampling distribution of a statistic, making no normality
assumption. We report percentile intervals.

Correct interpretation: "95% of such intervals, computed from repeated samples,
would contain the true parameter." It is NOT correct to say "there is a 95%
probability the true value lies in this particular interval" (frequentist CIs
make no probability statement about a fixed parameter).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _ci_from_dist(estimates: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lo = np.percentile(estimates, 100 * alpha / 2)
    hi = np.percentile(estimates, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def bootstrap_statistic(
    x: pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for a univariate statistic (default: mean)."""
    data = x.dropna().to_numpy()
    if len(data) < 3:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": len(data)}
    rng = np.random.default_rng(seed)
    n = len(data)
    estimates = np.array(
        [statistic(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)]
    )
    lo, hi = _ci_from_dist(estimates, confidence)
    return {
        "estimate": float(statistic(data)),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
        "n": n,
    }


def bootstrap_difference(
    a: pd.Series,
    b: pd.Series,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the difference in a statistic between two samples."""
    da, db = a.dropna().to_numpy(), b.dropna().to_numpy()
    if len(da) < 3 or len(db) < 3:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    diffs = np.array(
        [
            statistic(rng.choice(da, size=len(da), replace=True))
            - statistic(rng.choice(db, size=len(db), replace=True))
            for _ in range(n_boot)
        ]
    )
    lo, hi = _ci_from_dist(diffs, confidence)
    return {
        "estimate": float(statistic(da) - statistic(db)),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
    }


def bootstrap_correlation(
    x: pd.Series, y: pd.Series, n_boot: int = 2000, confidence: float = 0.95, seed: int = 42
) -> dict[str, float]:
    """Bootstrap CI for the Spearman correlation between two paired series."""
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 5:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    arr = xy.to_numpy()
    n = len(arr)
    from scipy.stats import spearmanr

    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        estimates.append(spearmanr(sample[:, 0], sample[:, 1]).correlation)
    estimates = np.array(estimates)
    lo, hi = _ci_from_dist(estimates, confidence)
    return {
        "estimate": float(spearmanr(arr[:, 0], arr[:, 1]).correlation),
        "ci_low": lo,
        "ci_high": hi,
        "confidence": confidence,
        "n": n,
    }
