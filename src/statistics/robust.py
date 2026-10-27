"""Robust outlier detection.

Environmental concentration data is right-skewed and heavy-tailed, so methods
built on the mean/standard deviation (e.g. z-score) break down: a few extreme
values inflate the std and mask real outliers (low breakdown point). These
methods use order statistics (quantiles, median, MAD) with a high breakdown
point, and a context-aware envelope so normal diurnal peaks are not mislabeled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 1.4826 scales MAD to be a consistent estimator of std for normal data.
_MAD_SCALE = 1.4826


def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fence bounds. k=1.5 'outlier', k=3.0 'extreme/far out'."""
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_flags(x: pd.Series, k: float = 1.5) -> pd.Series:
    """Global IQR outlier flags. Skew-robust and distribution-free."""
    lo, hi = iqr_bounds(x, k)
    return (x < lo) | (x > hi)


def mad_robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score (Iglewicz & Hoaglin): 0.6745 * (x - median) / MAD.

    0.6745 is the 0.75 quantile of the standard normal, making the statistic
    comparable to a z-score. More robust than mean/std because the median and
    MAD have a 50% breakdown point.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        # Degenerate (constant-ish) series: fall back to a tiny epsilon.
        mad = 1e-9
    return 0.6745 * (x - med) / mad


def mad_flags(x: pd.Series, threshold: float = 3.5) -> pd.Series:
    return mad_robust_z(x).abs() > threshold


def hampel_filter(
    x: pd.Series, window: int = 7, n_sigma: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Hampel filter for LOCAL anomalies in a time series.

    Uses a centered rolling median and rolling MAD. A point is anomalous if it
    deviates from the local median by more than n_sigma * scaled-MAD. Unlike the
    global methods above, this adapts to slow trends and seasonality, catching
    point spikes that are abnormal *relative to their neighbourhood*.

    Returns (flags, rolling_median).
    """
    med = x.rolling(window, center=True, min_periods=1).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * _MAD_SCALE * mad
    # Normal case: deviation exceeds the robust local scale. Degenerate case:
    # a perfectly constant neighbourhood gives MAD=0, so any nonzero deviation
    # from that constant level is itself anomalous.
    flags = np.where(threshold > 0, abs_dev > threshold, abs_dev > 0)
    return pd.Series(flags, index=x.index).fillna(False), med


def rolling_percentile_envelope(
    df: pd.DataFrame,
    value_col: str = "value",
    group_cols: tuple[str, ...] = ("hour",),
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.Series:
    """Context-aware flags: compare each value to the historical distribution for
    comparable periods (e.g. same hour-of-day), avoiding false positives on
    predictable patterns such as rush-hour peaks.

    Expects df to already contain the grouping columns (e.g. 'hour', 'dow').
    """
    lo = df.groupby(list(group_cols))[value_col].transform(lambda s: s.quantile(lower_q))
    hi = df.groupby(list(group_cols))[value_col].transform(lambda s: s.quantile(upper_q))
    return (df[value_col] < lo) | (df[value_col] > hi)
