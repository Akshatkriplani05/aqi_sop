"""Time-series statistics.

Environmental series are explicitly NOT i.i.d.: consecutive hours are strongly
autocorrelated, and there are diurnal, weekly, and seasonal cycles. That
temporal dependence is why we favour rolling/robust statistics and why naive
tests assuming independence overstate significance. These helpers quantify the
dependence (ACF/PACF) and decompose structure (STL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import acf as _sm_acf, pacf as _sm_pacf
    from statsmodels.tsa.seasonal import STL as _STL
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


def rolling_stats(x: pd.Series, window: int) -> pd.DataFrame:
    """Rolling mean, median, std, MAD, IQR, and coefficient of variation."""
    roll = x.rolling(window, min_periods=max(2, window // 3))
    mean = roll.mean()
    median = roll.median()
    std = roll.std()
    mad = roll.apply(lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
    q1 = roll.quantile(0.25)
    q3 = roll.quantile(0.75)
    iqr = q3 - q1
    cv = std / mean.replace(0, np.nan)
    return pd.DataFrame(
        {
            "rolling_mean": mean,
            "rolling_median": median,
            "rolling_std": std,
            "rolling_mad": mad,
            "rolling_iqr": iqr,
            "rolling_cv": cv,
        }
    )


def differences(x: pd.Series) -> pd.DataFrame:
    """First difference and percentage change."""
    return pd.DataFrame({"delta": x.diff(), "pct_change": x.pct_change() * 100.0})


def lags(x: pd.Series, lag_list: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"lag_{k}": x.shift(k) for k in lag_list})


def percentile_rank(x: pd.Series) -> pd.Series:
    """Empirical percentile rank of each observation within the series (0-1)."""
    return x.rank(pct=True)


def acf_pacf(x: pd.Series, nlags: int = 48) -> pd.DataFrame:
    """Autocorrelation and partial autocorrelation up to nlags."""
    xv = x.dropna().to_numpy()
    nlags = min(nlags, len(xv) // 2 - 1) if len(xv) > 4 else 1
    if not _HAS_SM or nlags < 1:
        return pd.DataFrame(columns=["lag", "acf", "pacf"])
    a = _sm_acf(xv, nlags=nlags, fft=True)
    p = _sm_pacf(xv, nlags=nlags)
    return pd.DataFrame({"lag": range(len(a)), "acf": a, "pacf": p[: len(a)]})


def stl_decompose(x: pd.Series, period: int = 24) -> pd.DataFrame:
    """STL decomposition into observed / trend / seasonal / residual.

    period=24 targets the daily cycle for hourly data. Requires a reasonably
    complete, regularly-spaced series.
    """
    xv = x.interpolate(limit_direction="both")
    if not _HAS_SM or len(xv.dropna()) < 2 * period:
        return pd.DataFrame({"observed": x})
    res = _STL(xv, period=period, robust=True).fit()
    return pd.DataFrame(
        {
            "observed": x.values,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
        },
        index=x.index,
    )
