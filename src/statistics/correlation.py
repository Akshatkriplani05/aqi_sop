"""Association analysis between pollutants and weather.

Pearson measures *linear* association and assumes roughly bivariate-normal,
outlier-sensitive data. Spearman measures *monotonic* association on ranks and
is robust to skew/outliers (appropriate for environmental data). Partial
correlation isolates the association between two variables after linearly
removing the effect of confounders (e.g. the apparent PM2.5-wind link may be
confounded by temperature, humidity, hour, season).

NOTE: correlation (partial or not) is association, NOT causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson(x: pd.Series, y: pd.Series) -> dict[str, float]:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return {"r": np.nan, "p_value": np.nan, "n": len(xy)}
    r, p = stats.pearsonr(xy.iloc[:, 0], xy.iloc[:, 1])
    return {"r": r, "p_value": p, "n": len(xy)}


def spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": len(xy)}
    rho, p = stats.spearmanr(xy.iloc[:, 0], xy.iloc[:, 1])
    return {"rho": rho, "p_value": p, "n": len(xy)}


def partial_correlation(
    df: pd.DataFrame, x: str, y: str, covariates: list[str]
) -> dict[str, float]:
    """Partial correlation of x and y controlling for covariates.

    Computed as the correlation of the residuals of OLS regressions of x and y
    on the covariates. Equivalent to the standard partial-correlation formula.
    """
    cols = [x, y] + covariates
    data = df[cols].dropna()
    if len(data) < len(covariates) + 3:
        return {"partial_r": np.nan, "p_value": np.nan, "n": len(data)}

    Z = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in covariates])

    def _resid(target: str) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(Z, data[target].to_numpy(), rcond=None)
        return data[target].to_numpy() - Z @ beta

    rx, ry = _resid(x), _resid(y)
    r, p = stats.pearsonr(rx, ry)
    return {"partial_r": r, "p_value": p, "n": len(data)}


def lagged_correlation(
    pollutant: pd.Series, weather: pd.Series, lags: list[int], method: str = "spearman"
) -> pd.DataFrame:
    """Correlation of pollutant_t with weather_{t-lag} for each lag.

    A nonzero best-lag suggests a delayed meteorological response (e.g. wind
    clears pollution over several hours). Still association, not causation.
    """
    corr_fn = spearman if method == "spearman" else pearson
    key = "rho" if method == "spearman" else "r"
    rows = []
    for lag in lags:
        shifted = weather.shift(lag)
        res = corr_fn(pollutant, shifted)
        rows.append({"lag": lag, "corr": res[key], "p_value": res["p_value"], "n": res["n"]})
    return pd.DataFrame(rows)
