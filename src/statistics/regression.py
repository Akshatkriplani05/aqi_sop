"""Interpretable regression of pollutant level on weather + calendar effects.

The goal is interpretation, not prediction leaderboard performance, so we use
OLS with HC3 heteroskedasticity-robust standard errors (environmental residuals
are typically heteroskedastic and autocorrelated; HC3 guards the former). PM2.5
is strongly right-skewed, so a log1p transform of the target is offered to make
the additive linear model and its residual assumptions more reasonable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


def build_design(df: pd.DataFrame, weather_cols: list[str]) -> pd.DataFrame:
    """Assemble the design matrix with calendar dummies (hour/dow/season)."""
    d = df.copy()
    ts = pd.to_datetime(d["timestamp_utc"], utc=True)
    d["hour"] = ts.dt.hour
    d["dow"] = ts.dt.dayofweek
    d["season"] = (ts.dt.month % 12 // 3)  # 0=winter..3=autumn
    dummies = pd.get_dummies(
        d[["hour", "dow", "season"]].astype("category"), drop_first=True, dtype=float
    )
    X = pd.concat([d[weather_cols], dummies], axis=1)
    return X


def fit_ols(
    df: pd.DataFrame,
    target: str,
    weather_cols: list[str],
    log_target: bool = True,
    robust: bool = True,
) -> dict:
    """Fit OLS and return coefficients, CIs, R^2, and residual diagnostics."""
    if not _HAS_SM:
        return {"error": "statsmodels not available"}

    data = df.dropna(subset=[target] + weather_cols).copy()
    if len(data) < len(weather_cols) + 10:
        return {"error": "insufficient data", "n": len(data)}

    y = np.log1p(data[target]) if log_target else data[target]
    X = build_design(data, weather_cols)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    model = sm.OLS(y.astype(float), X)
    res = model.fit(cov_type="HC3") if robust else model.fit()

    ci = res.conf_int()
    coefs = pd.DataFrame(
        {
            "coef": res.params,
            "std_err": res.bse,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": res.pvalues,
        }
    )
    # Keep weather terms + const in the headline table (dummies summarized by R^2).
    headline = coefs.loc[[c for c in coefs.index if c in (["const"] + weather_cols)]]
    return {
        "n": int(res.nobs),
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "log_target": log_target,
        "cov_type": res.cov_type,
        "coefficients": coefs,
        "headline": headline,
        "resid": pd.Series(res.resid, index=data.index),
        "fitted": pd.Series(res.fittedvalues, index=data.index),
        "durbin_watson": float(sm.stats.durbin_watson(res.resid)),
    }


def simple_vs_multiple(
    df: pd.DataFrame, target: str, primary: str, weather_cols: list[str], log_target: bool = True
) -> pd.DataFrame:
    """Compare a simple regression (target ~ primary) with the full model to show
    how controlling for other variables changes the primary coefficient.
    """
    simple = fit_ols(df, target, [primary], log_target=log_target)
    full = fit_ols(df, target, weather_cols, log_target=log_target)
    rows = []
    for label, res in [("simple", simple), ("multiple", full)]:
        if "coefficients" in res and primary in res["coefficients"].index:
            c = res["coefficients"].loc[primary]
            rows.append(
                {
                    "model": label,
                    "primary_coef": c["coef"],
                    "ci_low": c["ci_low"],
                    "ci_high": c["ci_high"],
                    "p_value": c["p_value"],
                    "r_squared": res.get("r_squared"),
                }
            )
    return pd.DataFrame(rows)
