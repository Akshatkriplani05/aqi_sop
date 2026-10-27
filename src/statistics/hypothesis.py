"""Nonparametric hypothesis testing for distribution comparisons.

Pollutant distributions are skewed and heteroskedastic, so we use rank-based
tests that do not assume normality:
  * Mann-Whitney U  -> two independent groups.
  * Kruskal-Wallis  -> 3+ groups (omnibus), followed by pairwise Mann-Whitney
    with Holm-Bonferroni correction (controls family-wise error) as a post-hoc.

We always report an EFFECT SIZE alongside the p-value, because a p-value answers
"is there a detectable difference?" not "is the difference large/important?".
With large n, trivial differences become statistically significant.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Two-sample test + rank-biserial effect size."""
    da, db = a.dropna().to_numpy(), b.dropna().to_numpy()
    if len(da) < 3 or len(db) < 3:
        return {"U": np.nan, "p_value": np.nan, "rank_biserial": np.nan}
    U, p = stats.mannwhitneyu(da, db, alternative="two-sided")
    # Rank-biserial correlation = 1 - 2U/(n1*n2); magnitude in [0,1].
    rbc = 1.0 - (2.0 * U) / (len(da) * len(db))
    return {
        "U": float(U),
        "p_value": float(p),
        "rank_biserial": float(rbc),
        "n1": len(da),
        "n2": len(db),
    }


def kruskal_wallis(groups: dict[str, pd.Series]) -> dict[str, float]:
    """Omnibus test across groups + epsilon-squared effect size."""
    samples = [g.dropna().to_numpy() for g in groups.values() if len(g.dropna()) >= 3]
    if len(samples) < 2:
        return {"H": np.nan, "p_value": np.nan, "epsilon_squared": np.nan}
    H, p = stats.kruskal(*samples)
    n = sum(len(s) for s in samples)
    k = len(samples)
    # Epsilon-squared: (H - k + 1) / (n - k).
    eps2 = (H - k + 1) / (n - k) if n > k else np.nan
    return {"H": float(H), "p_value": float(p), "epsilon_squared": float(eps2), "k": k, "n": n}


def _holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        prev = max(prev, min(val, 1.0))
        adjusted[idx] = prev
    return adjusted.tolist()


def posthoc_pairwise(groups: dict[str, pd.Series], alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Mann-Whitney with Holm correction + effect sizes."""
    names = list(groups.keys())
    rows = []
    pairs = list(itertools.combinations(names, 2))
    raw_p = []
    for g1, g2 in pairs:
        res = mann_whitney(groups[g1], groups[g2])
        rows.append(
            {
                "group_1": g1,
                "group_2": g2,
                "U": res["U"],
                "p_raw": res["p_value"],
                "rank_biserial": res["rank_biserial"],
            }
        )
        raw_p.append(res["p_value"])
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = _holm_correction([p if not np.isnan(p) else 1.0 for p in raw_p])
        out["significant"] = out["p_holm"] < alpha
    return out
