"""Information coefficient analysis.

IC here = Spearman rank correlation between the signal cross-section at t
and forward returns t -> t+1. Monthly ICs overlap in information (signals
are autocorrelated), so plain t-stats overstate significance; we report
Newey-West adjusted t-stats instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def ic_series(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Spearman IC per rebalance date. Needs >= 30 joint names to count."""
    out = {}
    for dt in signal.index:
        if dt not in fwd_ret.index:
            continue
        s = signal.loc[dt]
        r = fwd_ret.loc[dt]
        mask = s.notna() & r.notna()
        if mask.sum() < 30:
            continue
        out[dt] = spearmanr(s[mask], r[mask]).statistic
    return pd.Series(out, name="ic")


def newey_west_tstat(x: pd.Series, lags: int = 6) -> float:
    """t-stat of the mean with a Newey-West (Bartlett kernel) variance."""
    x = x.dropna().to_numpy()
    n = x.size
    if n < 10:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        var += 2.0 * w * (e[lag:] @ e[:-lag]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values: the FDR at which each test would be called.

    Seven signals are tested against the same returns, so the largest t-stat among
    them is not distributed like a single t-stat. Bonferroni would control the chance
    of *any* false positive and throw away most of the real ones; BH controls the
    expected share of false positives among the signals called significant, which is
    the question a research process actually asks. Harvey, Liu and Zhu (2016) make the
    case that the cross-sectional literature needs this and mostly does not do it.
    """
    p = pvalues.dropna().sort_values()
    m = len(p)
    q = (p.to_numpy() * m / np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]          # enforce monotonicity
    return pd.Series(np.minimum(q, 1.0), index=p.index).reindex(pvalues.index)


def ic_summary(ics: dict[str, pd.Series], start=None, end=None) -> pd.DataFrame:
    """One row per signal: mean IC, IC IR (annualized), NW t-stat, hit rate, FDR q.

    `start` and `end` restrict every signal to the same window. Comparing a signal
    that exists from 2016 against one that exists from 2010 on their own samples is
    not a comparison; it is two different decades.
    """
    rows = []
    for name, s in ics.items():
        s = s.dropna()
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        t = newey_west_tstat(s)
        rows.append(
            {
                "signal": name,
                "mean_ic": s.mean(),
                "ic_ir": s.mean() / s.std() * np.sqrt(12) if len(s) > 1 else np.nan,
                "nw_tstat": t,
                "p_value": 2.0 * (1.0 - norm.cdf(abs(t))) if np.isfinite(t) else np.nan,
                "hit_rate": (s > 0).mean() if len(s) else np.nan,
                "n_months": len(s),
            }
        )
    out = pd.DataFrame(rows).set_index("signal")
    out["fdr_q"] = benjamini_hochberg(out["p_value"])
    return out.sort_values("nw_tstat", ascending=False)


def ic_decay(signal: pd.DataFrame, px: pd.DataFrame, dates: pd.DatetimeIndex, horizons=(1, 2, 3, 6, 12)) -> pd.Series:
    """Mean IC of the signal against k-period-ahead returns, k in horizons.

    Tells you how fast the information dies - flat decay means low turnover
    can capture it, steep decay means you pay up in trading costs.
    """
    p = px.loc[dates]
    out = {}
    for k in horizons:
        fwd_k = p.shift(-k) / p - 1.0
        out[k] = ic_series(signal, fwd_k).mean()
    return pd.Series(out, name="mean_ic_by_horizon")
