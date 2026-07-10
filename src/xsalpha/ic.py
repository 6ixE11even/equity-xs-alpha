"""Information coefficient analysis.

IC here = Spearman rank correlation between the signal cross-section at t
and forward returns t -> t+1. Monthly ICs overlap in information (signals
are autocorrelated), so plain t-stats overstate significance; we report
Newey-West adjusted t-stats instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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


def ic_summary(ics: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per signal: mean IC, IC IR (annualized), NW t-stat, hit rate."""
    rows = []
    for name, s in ics.items():
        s = s.dropna()
        rows.append(
            {
                "signal": name,
                "mean_ic": s.mean(),
                "ic_ir": s.mean() / s.std() * np.sqrt(12),
                "nw_tstat": newey_west_tstat(s),
                "hit_rate": (s > 0).mean(),
                "n_months": len(s),
            }
        )
    return pd.DataFrame(rows).set_index("signal").sort_values("nw_tstat", ascending=False)


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
