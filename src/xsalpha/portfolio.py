"""Quintile portfolio construction with a linear transaction-cost model.

Long-short spread: long the top quintile of the signal, short the bottom,
equal weight inside each leg, rebalanced at every signal date. Costs are
charged as one-way turnover * cost_bps on both legs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quintile_weights(signal_row: pd.Series, q: int = 5) -> pd.Series:
    """Equal-weight top-minus-bottom quintile weights for one date.

    Bucketing by qcut on a first-tie-broken rank, which is what
    `quintile_mean_returns` already did. Thresholding a percentile rank instead
    broke in two ways: `ranks >= 1 - 1/q` caught 21 names against the short leg's
    20 at n=100, leaving the "market-neutral" spread with a standing long tilt;
    and average-ranked ties put every name on one side of the cut, so a signal
    taking 60/40 values produced 40 longs, no shorts, and -1/0 = -inf weights.
    """
    s = signal_row.dropna()
    if len(s) < q * 4:  # too few names for a meaningful sort
        return pd.Series(dtype=float)
    if s.nunique() == 1:
        # first-tie-breaking would sort by whatever order the names arrived in,
        # which is a position, not a view.
        return pd.Series(dtype=float)
    buckets = pd.qcut(s.rank(method="first"), q, labels=False)
    long = buckets == q - 1
    short = buckets == 0
    if not long.any() or not short.any():
        return pd.Series(dtype=float)
    w = pd.Series(0.0, index=s.index)
    w[long] = 1.0 / int(long.sum())
    w[short] = -1.0 / int(short.sum())
    return w


def backtest_ls(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    cost_bps: float = 10.0,
    q: int = 5,
) -> pd.DataFrame:
    """Run the long-short quintile backtest.

    Returns a DataFrame indexed by rebalance date with columns:
    gross_ret, net_ret, turnover (one-way, both legs summed).
    """
    dates = [d for d in signal.index if d in fwd_ret.index]
    prev_w = pd.Series(dtype=float)
    rows = []
    for dt in dates:
        w = quintile_weights(signal.loc[dt], q=q)
        if w.empty:
            continue
        r = fwd_ret.loc[dt].reindex(w.index)
        gross = float((w * r).sum())
        # turnover vs previous weights (union of holdings)
        union = w.index.union(prev_w.index)
        tw = w.reindex(union, fill_value=0.0)
        pw = prev_w.reindex(union, fill_value=0.0)
        turnover = float((tw - pw).abs().sum()) / 2.0
        net = gross - turnover * cost_bps / 1e4
        rows.append({"date": dt, "gross_ret": gross, "net_ret": net, "turnover": turnover})
        prev_w = w
    return pd.DataFrame(rows).set_index("date")


def perf_stats(rets: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    r = rets.dropna()
    if r.empty:
        return {}
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    curve = (1 + r).cumprod()
    dd = (curve / curve.cummax() - 1.0).min()
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else np.nan,
        "max_dd": float(dd),
        "n_periods": int(len(r)),
    }


def quintile_mean_returns(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: int = 5) -> pd.Series:
    """Average forward return per quintile bucket - monotonicity check."""
    buckets: dict[int, list[float]] = {i: [] for i in range(1, q + 1)}
    for dt in signal.index:
        if dt not in fwd_ret.index:
            continue
        s = signal.loc[dt].dropna()
        r = fwd_ret.loc[dt].reindex(s.index)
        if len(s) < q * 4:
            continue
        labels = pd.qcut(s.rank(method="first"), q, labels=False) + 1  # 1 = bottom
        for i in range(1, q + 1):
            buckets[i].append(r[labels == i].mean())
    return pd.Series({f"Q{i}": np.nanmean(v) for i, v in buckets.items()})
