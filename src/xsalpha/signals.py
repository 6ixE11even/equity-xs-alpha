"""Signal definitions.

Every signal is computed from daily data and sampled at month-end
rebalance dates, then cleaned cross-sectionally (winsorize + z-score).
Sign convention: HIGHER signal value = HIGHER expected forward return,
so a long-short portfolio is always long the top quintile.

Signals implemented (see README for the math and references):
    mom_12_1   12-month momentum, skipping the most recent month
    strev_1m   short-term reversal (negative of last month's return)
    lowvol_60d low volatility (negative of 60d realized vol)
    max5       lottery effect (negative of mean of 5 largest daily returns, 21d)
    amihud     illiquidity premium (Amihud 2002), 63d window
    skew_120d  negative coskewness proxy (negative of 120d return skewness)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mom_12_1(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    r = px.pct_change(fill_method=None)
    # cumulative return over t-252..t-21: classic 12-1 formation window
    cum = (1 + r).rolling(252 - 21).apply(np.prod, raw=True) - 1.0
    return cum.shift(21).loc[dates]


def strev_1m(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    r21 = px.pct_change(21, fill_method=None)
    return -r21.loc[dates]


def lowvol_60d(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    vol = px.pct_change(fill_method=None).rolling(60).std()
    return -vol.loc[dates]


def max5(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    r = px.pct_change(fill_method=None)

    def top5_mean(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        if a.size < 5:
            return np.nan
        return np.sort(a)[-5:].mean()

    m = r.rolling(21).apply(top5_mean, raw=True)
    return -m.loc[dates]


def amihud(px: pd.DataFrame, dollar_vol: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    r = px.pct_change(fill_method=None)
    illiq = (r.abs() / dollar_vol.replace(0, np.nan)).rolling(63).mean()
    # log-compress: raw Amihud spans orders of magnitude even inside the S&P 500
    return np.log(illiq).loc[dates]


def skew_120d(px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    sk = px.pct_change(fill_method=None).rolling(120).skew()
    return -sk.loc[dates]


# ── cross-sectional cleaning ──────────────────────────────────────────


def winsorize_xs(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Clip each row (cross-section) at median +/- k * MAD.

    A cross-section where more than half the names share a value has MAD = 0, and
    the clip then flattened the whole row onto the median - taking with it the
    dispersion the signal was there to measure, and leaving z-scoring to divide by
    zero and return NaN for the date. Those rows fall back to a standard-deviation
    scale, which is less robust but still separates the names. When even that is
    zero the row genuinely carries no information and is left alone.
    """
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0)).abs().median(axis=1)
    scale = k * 1.4826 * mad
    fallback = k * df.std(axis=1)
    scale = scale.where(scale > 0, fallback)
    degenerate = ~(scale > 0)
    lo = (med - scale).mask(degenerate, -np.inf)
    hi = (med + scale).mask(degenerate, np.inf)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    # A row with no dispersion carries no cross-sectional information. Dividing by
    # zero would give inf where NaN - "this date has no signal" - is the truth.
    return df.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return zscore_xs(winsorize_xs(df))


def build_signal_panel(
    px: pd.DataFrame, dollar_vol: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    """Compute all signals, cleaned, keyed by name."""
    raw = {
        "mom_12_1": mom_12_1(px, dates),
        "strev_1m": strev_1m(px, dates),
        "lowvol_60d": lowvol_60d(px, dates),
        "max5": max5(px, dates),
        "amihud": amihud(px, dollar_vol, dates),
        "skew_120d": skew_120d(px, dates),
    }
    return {name: clean(df) for name, df in raw.items()}
