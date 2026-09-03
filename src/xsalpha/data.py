"""Price/volume download and caching.

Everything downstream works off two wide DataFrames (date x ticker):
adjusted close and dollar volume. Cached to parquet so a full research
run doesn't hammer Yahoo every time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data")


def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adj_close, dollar_volume), both date x ticker.

    yfinance auto_adjust=True folds splits and dividends into Close,
    which is what we want for total-return-ish signals. Dollar volume
    uses the same adjusted close; good enough for an Amihud proxy.
    """
    cache_px = CACHE_DIR / "adj_close.parquet"
    cache_dv = CACHE_DIR / "dollar_vol.parquet"
    if cache and cache_px.exists() and cache_dv.exists():
        px_c, dv_c = pd.read_parquet(cache_px), pd.read_parquet(cache_dv)
        # The cache was returned for any request at all, so a panel built from
        # last year's universe answered a call for this year's, and a widened
        # ticker list came back with the old names and no warning.
        missing = set(tickers) - set(px_c.columns)
        # Compare against the requested start with a week of slack. The exact test
        # was `index.min() > start`, and since 2010-01-01 is a holiday the cached
        # panel always began on the 4th and always looked stale: the cache never once
        # hit, every run re-downloaded, and the ML combo's Sharpe moved by 0.09
        # between runs on nothing but Yahoo's latest adjustments. An ablation whose
        # effect is smaller than that is unmeasurable.
        slack = pd.Timedelta(days=7)
        stale = len(px_c) and px_c.index.min() > pd.Timestamp(start) + slack
        if not missing and not stale:
            return px_c[list(tickers)], dv_c[list(tickers)]
        print(f"  cache miss: {len(missing)} tickers absent"
              f"{', history starts late' if stale else ''} - refetching")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    px = raw["Close"].sort_index()
    vol = raw["Volume"].sort_index()
    # drop columns that came back completely empty (delisted/bad symbols)
    px = px.dropna(axis=1, how="all")
    vol = vol[px.columns]
    dollar_vol = px * vol

    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        px.to_parquet(cache_px)
        dollar_vol.to_parquet(cache_dv)
    return px, dollar_vol


def month_ends(px: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each month present in the index."""
    return px.groupby(px.index.to_period("M")).tail(1).index


def forward_returns(px: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One-period-ahead simple returns between consecutive rebalance dates.

    Row t holds the return earned from t to t+1; the last rebalance date
    has no forward return and is dropped by the caller.
    """
    p = px.loc[rebal_dates]
    return p.shift(-1) / p - 1.0
