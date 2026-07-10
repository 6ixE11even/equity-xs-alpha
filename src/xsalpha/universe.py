"""Investment universe construction.

Current S&P 500 members scraped from Wikipedia. This is a survivorship-biased
universe (see README - Limitations) but it's free and reproducible. Swap in
CRSP/Compustat point-in-time constituents if you have WRDS access.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

WIKI_SPX = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia 403s the default urllib user agent
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia.

    Yahoo uses '-' where the exchange uses '.' (BRK.B -> BRK-B).
    """
    req = urllib.request.Request(WIKI_SPX, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def filter_by_history(prices: pd.DataFrame, min_years: float = 3.0) -> pd.DataFrame:
    """Drop names with less than `min_years` of price history.

    Keeps the panel from being dominated by recent IPOs / spin-offs whose
    signals (12-month momentum especially) would be NaN most of the time.
    """
    min_obs = int(min_years * 252)
    counts = prices.notna().sum()
    keep = counts[counts >= min_obs].index
    return prices[keep]
