"""
SEC EDGAR client.

Everything here comes from EDGAR itself — the ticker-to-CIK map, the filing index
per company, and the filing documents. Nothing is embedded in the source: the CIK
map alone changes every time a company lists, delists or renames, and a hardcoded
copy is wrong within a month.

SEC asks for a real User-Agent with a contact address and rate-limits to 10
requests a second. Both are honoured below; going over gets the IP blocked, and
they mean it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

SEC_UA = "Tejas Pandya tbp8777@nyu.edu"
TICKER_MAP = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

CACHE_DIR = Path("data/edgar")
MIN_INTERVAL = 0.11          # ~9 req/s, just under SEC's stated ceiling

_last_call = 0.0


def _get(url: str, **kw) -> requests.Response:
    """Rate-limited GET. SEC blocks on burst, so every call goes through here."""
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"},
                     timeout=30, **kw)
    _last_call = time.monotonic()
    r.raise_for_status()
    return r


def ticker_to_cik(refresh: bool = False) -> dict[str, int]:
    """Map ticker -> CIK, straight from SEC's own published file."""
    cache = CACHE_DIR / "company_tickers.json"
    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text())
    else:
        raw = _get(TICKER_MAP).json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}


def filing_index(cik: int, forms=("10-K", "10-Q")) -> pd.DataFrame:
    """Every 10-K/10-Q EDGAR lists for one company, newest first.

    `submissions` returns the recent filings inline and older ones in separate
    files; both are read, because a 2010 start needs the overflow.
    """
    cache = CACHE_DIR / "index" / f"{cik}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    j = _get(SUBMISSIONS.format(cik=cik)).json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    # A reorganised registrant lists its predecessor here; without walking these
    # the history stops at the reorganisation date.
    for extra in j["filings"].get("files", []):
        frames.append(pd.DataFrame(_get(f"https://data.sec.gov/submissions/{extra['name']}").json()))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(forms)][["form", "filingDate", "reportDate",
                                     "accessionNumber", "primaryDocument"]]
    df["cik"] = cik
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df = df.sort_values("filingDate", ascending=False).reset_index(drop=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


CACHE_TEXT = False   # filings average ~500KB; 20k of them is 10GB for no gain


def filing_text(cik: int, accession: str, document: str) -> str:
    """Plain text of one filing. Cached — these are 1-5MB each and immutable.

    The archive path keys on the CIK that *filed* the document, which is not
    always the CIK the ticker maps to today. Exxon reorganised: XOM now resolves
    to CIK 2115436 with a single filing to its name, while twenty years of 10-Ks
    sit under the predecessor 34088. The accession number carries the filer's CIK
    in its first ten digits, so take it from there rather than from the argument.
    """
    filer = int(accession.split("-")[0])
    cik = filer or cik
    acc = accession.replace("-", "")
    cache = CACHE_DIR / "text" / str(cik) / f"{acc}.txt"
    if CACHE_TEXT and cache.exists():
        return cache.read_text()

    html = _get(ARCHIVE.format(cik=cik, acc=acc, doc=document)).text
    # Scripts and styles carry no prose and plenty of words that look like words.
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = _WS.sub(" ", _ENTITY.sub(" ", _TAG.sub(" ", html))).strip()

    if CACHE_TEXT:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text
