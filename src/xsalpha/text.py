"""
Tone signals from SEC filings.

The factor is *change* in tone, not tone. Loughran and McDonald (2011) is clear on
why: the level of negative-word usage is mostly a fixed effect of the company and
its industry — a biotech's risk section reads alarming every single year — while
the change against that company's own previous filing is the part that carries
information. So every score below is differenced against the same CIK's prior
filing before it becomes a signal.

Two other things this file exists to get right.

*The dictionary is downloaded, not embedded.* Sentiment on financial text needs a
financial lexicon: "liability", "depreciation" and "restructuring" are negative in
a general-purpose word list and neutral-to-meaningless in a 10-K. The
Loughran-McDonald master dictionary is the standard, it is maintained at Notre
Dame, and a copy pasted into source would be a snapshot that silently ages.

*The table of contents is not the document.* Every 10-K names its own sections in
a contents block near the top, so a naive search for "Item 1A" lands on the index
rather than the text. Worse, filings cross-reference themselves constantly - Apple's
2025 10-K says "see Item 1A of this Form 10-K" five separate times - so a section
that ends at the next "Item" match ends about two hundred characters in. A section
runs to the next heading with a genuinely *higher* item number.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Loughran-McDonald master dictionary, from the file Notre Dame links on
# sraf.nd.edu. The plain drive.google.com/uc endpoint now returns an HTML quota
# page instead of the file, and happily returns it with a 200, so the download is
# checked for size and header before it is cached.
LM_FILE_ID = "1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_URL = f"https://drive.usercontent.google.com/download?id={LM_FILE_ID}&export=download"
LM_MIN_BYTES = 1_000_000
LM_CACHE = Path("data/edgar/lm_master_dictionary.csv")

# Apostrophes and ampersands do not survive tag-stripping, so headings are matched
# loosely. \W* between words absorbs whatever the stripper left behind.
# (start heading, the headings that actually close it). Form 10-K fixes the order,
# so the successor is known rather than guessed: Item 1A is followed by 1B, and
# Item 7 by 7A. Searching for "the next higher item number" does not work - Apple's
# risk-factor section refers to Item 7 within its first 800 characters, and a naive
# boundary cuts the section off there.
_SECTIONS = {
    "risk_factors": (r"item\W{0,4}1A\W{0,4}risk\W{0,4}factors",
                     r"item\W{0,4}(?:1B\W{0,4}unresolved|2\W{0,4}propert)"),
    "mdna": (r"item\W{0,4}7\W{0,4}management\W{0,4}s?\W{0,4}discussion",
             r"item\W{0,4}(?:7A\W{0,4}quantitative|8\W{0,4}financial\W{0,4}statements)"),
}
_ITEM = re.compile(r"item\W{0,4}(\d+)([AB]?)\W", re.I)
_WORD = re.compile(r"[A-Za-z']+")

# A heading found in the first few percent of a filing is the contents block.
_TOC_FRACTION = 0.15
# Below this many words a "section" is a cross-reference, not prose.
MIN_SECTION_WORDS = 200
# Above this, the closing heading was missed and the "section" is most of the
# filing. JPMorgan's risk factors came out at 150,000 words on the first pass.
MAX_SECTION_WORDS = 60_000


@lru_cache(maxsize=1)
def lm_dictionary() -> dict[str, set[str]]:
    """Loughran-McDonald word lists, keyed by category.

    The file marks a word as Negative/Positive/Uncertainty/Litigious with the year
    it entered the list, or 0 if it is not in that category.
    """
    if not LM_CACHE.exists():
        LM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(LM_URL, timeout=180,
                         headers={"User-Agent": "Mozilla/5.0 (Macintosh) Chrome/124"})
        r.raise_for_status()
        if len(r.content) < LM_MIN_BYTES or r.content[:15].lower().startswith(b"<!doctype"):
            raise RuntimeError(
                f"the dictionary download returned {len(r.content)} bytes of "
                f"{r.headers.get('content-type')} rather than the CSV - Drive serves a "
                f"quota page with a 200, so this has to be checked. Fetch it by hand "
                f"from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ into "
                f"{LM_CACHE}.")
        LM_CACHE.write_bytes(r.content)

    df = pd.read_csv(LM_CACHE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    out = {}
    for cat in ("negative", "positive", "uncertainty", "litigious",
                "strong_modal", "weak_modal", "constraining"):
        if cat in df.columns:
            out[cat] = set(df.loc[df[cat] > 0, "word"].str.lower())
    return out


def extract_section(text: str, section: str) -> str:
    """Pull one named section out of a filing, skipping the table of contents."""
    pattern, closer = _SECTIONS[section]
    hits = [m.start() for m in re.finditer(pattern, text, re.I)]
    if not hits:
        return ""

    # Drop headings that sit inside the contents block at the head of the filing.
    body = [h for h in hits if h > len(text) * _TOC_FRACTION] or hits[-1:]

    best = ""
    for start in body:
        tail = text[start:]
        nxt = re.search(closer, tail[100:], re.I)   # step past the heading itself
        chunk = tail[: 100 + nxt.start()] if nxt else tail
        words = len(chunk.split())
        if MIN_SECTION_WORDS <= words <= MAX_SECTION_WORDS and words > len(best.split()):
            best = chunk
    return best


def scored_text(filing: str) -> tuple[str, str]:
    """The best available prose from a filing, and a label for what it is.

    Section extraction works on a plain industrial 10-K and does not work
    everywhere. Banks are the obvious case: JPMorgan incorporates its MD&A by
    reference to the annual report, so Item 7 in the 10-K is a pointer, and its
    risk-factor section ran to 150,000 words before a length bound went in.

    Rather than drop those filers, fall back to the whole filing and say so. Tone
    over a complete 10-K is what Loughran and McDonald score in the original
    paper; the section cut is a refinement, not a prerequisite. The label travels
    with the score so the analysis can check whether the two behave differently
    instead of quietly mixing them.
    """
    for name in ("risk_factors", "mdna"):
        section = extract_section(filing, name)
        n = len(section.split())
        if MIN_SECTION_WORDS <= n <= MAX_SECTION_WORDS:
            return section, name
    return filing, "full_filing"


def tone(text: str) -> dict[str, float]:
    """Loughran-McDonald category frequencies, as a share of total words.

    Returned as proportions rather than counts: filings vary in length by an order
    of magnitude, and a raw count is mostly measuring how much the company wrote.
    """
    lm = lm_dictionary()
    words = [w.lower() for w in _WORD.findall(text)]
    n = len(words)
    if n == 0:
        return {c: np.nan for c in lm} | {"n_words": 0.0, "net_tone": np.nan}

    counts = {cat: sum(w in vocab for w in words) for cat, vocab in lm.items()}
    out = {cat: c / n for cat, c in counts.items()}
    out["n_words"] = float(n)
    # Positive minus negative, the conventional net-tone reading.
    out["net_tone"] = out.get("positive", 0.0) - out.get("negative", 0.0)
    return out


def tone_change_panel(scores: pd.DataFrame, column: str = "net_tone",
                      by_form: bool = True) -> pd.DataFrame:
    """Turn per-filing tone into a date x ticker panel of *changes*.

    `scores` needs ticker, filing_date, form and the tone column. Each value is the
    difference against that ticker's own previous filing, so a company that simply
    writes more darkly than its peers contributes nothing until its own tone moves.

    `by_form` differences a 10-K against the previous 10-K and a 10-Q against the
    previous 10-Q. It defaults on because ignoring it does not add noise, it adds a
    signal - the wrong one. In this panel a 10-K's median net tone is -0.0164 against
    a 10-Q's -0.0097, and the annual report is 1.6x longer, so an annual report reads
    darker than a quarterly for reasons of genre rather than of business. Differencing
    across the boundary gives every Q-to-K transition a mean change of -0.0081 and
    every K-to-Q transition +0.0086, while a Q-to-Q change averages -0.0001. The
    signal becomes a calendar sawtooth with a standard deviation four times the real
    within-form move, and every company in the index files on the same calendar, so
    it does not even wash out cross-sectionally.
    """
    df = scores.sort_values(["ticker", "filing_date"]).copy()
    keys = ["ticker", "form"] if (by_form and "form" in df.columns) else ["ticker"]
    df["delta"] = df.groupby(keys)[column].diff()
    panel = df.pivot_table(index="filing_date", columns="ticker", values="delta")
    return panel.sort_index()


def as_monthly_signal(panel: pd.DataFrame, dates: pd.DatetimeIndex,
                      stale_after_days: int = 120, publication_lag_days: int = 1
                      ) -> pd.DataFrame:
    """Carry each filing's tone change forward to the month-end rebalance dates.

    A filing is information from the day it lands until it goes stale; holding it
    forever would let a 2015 10-K keep voting in 2026. `stale_after_days` is a
    quarter plus a fortnight, which is roughly the gap between filings.

    `publication_lag_days` shifts every filing one day later before it is allowed to
    vote. EDGAR accepts filings until 22:00 Eastern and stamps them with that day's
    date, so a filing dated the 31st can arrive after the close on the 31st. Trading
    a month-end signal on a filing stamped that same day is a look-ahead of exactly
    one day, which is small, invisible in the output, and still a look-ahead.
    """
    if panel.empty:
        return pd.DataFrame(index=dates)
    if publication_lag_days:
        panel = panel.copy()
        panel.index = panel.index + pd.Timedelta(days=publication_lag_days)
    filled = panel.reindex(panel.index.union(dates)).ffill()
    # For each cell, the date of the filing whose value is currently being carried.
    # Subtracting that from the rebalance date gives the value's age. The units have to
    # be pinned: an un-united datetime64("NaT") makes the subtraction produce a
    # generic timedelta, which numpy now warns about and will later refuse.
    stamp_values = np.where(panel.notna(),
                            panel.index.values.astype("datetime64[ns]")[:, None],
                            np.datetime64("NaT", "ns"))
    stamp = pd.DataFrame(stamp_values, index=panel.index, columns=panel.columns)
    stamp = stamp.reindex(filled.index).ffill()
    now = filled.index.values.astype("datetime64[ns]")[:, None]
    age_days = (now - stamp.to_numpy().astype("datetime64[ns]")) / np.timedelta64(1, "D")

    fresh = filled.where(age_days <= stale_after_days)
    return fresh.reindex(dates)
