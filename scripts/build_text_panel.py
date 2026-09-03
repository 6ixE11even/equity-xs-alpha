"""
Score every S&P 500 filing on EDGAR and write a tidy tone table.

    uv run python scripts/build_text_panel.py --since 2015-01-01

One row per filing: ticker, filing date, which section was scored, and the
Loughran-McDonald category frequencies. Turning that into a signal happens later
in `signals.py`; this script only does the slow part, and it caches so it only
does it once.

Runtime is set by SEC's rate limit, not by the CPU. Roughly nine requests a second
is the ceiling they publish, so twenty thousand filings takes the better part of
an hour. The score table is a few megabytes; the filings behind it are ten
gigabytes, which is why only the scores are kept.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xsalpha import universe                                  # noqa: E402
from xsalpha.edgar import filing_index, filing_text, ticker_to_cik  # noqa: E402
from xsalpha.text import scored_text, tone                    # noqa: E402

OUT = ROOT / "data" / "edgar" / "tone_scores.parquet"


def main(since: str, limit: int | None, forms: tuple[str, ...]) -> None:
    tickers = universe.sp500_tickers()
    if limit:
        tickers = tickers[:limit]
    cik = ticker_to_cik()
    known = [t for t in tickers if t in cik]
    print(f"universe {len(tickers)} tickers, {len(known)} resolve to a CIK on EDGAR")

    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    seen = set(done["accession"]) if len(done) else set()

    rows, t0, filings = list(done.to_dict("records")), time.time(), 0
    for i, tk in enumerate(known, 1):
        try:
            idx = filing_index(cik[tk], forms=forms)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {tk}: index failed ({exc})")
            continue
        idx = idx[idx["filingDate"] >= since]

        for f in idx.itertuples():
            if f.accessionNumber in seen:
                continue
            try:
                raw = filing_text(cik[tk], f.accessionNumber, f.primaryDocument)
            except Exception:                          # noqa: BLE001
                continue                                # a pulled document, not a crash
            section, scope = scored_text(raw)
            rows.append({"ticker": tk, "cik": cik[tk], "form": f.form,
                         "filing_date": f.filingDate, "accession": f.accessionNumber,
                         "scope": scope, **tone(section)})
            filings += 1

        if i % 25 == 0 or i == len(known):
            el = time.time() - t0
            print(f"  [{i:>3}/{len(known)}] {tk:<6} {filings:>6,} filings  "
                  f"{el/60:>5.1f} min  {filings/max(el,1):.1f}/s")
            pd.DataFrame(rows).to_parquet(OUT, index=False)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df):,} filings, {df.ticker.nunique()} tickers)")
    print(df["scope"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--limit", type=int, default=None, help="first N tickers, for a smoke run")
    ap.add_argument("--forms", default="10-K,10-Q")
    a = ap.parse_args()
    main(a.since, a.limit, tuple(a.forms.split(",")))
