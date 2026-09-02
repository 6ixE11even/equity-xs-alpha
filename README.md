# equity-xs-alpha

Cross-sectional equity alpha research, end to end: signal construction → information
coefficient analysis → cost-aware long-short portfolios → ML signal combination under
purged walk-forward cross-validation.

The point of this repo is not "here is a Sharpe 3 strategy" (it isn't, and single-name
monthly factors on the S&P 500 shouldn't be). The point is the **research process**:
every number is out-of-sample where it claims to be, significance is measured with
autocorrelation-robust errors, and costs are charged before anything is called
performance.

```
pip install -e ".[dev]"
python scripts/run_research.py      # full run: download -> signals -> IC -> portfolios -> ML
pytest                              # 16 tests on synthetic data
```

Results land in `reports/results.md` and `reports/figures/`.

## Pipeline

```
universe.py    S&P 500 members (Wikipedia scrape), >=3y history filter
data.py        yfinance daily adjusted close + dollar volume, parquet cache
signals.py     6 signals, winsorized (median +/- 3*1.4826*MAD) and z-scored per date
ic.py          Spearman rank IC, Newey-West t-stats, IC decay by horizon
portfolio.py   quintile long-short, equal weight per leg, linear cost model
cv.py          purged walk-forward folds with embargo
ml.py          LightGBM on cross-sectional rank labels, OOS scores only
```

## Signals

All signals are oriented so that **higher = higher expected forward return**, sampled
at month-end, and cleaned cross-sectionally. With daily returns $r_{i,t}$ and prices
$P_{i,t}$:

**Momentum (12-1)** — Jegadeesh & Titman (1993). Formation over months $t-12 \dots t-1$,
skipping the most recent month to avoid contamination by short-term reversal:

$$\text{MOM}_{i,t} = \prod_{s=t-252}^{t-21}(1+r_{i,s}) - 1$$

**Short-term reversal** — Jegadeesh (1990). Last month's losers outperform:

$$\text{STREV}_{i,t} = -\left(P_{i,t}/P_{i,t-21} - 1\right)$$

**Low volatility** — Ang, Hodrick, Xing & Zhang (2006). Negative of 60-day realized
daily-return standard deviation.

**MAX effect** — Bali, Cakici & Whitelaw (2011). Lottery-like names underperform; the
signal is the negative of the mean of the 5 largest daily returns over the last 21 days.

**Amihud illiquidity** — Amihud (2002). Illiquidity premium, log-compressed because the
raw measure spans orders of magnitude even inside the S&P 500:

$$\text{ILLIQ}_{i,t} = \log\left(\frac{1}{63}\sum_{s=t-62}^{t}\frac{|r_{i,s}|}{\text{DollarVol}_{i,s}}\right)$$

**Skewness** — negative of 120-day return skewness (investors overpay for positive skew).

Cleaning per cross-section: winsorize at median $\pm 3 \times 1.4826 \times \text{MAD}$
(the 1.4826 factor makes MAD consistent with $\sigma$ under normality), then z-score.

## Methodology

### Information coefficient

$$\text{IC}_t = \text{corr}_{\text{Spearman}}\left(\text{signal}_{i,t},\; r_{i,t \to t+1}\right)$$

Monthly ICs are autocorrelated (signals overlap heavily month to month), so the
reported t-stat on the mean IC uses a **Newey-West (1987)** variance with a Bartlett
kernel at 6 lags:

$$\hat{V} = \hat\gamma_0 + 2\sum_{\ell=1}^{L}\left(1-\tfrac{\ell}{L+1}\right)\hat\gamma_\ell,
\qquad t = \frac{\overline{\text{IC}}}{\sqrt{\hat V / n}}$$

**IC decay** re-computes the mean IC against $k$-month-forward returns,
$k \in \{1,2,3,6,12\}$. Flat decay means the signal survives slow rebalancing; steep
decay means turnover (and therefore costs) is structural.

### Portfolios and costs

Top-minus-bottom quintile, equal weight inside each leg, dollar neutral, monthly
rebalance. Costs are linear: one-way turnover $\times$ 10 bps, charged every rebalance:

$$r^{\text{net}}_t = w_t^\top r_{t\to t+1} - \frac{\lVert w_t - w_{t-1}\rVert_1}{2}\cdot\text{10bps}$$

10 bps one-way is conservative-realistic for S&P 500 names at institutional size; the
cost parameter is a single argument if you want to stress it.

### Purged walk-forward CV

Random K-fold leaks here: the label at $t$ is the return over $(t, t+1]$, so adjacent
train/test samples share information (López de Prado, *Advances in Financial ML*, ch. 7).
The splitter in `cv.py` trains on an expanding window, **embargoes** one period between
train end and test start (removing the one-period label overlap), and predicts strictly
forward — 60 months minimum training, 12-month test blocks, rolled to the end of sample.

The ML model is deliberately boring: shallow LightGBM (depth 4, 15 leaves, heavy
subsampling) regressing the **cross-sectional percentile rank** of next-month return on
the 6 signals. Rank labels keep the target stationary across vol regimes. The model's
OOS score is then treated exactly like any raw signal — same z-scoring, same quintile
portfolio, same costs — so the comparison against the equal-weight signal combo is fair.

## Results

<!-- RESULTS:BEGIN -->
Regenerated 2026-09-02 with `uv run python scripts/run_research.py`. 494 names,
2010-01 to 2026-09, monthly rebalance, 10 bps one-way. Full tables in
`reports/results.md`. (503 tickers scraped; PLTR failed to download and eight more
had under three years of history.)

**Standalone signal ICs** (full sample, Newey-West t-stats at 6 lags):

| signal | mean IC | IC IR | NW t | hit rate | months |
|---|---|---|---|---|---|
| amihud | 0.032 | 1.14 | **5.36** | 63% | 197 |
| mom_12_1 | 0.008 | 0.14 | 0.67 | 55% | 188 |
| strev_1m | 0.006 | 0.13 | 0.61 | 48% | 199 |
| skew_120d | -0.005 | -0.18 | -0.74 | 46% | 195 |
| max5 | -0.015 | -0.26 | -1.17 | 48% | 199 |
| lowvol_60d | -0.023 | -0.33 | -1.51 | 47% | 198 |

Only the illiquidity premium clears significance standalone. Momentum and short-term
reversal are directionally right but weak in this sample; low-vol and the lottery
signals actually inverted over 2010-2026 in S&P 500 names — a useful reminder that
published premia are regime- and universe-dependent.

**Long-short quintile portfolios, net of costs** (OOS window = the ML model's
walk-forward test period, 2015-02 onward, 139 months):

| portfolio | ann. ret | ann. vol | Sharpe | max DD | avg turnover |
|---|---|---|---|---|---|
| amihud | 8.7% | 8.1% | **1.07** | -13.5% | 0.14 |
| **ml_combo** | **6.6%** | **8.6%** | **0.77** | **-9.8%** | 1.12 |
| mom_12_1 | 1.9% | 15.9% | 0.12 | -36.3% | 0.46 |
| strev_1m | -3.9% | 13.6% | -0.29 | -57.3% | 1.55 |
| skew_120d | -2.5% | 7.6% | -0.33 | -40.4% | 0.52 |
| equal_weight_combo | -8.7% | 15.0% | -0.58 | -74.1% | 1.03 |
| lowvol_60d | -15.0% | 19.4% | -0.77 | -87.9% | 0.45 |
| max5 | -14.5% | 17.2% | -0.84 | -86.4% | 1.13 |

The interesting result is `equal_weight_combo` against `ml_combo`. Naively averaging
six z-scored signals — three of which have negative ICs — loses 8.7% a year, while the
purged walk-forward LightGBM learns out-of-sample which signals carry information and
lands within 2 points of the best standalone signal at three quarters of its drawdown.
The model isn't manufacturing alpha; it's doing signal selection honestly.

Note the turnover column next to it. `amihud` earns its Sharpe on 0.14 turnover;
`ml_combo` needs 1.12 to earn less. At 10 bps that gap is affordable and at 50 bps it
is not, which is the more useful thing to know about the ML model than its Sharpe.

<sub>These numbers moved slightly from the previously committed table (amihud Sharpe
1.03 → 1.07, ml_combo 0.89 → 0.77). Three reasons, in order of size: the sample now
runs two months longer and prices are re-adjusted; the quintile legs used to be
unbalanced, taking 21 long names against 20 short; and cross-sections whose MAD was
zero used to be dropped entirely by the winsorizer. See the commit history.</sub>
<!-- RESULTS:END -->

![IC summary](reports/figures/ic_summary.png)
![Equity curves](reports/figures/equity_curves.png)

## Limitations (read this before quoting numbers)

- **Survivorship bias.** The universe is *today's* S&P 500 members extended backward.
  Names that were removed (bankruptcies, acquisitions, deletions) never enter the
  panel. This flatters long legs and dampens short legs — real point-in-time
  constituents (CRSP/Compustat via WRDS) are the correct fix and the code is structured
  so only `universe.py` needs to change.
- **No delisting returns**, same direction of bias as above.
- **Adjusted-close total returns** from Yahoo fold dividends into price; good enough
  for rank signals, not for precise level accounting.
- **Costs are linear.** No market impact, no borrow fees on the short leg.
- Monthly rebalance only; the IC-decay table hints at what faster rebalancing would
  buy, but intraday/weekly execution is out of scope here.

## References

Amihud (2002) · Ang, Hodrick, Xing, Zhang (2006) · Bali, Cakici, Whitelaw (2011) ·
Jegadeesh (1990) · Jegadeesh & Titman (1993) · Newey & West (1987) ·
López de Prado (2018), *Advances in Financial Machine Learning*

## License

MIT
