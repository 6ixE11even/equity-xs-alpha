import numpy as np
import pandas as pd

from xsalpha import ic


def _panel(corr: float, n_dates=36, n_names=100, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    names = [f"T{i}" for i in range(n_names)]
    sig = rng.normal(size=(n_dates, n_names))
    noise = rng.normal(size=(n_dates, n_names))
    fwd = corr * sig + np.sqrt(1 - corr**2) * noise
    return (
        pd.DataFrame(sig, index=dates, columns=names),
        pd.DataFrame(fwd, index=dates, columns=names),
    )


def test_ic_recovers_known_correlation():
    sig, fwd = _panel(corr=0.3)
    ics = ic.ic_series(sig, fwd)
    # Spearman of jointly normal vars ~ (6/pi) * arcsin(rho/2) ~= 0.287 for rho=0.3
    assert 0.2 < ics.mean() < 0.4


def test_zero_signal_has_no_ic():
    sig, fwd = _panel(corr=0.0)
    ics = ic.ic_series(sig, fwd)
    assert abs(ics.mean()) < 0.05


def test_newey_west_downweights_autocorrelation():
    rng = np.random.default_rng(5)
    # AR(1) series: NW t-stat must be smaller than the naive t-stat
    e = rng.normal(size=500)
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.8 * x[t - 1] + e[t]
    s = pd.Series(x + 0.05)
    naive_t = s.mean() / (s.std() / np.sqrt(len(s)))
    nw_t = ic.newey_west_tstat(s, lags=10)
    assert nw_t < naive_t


def test_equal_weight_combo_aligns_on_tickers_not_positions():
    """np.nanmean over .to_numpy() averaged by column position, so a signal whose
    columns came back in a different order blended into the wrong tickers."""
    from xsalpha.ml import equal_weight_combo

    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    a = pd.DataFrame(1.0, index=idx, columns=["AAA", "BBB", "CCC"])
    b = pd.DataFrame([[10.0, 20.0, 30.0]] * 3, index=idx, columns=["CCC", "BBB", "AAA"])
    combo = equal_weight_combo({"a": a, "b": b})
    assert abs(combo.loc[idx[0], "AAA"] - 15.5) < 1e-12
    assert abs(combo.loc[idx[0], "CCC"] - 5.5) < 1e-12
