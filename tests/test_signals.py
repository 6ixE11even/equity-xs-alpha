import numpy as np
import pandas as pd
import pytest

from xsalpha import signals


@pytest.fixture
def px():
    # 400 trading days, 40 names, geometric random walk with a seed so
    # tests are deterministic
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=400)
    rets = rng.normal(0.0004, 0.015, size=(400, 40))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"T{i}" for i in range(40)])


@pytest.fixture
def dates(px):
    return px.groupby(px.index.to_period("M")).tail(1).index


def test_momentum_skips_last_month(px, dates):
    mom = signals.mom_12_1(px, dates)
    # formation window ends 21 days before t: shocking the last 21 days
    # must NOT move the signal
    bumped = px.copy()
    bumped.iloc[-15:, 0] *= 1.5
    mom_bumped = signals.mom_12_1(bumped, dates)
    assert np.isclose(mom.iloc[-1, 0], mom_bumped.iloc[-1, 0], equal_nan=True)


def test_reversal_sign(px, dates):
    # a name that crashed last month should have a HIGH reversal signal
    crashed = px.copy()
    crashed.iloc[-21:, 0] *= 0.7
    sig = signals.strev_1m(crashed, dates)
    assert sig.iloc[-1, 0] > sig.iloc[-1, 1:].median()


def test_lowvol_prefers_quiet_names(px, dates):
    noisy = px.copy()
    rng = np.random.default_rng(0)
    noisy.iloc[:, 0] *= np.exp(np.cumsum(rng.normal(0, 0.05, 400)))
    sig = signals.lowvol_60d(noisy, dates)
    assert sig.iloc[-1, 0] < sig.iloc[-1, 1:].median()


def test_zscore_is_standard(px, dates):
    sig = signals.clean(signals.strev_1m(px, dates))
    row = sig.iloc[-1].dropna()
    assert abs(row.mean()) < 1e-9
    assert abs(row.std() - 1.0) < 1e-9


def test_winsorize_clips_outliers():
    df = pd.DataFrame([[0.0] * 20 + [100.0]])
    w = signals.winsorize_xs(df, k=3)
    assert w.iloc[0, -1] < 100.0
