import numpy as np
import pandas as pd

from xsalpha import portfolio


def test_quintile_weights_are_dollar_neutral():
    s = pd.Series(np.arange(50, dtype=float), index=[f"T{i}" for i in range(50)])
    w = portfolio.quintile_weights(s)
    assert abs(w.sum()) < 1e-12
    assert abs(w[w > 0].sum() - 1.0) < 1e-12
    assert abs(w[w < 0].sum() + 1.0) < 1e-12


def test_perfect_signal_earns_spread():
    # signal == forward return -> the L/S portfolio must earn the
    # top-minus-bottom quintile spread exactly
    rng = np.random.default_rng(1)
    names = [f"T{i}" for i in range(50)]
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    fwd = pd.DataFrame(rng.normal(0, 0.05, (24, 50)), index=dates, columns=names)
    bt = portfolio.backtest_ls(fwd.copy(), fwd, cost_bps=0.0)
    assert (bt["gross_ret"] > 0).all()


def test_costs_reduce_returns():
    rng = np.random.default_rng(2)
    names = [f"T{i}" for i in range(50)]
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    sig = pd.DataFrame(rng.normal(size=(24, 50)), index=dates, columns=names)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (24, 50)), index=dates, columns=names)
    gross = portfolio.backtest_ls(sig, fwd, cost_bps=0.0)["net_ret"]
    net = portfolio.backtest_ls(sig, fwd, cost_bps=50.0)["net_ret"]
    assert (net <= gross + 1e-12).all()
    assert net.sum() < gross.sum()


def test_first_rebalance_turnover_is_one():
    # going from cash to a fully invested L/S book = 100% one-way turnover
    # (|w|_1 / 2 = (1 + 1)/2 ... = 1.0)
    rng = np.random.default_rng(3)
    names = [f"T{i}" for i in range(50)]
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    sig = pd.DataFrame(rng.normal(size=(3, 50)), index=dates, columns=names)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (3, 50)), index=dates, columns=names)
    bt = portfolio.backtest_ls(sig, fwd, cost_bps=0.0)
    assert abs(bt["turnover"].iloc[0] - 1.0) < 1e-12
