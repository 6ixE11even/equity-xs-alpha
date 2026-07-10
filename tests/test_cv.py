import pandas as pd

from xsalpha.cv import walk_forward_folds


def _dates(n=120):
    return pd.date_range("2015-01-31", periods=n, freq="ME")


def test_no_overlap_between_train_and_test():
    for fold in walk_forward_folds(_dates(), min_train=60, test_size=12, embargo=1):
        assert fold.train_dates.max() < fold.test_dates.min()


def test_embargo_gap():
    dates = _dates()
    for fold in walk_forward_folds(dates, min_train=60, test_size=12, embargo=2):
        gap = list(dates).index(fold.test_dates[0]) - list(dates).index(fold.train_dates[-1])
        assert gap >= 2  # embargo periods sit strictly between train and test


def test_folds_cover_forward_only():
    folds = walk_forward_folds(_dates(), min_train=60, test_size=12, embargo=1)
    seen = set()
    for fold in folds:
        for d in fold.test_dates:
            assert d not in seen  # each test date predicted exactly once
            seen.add(d)


def test_rolling_caps_train_window():
    folds = walk_forward_folds(
        _dates(240), min_train=60, test_size=12, embargo=1, rolling=True, max_train=60
    )
    assert all(len(f.train_dates) <= 60 for f in folds)
