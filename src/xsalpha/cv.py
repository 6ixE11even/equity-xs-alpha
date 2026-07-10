"""Purged walk-forward cross-validation.

Standard K-fold leaks badly in this setting: the label at date t is the
return over (t, t+1], so a training sample at t and a test sample at t-1
share information. The fix (Lopez de Prado, "Advances in Financial ML",
ch. 7) is to purge training samples whose label window overlaps the test
window and embargo a buffer after the test set.

Walk-forward variant here: expanding or rolling train window, always
predicting strictly forward, with an embargo gap between train end and
test start. Simpler than combinatorial purged CV and closer to how the
model would actually be deployed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    min_train: int = 60,
    test_size: int = 12,
    embargo: int = 1,
    rolling: bool = False,
    max_train: int = 120,
) -> list[Fold]:
    """Build purged walk-forward folds over an ordered date index.

    min_train  first fold trains on at least this many periods
    test_size  periods per test block
    embargo    gap (periods) between train end and test start; >=1 removes
               the label-overlap leak for one-period-ahead labels
    rolling    if True, cap the train window at max_train (recent regime
               weighting); if False, expanding window
    """
    dates = pd.DatetimeIndex(dates)
    folds = []
    start = min_train
    while start + embargo < len(dates):
        test = dates[start + embargo : start + embargo + test_size]
        if len(test) == 0:
            break
        lo = max(0, start - max_train) if rolling else 0
        train = dates[lo:start]
        folds.append(Fold(train_dates=train, test_dates=test))
        start += test_size
    return folds
