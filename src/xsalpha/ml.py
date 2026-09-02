"""Combining signals with a gradient-boosted model.

The label is the cross-sectional RANK of next-period return, not the raw
return. Ranks are stationary across regimes and stop the model from
chasing outliers; we only care about ordering the cross-section anyway.

Trained under purged walk-forward CV (see cv.py). The output score is
treated exactly like any other signal: z-scored per date, quintile
portfolio, same cost model. That keeps the ML vs. linear-baseline
comparison apples-to-apples.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .cv import walk_forward_folds

# LightGBM is imported inside ml_scores, not here. Its wheel links against OpenMP,
# which macOS does not ship: on a clean machine `import lightgbm` raises
# "Library not loaded: @rpath/libomp.dylib" and took the whole module with it,
# including equal_weight_combo, which is plain pandas and needs none of it.
# Fix the environment with `brew install libomp`.

# deliberately shallow trees + strong subsampling: monthly cross-sections
# are small and noisy, anything deeper memorizes the panel
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_lambda=1.0,
    random_state=7,
    verbose=-1,
    # LightGBM's sklearn wrapper defaults to "split" - a count of how often a
    # feature was used, which says nothing about how much it helped. The plot and
    # the write-up both claim gain, so ask for gain.
    importance_type="gain",
)


def _stack(signals: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long panel: index (date, ticker), one column per signal."""
    parts = {name: df.loc[df.index.isin(dates)].stack() for name, df in signals.items()}
    return pd.DataFrame(parts)


def ml_scores(
    signals: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    min_train: int = 60,
    test_size: int = 12,
    embargo: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purged walk-forward LightGBM. Returns (scores, feature_importance).

    scores: date x ticker DataFrame of out-of-sample model scores
    feature_importance: one row per fold, gain importance per feature
    """
    from lightgbm import LGBMRegressor

    dates = fwd_ret.index
    X = _stack(signals, dates).dropna()
    y = fwd_ret.stack().reindex(X.index)
    ok = y.notna()
    X, y = X[ok], y[ok]
    # label: cross-sectional percentile rank of forward return
    y_rank = y.groupby(level=0).rank(pct=True)

    folds = walk_forward_folds(dates, min_train=min_train, test_size=test_size, embargo=embargo)
    preds = []
    importances = []
    for fold in folds:
        tr = X.index.get_level_values(0).isin(fold.train_dates)
        te = X.index.get_level_values(0).isin(fold.test_dates)
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X[tr], y_rank[tr])
        preds.append(pd.Series(model.predict(X[te]), index=X.index[te]))
        importances.append(
            pd.Series(model.feature_importances_, index=X.columns, name=fold.test_dates[0])
        )

    if not preds:
        raise ValueError("no folds produced predictions - not enough history?")
    score_long = pd.concat(preds)
    scores = score_long.unstack()
    imp = pd.DataFrame(importances)
    imp = imp.div(imp.sum(axis=1), axis=0)  # normalize gain per fold
    return scores, imp


def equal_weight_combo(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Naive 1/N combo of z-scored signals - the baseline ML has to beat.

    Averaged on the labels. Stacking `.to_numpy()` averaged by position, so any
    signal whose columns came back in a different order - a different dropna, a
    different source frame - was silently blended into the wrong tickers.
    """
    if not signals:
        raise ValueError("no signals to combine")
    panel = pd.concat(signals.values(), keys=list(signals), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows early in sample
        return panel.groupby(level=1).mean()
