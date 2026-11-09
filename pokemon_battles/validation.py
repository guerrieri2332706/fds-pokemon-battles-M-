"""The cross-validation shared by all our experiments.

Every model in every notebook was compared on the same 5 stratified folds (seed 42), fold by
fold: a change was kept only if it raised the mean accuracy and won in most folds.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import SEED


def make_cv():
    """The 5 stratified folds used everywhere."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_accuracy(model, X, y, cv):
    """Accuracy on each of the 5 validation folds."""
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")


def grid_fold_scores(search):
    """Fold-by-fold scores of the best setting of a fitted ``GridSearchCV``."""
    best = search.best_index_
    n_splits = search.n_splits_
    return np.array([search.cv_results_[f"split{k}_test_score"][best] for k in range(n_splits)])


def fold_table(scores, reference=None):
    """Table with one column per model, one row per fold, plus mean and standard deviation.

    If ``reference`` names one of the columns, a last row counts in how many folds each
    model beats it.
    """
    table = pd.DataFrame(scores, index=[f"fold {k}" for k in range(1, len(next(iter(scores.values()))) + 1)])
    summary = pd.DataFrame([table.mean(), table.std(ddof=0)], index=["mean", "std"])
    table = pd.concat([table, summary])
    table = table.map(lambda v: f"{v:.2%}")
    if reference is not None:
        wins = {name: int((np.asarray(s) > np.asarray(scores[reference])).sum()) for name, s in scores.items()}
        table.loc[f"folds better than {reference}"] = [str(wins[name]) if name != reference else "-"
                                                       for name in table.columns]
    return table
