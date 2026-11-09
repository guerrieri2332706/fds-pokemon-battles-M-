"""Team M4 - FDS Kaggle challenge "Pokemon Battles Prediction 2026".

The package contains all the code behind our three final submissions:

* ``data``               loading the competition files, sanity checks, writing submissions;
* ``lookups``            tables built from the training set only (species types, base stats,
                         frequent moves and species) and the Gen 2 type effectiveness;
* ``features_logistic``  feature versions 1-7 used by the logistic regression and gradient boosting;
* ``features_forest``    feature versions 1-2 used by the random forest;
* ``models``             the estimators and the small grid searches we ran;
* ``validation``         the shared 5-fold cross-validation and fold-by-fold comparison tables.

Every random component uses ``SEED = 42``.
"""

SEED = 42

__all__ = ["SEED"]
