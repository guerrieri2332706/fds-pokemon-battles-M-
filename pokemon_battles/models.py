"""Estimators of our three model families, with the settings we validated.

All of them are scikit-learn estimators, so preprocessing (standardisation) and stacking are
refitted inside every training fold of the cross-validation: no information from a validation
fold reaches the model trained on the other folds.
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import SEED


# ---------------------------------------------------------------- logistic regression
def logistic_plain():
    """Logistic regression on raw features (versions 1-3)."""
    return LogisticRegression(random_state=SEED, max_iter=1000)


def logistic_scaled():
    """Standardisation + L2 logistic regression (from version 4; final model of submission 1).

    Without standardisation the solver did not converge once counts and HP were mixed.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, max_iter=1000))


# ---------------------------------------------------------------- gradient boosting
GB_GRID = {"learning_rate": [0.03, 0.06], "max_leaf_nodes": [7, 15], "max_iter": [300, 600]}


def gradient_boosting_search(cv):
    """Grid search (8 settings) for the histogram gradient boosting of submission 2.

    The best setting on our folds is learning_rate=0.03, max_leaf_nodes=15, max_iter=300;
    ``refit=True`` (default) refits it on all the training battles.
    """
    return GridSearchCV(
        HistGradientBoostingClassifier(l2_regularization=1.0, random_state=SEED),
        param_grid=GB_GRID, cv=cv, scoring="accuracy", n_jobs=-1,
    )


# ---------------------------------------------------------------- random forest
FOREST_GRID = {"max_features": ["sqrt", "log2"], "min_samples_leaf": [1, 5, 20]}


def random_forest():
    """Random forest with scikit-learn's default settings (100 trees), version 1."""
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)


def random_forest_search(cv):
    """Control grid around the default forest (6 settings); all within about 0.6 points."""
    return GridSearchCV(random_forest(), param_grid=FOREST_GRID, cv=cv, scoring="accuracy",
                        return_train_score=True)


def logistic_sparse():
    """Stronger regularisation (C=0.1) for the 296 features of version 2, many of them sparse."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def stacked_forest(max_features=0.2, min_samples_leaf=5):
    """Random forest (final model) that also receives the out-of-fold logistic score (submission 3).

    The logistic score of each training battle comes from an inner 5-fold split, so the forest
    never sees a score computed on the battle itself; with ``passthrough=True`` the forest also
    receives all the original features.
    """
    return StackingClassifier(
        estimators=[("logistic_score", logistic_sparse())],
        final_estimator=RandomForestClassifier(n_estimators=300, max_features=max_features,
                                               min_samples_leaf=min_samples_leaf, random_state=SEED, n_jobs=-1),
        passthrough=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method="predict_proba",
    )
