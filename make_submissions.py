"""Write our three final submissions without opening the notebook.

    python make_submissions.py --data-dir path/to/competition/files --out-dir .

The script runs the same steps as ``kaggle_notebook.ipynb`` (without the extra comparisons)
and writes ``submission_lr.csv``, ``submission_gb.csv`` and ``submission_rf.csv``.
"""

import argparse
from pathlib import Path

import pandas as pd

from pokemon_battles import models
from pokemon_battles.data import check_data, find_data_dir, load_competition_data, write_submission
from pokemon_battles.features_forest import build_forest_features
from pokemon_battles.features_logistic import build_logistic_features, feature_columns
from pokemon_battles.lookups import build_lookups
from pokemon_battles.validation import make_cv


def main(data_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_competition_data(data_dir)
    check_data(data)
    lookups = build_lookups(data.train, data.type_chart)
    y = pd.Series([b["player_won"] for b in data.train], name="player_won")
    test_ids = [b["battle_id"] for b in data.test]
    cv = make_cv()

    # Submission 1: logistic regression, version 6 features (154)
    lr_train = build_logistic_features(data.train, lookups)
    lr_test = build_logistic_features(data.test, lookups)
    cols_v6 = feature_columns(lr_train["v6"])
    logistic = models.logistic_scaled().fit(lr_train["v6"][cols_v6], y)
    write_submission(logistic.predict(lr_test["v6"][cols_v6]), test_ids, data.sample_submission,
                     out_dir / "submission_lr.csv")

    # Submission 2: gradient boosting, version 7 features (159), best setting of the grid
    cols_v7 = feature_columns(lr_train["v7"])
    gb_search = models.gradient_boosting_search(cv).fit(lr_train["v7"][cols_v7], y)
    print("Gradient boosting, best setting:", gb_search.best_params_)
    write_submission(gb_search.best_estimator_.predict(lr_test["v7"][cols_v7]), test_ids,
                     data.sample_submission, out_dir / "submission_gb.csv")

    # Submission 3: stacked random forest, forest version 2 features (296)
    rf_train, groups = build_forest_features(data.train, lookups)
    rf_test, _ = build_forest_features(data.test, lookups)
    forest = models.stacked_forest().fit(rf_train[groups["v2"]], y)
    write_submission(forest.predict(rf_test[groups["v2"]]), test_ids, data.sample_submission,
                     out_dir / "submission_rf.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=None,
                        help="folder with train.jsonl, test.jsonl, sample_submission.csv, type_chart_gen2.csv "
                             "(default: searched under /kaggle/input)")
    parser.add_argument("--out-dir", default=".", help="where to write the three csv files")
    args = parser.parse_args()
    main(args.data_dir or find_data_dir(), args.out_dir)
