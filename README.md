# FDS · Pokémon Battles Prediction 2026 · Team M⁴

Code of team **M⁴** (Matteo Schirru, Alessia Manfredi, Daniela Madonna, Margherita Guerrieri, Sapienza University of Rome)
for the Kaggle competition [FDS: Pokémon Battles Prediction 2026](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2026):
predict whether Player 1 wins a Gen 2 OU battle (`player_won`) from Player 1's team, Player 2's lead and the first 30 turns.

Our public Kaggle notebook (`kaggle_notebook.ipynb`) clones this repository and calls the package `pokemon_battles`
to rebuild our three final submissions.

## Final submissions

| File | Model | Features | CV accuracy | Public | Private |
|---|---|---|---|---|---|
| `submission_lr.csv` | Standardised L2 logistic regression | 154 (version 6) | 67.86% | 68.67% | 66.97% |
| `submission_gb.csv` | Histogram gradient boosting (lr 0.03, 15 leaves, 300 iterations) | 159 (version 7) | 67.62% | 68.00% | 67.26% |
| `submission_rf.csv` | Random forest (300 trees) stacked on an out-of-fold logistic score | 296 (forest version 2) | 67.78% | 68.67% | 66.66% |

CV accuracy is the mean over the same 5 stratified folds (seed 42) used for every model.

## Repository structure

```
kaggle_notebook.ipynb        public Kaggle notebook: clones this repo and writes the three submissions
make_submissions.py          the same three submissions from the command line
requirements.txt
pokemon_battles/
    data.py                  loading the competition files, sanity checks, writing a submission
    lookups.py               tables from the training set only: species types and base stats,
                             frequent moves and species, Gen 2 type effectiveness
    features_logistic.py     feature versions 1-7 (logistic regression and gradient boosting)
    features_forest.py       feature versions 1-2 of the random forest
    models.py                the estimators and the grid searches we ran
    validation.py            the shared 5-fold cross-validation and fold-by-fold tables
```

The competition data are not included: on Kaggle they are attached to the notebook as input.

## How to run

**On Kaggle.** Create a notebook from the competition page (*Code → New Notebook*, so the data are attached),
import `kaggle_notebook.ipynb` (*File → Import Notebook*), turn **Internet** on in the notebook settings and run all
cells. The three csv files appear in `/kaggle/working`. The whole notebook takes a few minutes on CPU.

**Locally.**

```bash
pip install -r requirements.txt
python make_submissions.py --data-dir path/to/competition/files --out-dir .
```

`--data-dir` is the folder with `train.jsonl`, `test.jsonl`, `sample_submission.csv` and `type_chart_gen2.csv`.

## Method in brief

* **Validation.** One 5-fold `StratifiedKFold` (shuffled, seed 42) for every experiment, compared fold by fold;
  a change was kept only if it raised the mean accuracy and won in most folds. The leaderboard never guided our choices.
* **No leakage.** Lookup tables are built from the training set without the target; every feature depends on a single
  battle; standardisation and stacking are inside scikit-learn estimators, so they are refitted in each training fold.
* **Features.** Logistic regression: seven incremental versions, from pre-battle stats (51.78%) to whole-team HP,
  momentum, type effectiveness and frequent moves (version 6, 67.86%); immunity features (version 7) are used only
  by gradient boosting. Random forest: battle state and dynamics (version 1, 47 features), then team condition,
  type effectiveness and 198 sparse move/species columns (version 2, 296 features).
* **Models.** Logistic regression, histogram gradient boosting tuned by an 8-setting grid, and a random forest that
  receives the out-of-fold logistic probability through a `StackingClassifier` (`passthrough=True`).
* **Rules.** Classical machine learning only; the only data used are the competition files and the Gen 2 type chart
  provided with them.

## Reproducibility

Every random component uses `SEED = 42`. The scores above were obtained in Kaggle's Python 3.13 environment.
In a different environment (another scikit-learn version, another machine) the logistic regression gives the same
results, while gradient boosting and the random forest can move by a few tenths of a point.
