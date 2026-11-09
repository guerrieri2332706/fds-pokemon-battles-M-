"""Loading the competition files, sanity checks and writing submission files."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COMPETITION = "fds-pokemon-battles-prediction-2026"


@dataclass
class CompetitionData:
    """Everything provided by the competition."""

    train: list                       # 10,000 labelled battles (one dict per battle)
    test: list                        # 5,000 battles to predict
    sample_submission: pd.DataFrame   # template with the expected battle_id order
    type_chart: pd.DataFrame          # Gen 2 type chart (attacking_type, defending_type, multiplier)


def find_data_dir(root="/kaggle/input"):
    """Return the folder that contains ``train.jsonl``.

    On Kaggle the competition data are mounted under ``/kaggle/input`` (the exact sub-folder
    has changed over time), so we search for the file instead of hard-coding the path.
    """
    root = Path(root)
    if (root / "train.jsonl").exists():
        return root
    matches = sorted(root.rglob("train.jsonl"))
    if not matches:
        raise FileNotFoundError(f"train.jsonl not found under {root}: attach the competition data.")
    return matches[0].parent


def load_battles(file_path):
    """Parse a JSONL file (one battle per line) into a list of dictionaries."""
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        battles = [json.loads(line) for line in f]
    print(f"{file_path.name}: {len(battles)} battles loaded.")
    return battles


def load_competition_data(data_dir):
    """Load the four competition files from ``data_dir``."""
    data_dir = Path(data_dir)
    return CompetitionData(
        train=load_battles(data_dir / "train.jsonl"),
        test=load_battles(data_dir / "test.jsonl"),
        sample_submission=pd.read_csv(data_dir / "sample_submission.csv"),
        type_chart=pd.read_csv(data_dir / "type_chart_gen2.csv"),
    )


def check_data(data):
    """Sanity checks: unique ids, target only in train, template matching the test ids."""
    train_ids = [b["battle_id"] for b in data.train]
    test_ids = [b["battle_id"] for b in data.test]

    assert len(train_ids) == len(set(train_ids)), "Duplicate IDs in train."
    assert len(test_ids) == len(set(test_ids)), "Duplicate IDs in test."
    assert all("player_won" in b for b in data.train), "Missing target in train."
    assert all("player_won" not in b for b in data.test), "Target found in test."

    target = pd.Series([b["player_won"] for b in data.train], name="player_won")
    assert target.isin([0, 1]).all(), "Target is not binary."

    assert list(data.sample_submission.columns) == ["battle_id", "player_won"]
    assert set(data.sample_submission["battle_id"]) == set(test_ids), "Template does not match test IDs."

    turns = pd.Series([len(b["battle_timeline"]) for b in data.train + data.test])
    print("All checks passed.")
    print(f"Train battles: {len(train_ids)}   Test battles: {len(test_ids)}   "
          f"Player 1 win rate in train: {target.mean():.1%}")
    print("Turns per battle (train + test):", turns.value_counts().to_dict())


def write_submission(predictions, battle_ids, sample_submission, filename):
    """Write ``predictions`` in the order of the template, matching them by ``battle_id``."""
    predictions_by_id = pd.Series(predictions, index=list(battle_ids))
    submission = sample_submission.copy()
    submission["player_won"] = submission["battle_id"].map(predictions_by_id)

    if submission["player_won"].isna().any():
        raise ValueError("Some battle_ids have no prediction.")

    submission["player_won"] = submission["player_won"].astype(int)
    submission.to_csv(filename, index=False)
    print(f"{filename} created: {submission.shape[0]} rows, "
          f"predicted Player 1 win rate {submission['player_won'].mean():.1%}")
    return submission
