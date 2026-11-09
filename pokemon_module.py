"""Team M4 - FDS Kaggle challenge "Pokemon Battles Prediction 2026".

All the code behind our three final submissions, in a single module that our Kaggle notebook
clones from GitHub and imports (``from pokemon_module import ...``). Sections:

    1. Data               loading the competition files, sanity checks, writing a submission
    2. Lookup tables      built from the training set only: species types and base stats,
                          frequent moves and species, Gen 2 type effectiveness
    3. Logistic features  feature versions 1-7 (logistic regression and gradient boosting)
    4. Forest features    feature versions 1-2 of the random forest
    5. Models             the estimators and the grid searches we ran
    6. Validation         the shared 5-fold cross-validation and fold-by-fold tables

Every random component uses SEED = 42.
"""

import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
NON_FEATURES = ["battle_id", "player_won"]   # columns of a feature table that are not features


# ==================================================================================================
# 1. DATA
# ==================================================================================================
# Loading the competition files, sanity checks and writing submission files.

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


# ==================================================================================================
# 2. LOOKUP TABLES (training set only)
# ==================================================================================================
# Lookup tables, built from the training set only.
#
# None of these tables uses the target ``player_won``: they only describe which species, moves
# and types appear in the training battles. Test battles never contribute to them, so applying
# the feature functions to the test set cannot leak information.

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
STATUSES = ["slp", "par", "tox", "psn", "brn", "frz"]

MIN_MOVE_USES = 1000      # a move is "frequent" if used at least 1,000 times in the training timelines
MIN_SPECIES_TEAMS = 150   # a species is "frequent" if it appears in at least 150 Player 1 teams


@dataclass
class Lookups:
    type_multiplier: dict      # (attacking type, defending type) -> damage multiplier
    species_types: dict        # species -> list of its types
    base_stat_total: dict      # species -> sum of its six base stats
    mean_bst: float            # mean base stat total over the known species
    frequent_moves: list       # moves used at least MIN_MOVE_USES times (58 moves)
    frequent_species: list     # species in at least MIN_SPECIES_TEAMS Player 1 teams (41 species)

    def effectiveness(self, move_type, defender_types):
        """Damage multiplier of a move type against a Pokemon with the given types."""
        multiplier = 1.0
        for defender_type in defender_types:
            if defender_type != "notype":
                multiplier *= self.type_multiplier.get((move_type, defender_type), 1.0)
        return multiplier


def build_lookups(train_data, type_chart):
    """Build every lookup table from the training battles and the provided type chart."""
    type_multiplier = {
        (row.attacking_type, row.defending_type): row.multiplier
        for row in type_chart.itertuples(index=False)
    }

    # Species seen in the training set: Player 1's full teams and Player 2's leads
    species_types = {}
    base_stat_total = {}
    for battle in train_data:
        for pokemon in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            species_types[pokemon["name"]] = pokemon["types"]
            base_stat_total[pokemon["name"]] = sum(pokemon[s] for s in STATS)
    mean_bst = float(np.mean(list(base_stat_total.values())))

    move_usage = collections.Counter(
        turn[f"{side}_move_details"]["name"]
        for battle in train_data
        for turn in battle["battle_timeline"]
        for side in ("p1", "p2")
        if turn[f"{side}_move_details"]
    )
    frequent_moves = sorted(name for name, count in move_usage.items() if count >= MIN_MOVE_USES)

    species_count = collections.Counter(p["name"] for battle in train_data for p in battle["p1_team_details"])
    frequent_species = sorted(name for name, count in species_count.items() if count >= MIN_SPECIES_TEAMS)

    return Lookups(type_multiplier, species_types, base_stat_total, mean_bst, frequent_moves, frequent_species)


# ==================================================================================================
# 3. FEATURES OF THE LOGISTIC REGRESSION AND GRADIENT BOOSTING (versions 1-7)
# ==================================================================================================
# Feature versions 1-7 of the logistic regression notebook (also used by gradient boosting).
#
# Each version adds one group of features to the previous one, and each group tests one idea
# about what decides a battle. Every function works on one battle at a time (plus the lookup
# tables built on the training set), so it can be applied to the test set without leakage.
#
#     v1  pre-battle stats: mean base stats of Player 1's team, base stats of Player 2's lead
#     v2  + HP of the two active Pokemon at turn 30
#     v3  + number of Pokemon lost by Player 1
#     v4  + Player 2's losses, status conditions of each team, stat boosts of the active Pokemon
#     v5  + Pokemon "in good shape" (at least 50% HP and no status) for each player
#     v6  + whole-team HP and its trend from turn 20 to 30, type effectiveness, 58 frequent moves
#     v7  + immunities (attacks wasted on immune Pokemon, immune matchups at turn 30)

def feature_columns(df):
    """All the columns of a feature table except the id and the target."""
    return [col for col in df.columns if col not in NON_FEATURES]


# ---------------------------------------------------------------- version 1
def create_simple_features(data):
    """Version 1: base stats known before the battle starts."""
    feature_list = []

    for battle in data:
        features = {}

        p1_team = battle["p1_team_details"]
        features["p1_mean_hp"] = np.mean([p["base_hp"] for p in p1_team])
        features["p1_mean_spe"] = np.mean([p["base_spe"] for p in p1_team])
        features["p1_mean_atk"] = np.mean([p["base_atk"] for p in p1_team])
        features["p1_mean_def"] = np.mean([p["base_def"] for p in p1_team])

        p2_lead = battle["p2_lead_details"]
        features["p2_lead_hp"] = p2_lead["base_hp"]
        features["p2_lead_spe"] = p2_lead["base_spe"]

        features["battle_id"] = battle["battle_id"]
        if "player_won" in battle:
            features["player_won"] = int(battle["player_won"])

        feature_list.append(features)

    return pd.DataFrame(feature_list)


# ---------------------------------------------------------------- version 2
def add_last_hp_features(data, base_df):
    """Version 2: HP of the two active Pokemon at the last observed turn (turn 30)."""
    result = base_df.copy()

    last_turns = [battle["battle_timeline"][-1] for battle in data]
    result["p1_active_hp_last"] = [turn["p1_pokemon_state"]["hp_pct"] for turn in last_turns]
    result["p2_active_hp_last"] = [turn["p2_pokemon_state"]["hp_pct"] for turn in last_turns]
    result["active_hp_last_diff"] = result["p1_active_hp_last"] - result["p2_active_hp_last"]

    return result


# ---------------------------------------------------------------- version 3
def add_p1_fainted_feature(data, base_df):
    """Version 3: number of distinct Player 1 Pokemon seen at 0 HP within the 30 turns."""
    result = base_df.copy()

    p1_fainted_counts = []
    for battle in data:
        fainted = {
            turn["p1_pokemon_state"]["name"]
            for turn in battle["battle_timeline"]
            if turn["p1_pokemon_state"]["hp_pct"] == 0
        }
        p1_fainted_counts.append(len(fainted))

    result["p1_fainted_count"] = p1_fainted_counts
    return result


# ---------------------------------------------------------------- version 4
def add_condition_features(data, base_df):
    """Version 4: Player 2's losses, last known status of every Pokemon seen, active stat boosts."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        features = {}

        features["p2_fainted_count"] = len({
            turn["p2_pokemon_state"]["name"]
            for turn in timeline
            if turn["p2_pokemon_state"]["hp_pct"] == 0
        })

        for side in ("p1", "p2"):
            last_status = {}
            for turn in timeline:
                state = turn[f"{side}_pokemon_state"]
                last_status[state["name"]] = state["status"]

            for status in STATUSES:
                features[f"{side}_n_{status}"] = sum(s == status for s in last_status.values())

            features[f"{side}_active_boosts"] = sum(timeline[-1][f"{side}_pokemon_state"]["boosts"].values())

        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- version 5
def add_healthy_features(data, base_df):
    """Version 5: Pokemon in good shape (>= 50% HP and no status); unseen Pokemon count as healthy."""
    rows = []
    for battle in data:
        features = {}
        for side in ("p1", "p2"):
            last_state = {}
            for turn in battle["battle_timeline"]:
                state = turn[f"{side}_pokemon_state"]
                last_state[state["name"]] = state

            never_seen = 6 - len(last_state)
            features[f"{side}_healthy_count"] = never_seen + sum(
                state["status"] == "nostatus" and state["hp_pct"] >= 0.5
                for state in last_state.values()
            )
        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- version 6
def team_hp(timeline, side):
    """Remaining HP of a whole team, in full Pokemon (6 = all six at full HP).

    Pokemon not yet seen in the given turns count as full HP.
    """
    last_hp = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        last_hp[state["name"]] = state["hp_pct"]
    never_seen = 6 - len(last_hp)
    return sum(last_hp.values()) + never_seen


def add_team_hp_features(data, base_df):
    """Version 6 (part 1): remaining HP of both teams at turn 30 and change of the HP difference since turn 20."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        p1_hp_30, p2_hp_30 = team_hp(timeline, "p1"), team_hp(timeline, "p2")
        p1_hp_20, p2_hp_20 = team_hp(timeline[:20], "p1"), team_hp(timeline[:20], "p2")
        rows.append({
            "p1_team_hp": p1_hp_30,
            "p2_team_hp": p2_hp_30,
            "team_hp_diff_change_last10": (p1_hp_30 - p2_hp_30) - (p1_hp_20 - p2_hp_20),
        })

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


def add_move_features(data, base_df, lookups):
    """Version 6 (part 2): type effectiveness, status moves, mean base power and usage of the frequent moves.

    The defender's types come from the species table built on the training set;
    unknown species count as neutral.
    """
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        features = {}

        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            super_effective = resisted = status_moves = 0
            powers = []
            used = collections.Counter()

            for turn in timeline:
                move = turn[f"{side}_move_details"]
                if move is None:          # no move recorded: the player switched
                    continue
                used[move["name"]] += 1
                if move["category"] == "status":
                    status_moves += 1
                elif move["base_power"] > 0:
                    powers.append(move["base_power"])
                    defender = turn[f"{opponent}_pokemon_state"]["name"]
                    multiplier = lookups.effectiveness(move["type"], lookups.species_types.get(defender, ["notype"]))
                    super_effective += multiplier > 1
                    resisted += multiplier < 1

            features[f"{side}_super_effective"] = super_effective
            features[f"{side}_resisted"] = resisted
            features[f"{side}_status_moves"] = status_moves
            features[f"{side}_mean_power"] = np.mean(powers) if powers else 0.0
            for move_name in lookups.frequent_moves:
                features[f"{side}_move_{move_name}"] = used[move_name]

        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- version 7
def add_immunity_features(data, base_df, lookups):
    """Version 7: attacks wasted on immune Pokemon and immunities between the Pokemon at turn 30."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        team_types = {p["name"]: [t for t in p["types"] if t != "notype"] for p in battle["p1_team_details"]}
        features = {}

        # Damaging moves that hit an immune Pokemon
        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            wasted = 0
            for turn in timeline:
                move = turn[f"{side}_move_details"]
                if move and move["base_power"] > 0:
                    defender = turn[f"{opponent}_pokemon_state"]["name"]
                    if lookups.effectiveness(move["type"], lookups.species_types.get(defender, ["notype"])) == 0:
                        wasted += 1
            features[f"{side}_immune_hits"] = wasted

        # Immunity between the two active Pokemon at turn 30 (own types as a proxy for their attacks)
        name1 = timeline[-1]["p1_pokemon_state"]["name"]
        name2 = timeline[-1]["p2_pokemon_state"]["name"]
        types1 = team_types.get(name1, [])
        types2 = [t for t in lookups.species_types.get(name2, ["notype"]) if t != "notype"]
        features["p2_active_immune_to_p1"] = int(any(lookups.effectiveness(t, types2) == 0 for t in types1))
        features["p1_active_immune_to_p2"] = int(any(lookups.effectiveness(t, types1) == 0 for t in types2))

        # Player 1's Pokemon that could switch in safely against Player 2's active Pokemon
        features["p1_team_immune_to_p2_active"] = sum(
            any(lookups.effectiveness(t, own_types) == 0 for t in types2) for own_types in team_types.values()
        )
        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- all versions
def build_logistic_features(data, lookups):
    """Feature tables of versions 1-7 for one set of battles (each version extends the previous one).

    Returns a dict ``{'v1': DataFrame, ..., 'v7': DataFrame}``; every table keeps ``battle_id``
    (and ``player_won`` for the training set), see :func:`feature_columns`.
    """
    versions = {"v1": create_simple_features(data)}
    versions["v2"] = add_last_hp_features(data, versions["v1"])
    versions["v3"] = add_p1_fainted_feature(data, versions["v2"])
    versions["v4"] = add_condition_features(data, versions["v3"])
    versions["v5"] = add_healthy_features(data, versions["v4"])
    versions["v6"] = add_move_features(data, add_team_hp_features(data, versions["v5"]), lookups)
    versions["v7"] = add_immunity_features(data, versions["v6"], lookups)
    return versions


# ==================================================================================================
# 4. FEATURES OF THE RANDOM FOREST (versions 1-2)
# ==================================================================================================
# Features of the random forest notebook (versions 1 and 2).
#
# Version 1 (47 features)
#     * state of the battle at turn 30 (7): team HP, losses, HP of the active Pokemon;
#     * momentum and damage: HP and loss differences at turns 10 and 20, damage dealt and recovered;
#     * play style: switches, status and priority moves, stat boosts gained, share of STAB attacks,
#       attacks without visible damage, turns spent asleep, paralysed, behind a substitute,
#       seeded or confused; team information (Pokemon used and revealed, Leftovers, mean base stats).
#
# Version 2 (296 features) adds three groups:
#     A. condition of each team at turn 30: sorted HP of the six Pokemon, HP bands, "usable" HP
#        (alive and not asleep or frozen), HP weighted by base stats, status counts, active status and boosts;
#     B. type effectiveness of the attacks: super effective, resisted, no effect, mean base power;
#     C. sparse columns: uses of the 58 frequent moves and HP of the 41 frequent species (198 columns).
#
# Every value depends on a single battle plus the lookup tables built on the training set.

CONDITIONS = {"turns_asleep": "slp", "turns_paralyzed": "par"}
EFFECTS = {"turns_substitute": "substitute", "turns_seeded": "leechseed", "turns_confused": "confusion"}


# ---------------------------------------------------------------- helpers
def last_seen_hp(timeline, side):
    """Last recorded HP of every Pokemon of one side that appeared in the given turns."""
    hp = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        hp[state["name"]] = state["hp_pct"]
    return hp


def forest_team_hp(timeline, side, p1_team):
    """Remaining HP of a whole team in "full Pokemon"; unseen Pokemon count as full HP."""
    hp = last_seen_hp(timeline, side)
    if side == "p1":
        return sum(hp.get(name, 1.0) for name in p1_team)
    return sum(hp.values()) + (6 - len(hp))


def fainted(timeline, side):
    """Number of Pokemon of one side whose last recorded HP is 0."""
    return sum(value == 0 for value in last_seen_hp(timeline, side).values())


# ---------------------------------------------------------------- version 1
def create_state_features(data):
    """State of the battle at turn 30 (7 features)."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        p1_team = [p["name"] for p in battle["p1_team_details"]]
        features = {
            "p1_team_hp": forest_team_hp(timeline, "p1", p1_team),
            "p2_team_hp": forest_team_hp(timeline, "p2", p1_team),
        }
        features["team_hp_diff"] = features["p1_team_hp"] - features["p2_team_hp"]
        features["p1_fainted"] = fainted(timeline, "p1")
        features["p2_fainted"] = fainted(timeline, "p2")
        features["p1_active_hp"] = timeline[-1]["p1_pokemon_state"]["hp_pct"]
        features["p2_active_hp"] = timeline[-1]["p2_pokemon_state"]["hp_pct"]

        features["battle_id"] = battle["battle_id"]
        if "player_won" in battle:
            features["player_won"] = int(battle["player_won"])
        rows.append(features)
    return pd.DataFrame(rows)


def add_damage_features(data, base_df):
    """Momentum at turns 10 and 20, damage dealt, largest hit, damage in the last 5 turns, HP recovered."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        p1_team = [p["name"] for p in battle["p1_team_details"]]
        features = {}

        # Momentum: how the advantage looked earlier in the battle
        for t in (10, 20):
            window = timeline[:t]
            features[f"team_hp_diff_t{t}"] = forest_team_hp(window, "p1", p1_team) - forest_team_hp(window, "p2", p1_team)
            features[f"fainted_diff_t{t}"] = fainted(window, "p2") - fainted(window, "p1")

        # Damage dealt and HP recovered, turn by turn, only while the same Pokemon stays on the field
        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            damage, largest_hit, damage_last5, healing = 0.0, 0.0, 0.0, 0.0
            for k in range(1, len(timeline)):
                before, after = timeline[k - 1][f"{opponent}_pokemon_state"], timeline[k][f"{opponent}_pokemon_state"]
                if before["name"] == after["name"] and after["hp_pct"] < before["hp_pct"]:
                    hit = before["hp_pct"] - after["hp_pct"]
                    damage += hit
                    largest_hit = max(largest_hit, hit)
                    if k >= len(timeline) - 5:
                        damage_last5 += hit
                own_before, own_after = timeline[k - 1][f"{side}_pokemon_state"], timeline[k][f"{side}_pokemon_state"]
                if own_before["name"] == own_after["name"] and own_after["hp_pct"] > own_before["hp_pct"]:
                    healing += own_after["hp_pct"] - own_before["hp_pct"]
            features[f"{side}_damage_dealt"] = damage
            features[f"{side}_healing"] = healing
            features[f"{side}_max_hit"] = largest_hit
            features[f"{side}_damage_last5"] = damage_last5

        features["damage_diff"] = features["p1_damage_dealt"] - features["p2_damage_dealt"]
        features["damage_last5_diff"] = features["p1_damage_last5"] - features["p2_damage_last5"]
        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


def add_style_features(data, base_df, lookups):
    """Play style, turns spent under each condition and team information."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        p1_types = {p["name"]: p["types"] for p in battle["p1_team_details"]}
        features = {}

        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            counts = collections.Counter()
            for k, turn in enumerate(timeline):
                move = turn[f"{side}_move_details"]
                state = turn[f"{side}_pokemon_state"]

                if move is None:                       # no move recorded: the player switched
                    counts["switches"] += 1
                else:
                    counts["priority_moves"] += move["priority"] > 0
                    if move["category"] == "status":
                        counts["status_moves"] += 1
                    else:
                        counts["attacks"] += 1
                        own_types = (p1_types.get(state["name"]) if side == "p1"
                                     else lookups.species_types.get(state["name"], []))
                        counts["stab_attacks"] += bool(own_types) and move["type"] in own_types
                        # Attack with no visible damage on the same opposing Pokemon
                        if k > 0:
                            before, after = timeline[k - 1][f"{opponent}_pokemon_state"], turn[f"{opponent}_pokemon_state"]
                            counts["attacks_no_damage"] += (before["name"] == after["name"]
                                                            and after["hp_pct"] >= before["hp_pct"])

                # Stat boosts gained while the same Pokemon stays on the field
                previous = timeline[k - 1][f"{side}_pokemon_state"] if k > 0 else None
                if previous is not None and previous["name"] == state["name"]:
                    counts["boost_gain"] += sum(max(0, state["boosts"][s] - previous["boosts"][s]) for s in state["boosts"])

                for name, status in CONDITIONS.items():
                    counts[name] += state["status"] == status
                for name, effect in EFFECTS.items():
                    counts[name] += effect in state["effects"]

            features[f"{side}_switches"] = counts["switches"]
            features[f"{side}_status_moves"] = counts["status_moves"]
            features[f"{side}_stab_share"] = counts["stab_attacks"] / counts["attacks"] if counts["attacks"] else 0.0
            features[f"{side}_priority_moves"] = counts["priority_moves"]
            features[f"{side}_boost_gain"] = counts["boost_gain"]
            for name in list(CONDITIONS) + list(EFFECTS):
                features[f"{side}_{name}"] = counts[name]
            features[f"{side}_attacks_no_damage"] = counts["attacks_no_damage"]

        # Team information
        features["p1_used"] = len(last_seen_hp(timeline, "p1"))
        features["p2_revealed"] = len(last_seen_hp(timeline, "p2"))
        features["p1_leftovers"] = sum(p.get("item") == "leftovers" for p in battle["p1_team_details"])
        features["p1_mean_bst"] = np.mean([sum(p[s] for s in STATS) for p in battle["p1_team_details"]])
        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- version 2
def team_at_turn30(battle, side):
    """(name, HP, status) of the six Pokemon of one side at turn 30.

    Player 2's Pokemon not yet revealed are unknown (name None), at full HP and with no status.
    """
    hp, status = {}, {}
    for turn in battle["battle_timeline"]:
        state = turn[f"{side}_pokemon_state"]
        hp[state["name"]] = state["hp_pct"]
        status[state["name"]] = state["status"]
    if side == "p1":
        names = [p["name"] for p in battle["p1_team_details"]]
    else:
        names = list(hp) + [None] * (6 - len(hp))
    return [(name, hp.get(name, 1.0), status.get(name, "nostatus")) for name in names]


def add_version2_features(data, base_df, lookups):
    """Groups A (team condition), B (type effectiveness) and C (moves and species)."""
    rows = []
    for battle in data:
        timeline = battle["battle_timeline"]
        p1_types = {p["name"]: p["types"] for p in battle["p1_team_details"]}
        features = {}

        # A. Condition of each team at turn 30
        for side in ("p1", "p2"):
            team = team_at_turn30(battle, side)
            hp_values = [hp for _, hp, _ in team]
            for rank, value in enumerate(sorted(hp_values, reverse=True), start=1):
                features[f"{side}_hp_rank{rank}"] = value
            features[f"{side}_n_low"] = sum(0 < v <= 0.25 for v in hp_values)
            features[f"{side}_n_mid"] = sum(0.25 < v <= 0.75 for v in hp_values)
            features[f"{side}_n_high"] = sum(v > 0.75 for v in hp_values)
            features[f"{side}_usable_hp"] = sum(hp for _, hp, s in team if hp > 0 and s not in ("slp", "frz"))
            features[f"{side}_weighted_hp"] = sum(hp * lookups.base_stat_total.get(name, lookups.mean_bst) / lookups.mean_bst
                                                  for name, hp, _ in team)
            for status in STATUSES:
                features[f"{side}_n_{status}"] = sum(s == status for _, _, s in team)
            last = timeline[-1][f"{side}_pokemon_state"]
            features[f"{side}_active_status"] = int(last["status"] not in ("nostatus", "fnt"))
            features[f"{side}_active_boosts"] = sum(last["boosts"].values())
        for name in ("n_low", "n_mid", "n_high", "usable_hp", "weighted_hp"):
            features[f"{name}_diff"] = features[f"p1_{name}"] - features[f"p2_{name}"]

        # B. Type effectiveness of the attacks against the Pokemon hit
        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            super_effective = resisted = no_effect = 0
            powers = []
            for turn in timeline:
                move = turn[f"{side}_move_details"]
                if move and move["base_power"] > 0:
                    defender = turn[f"{opponent}_pokemon_state"]["name"]
                    defender_types = (p1_types.get(defender) if opponent == "p1"
                                      else lookups.species_types.get(defender, ["notype"]))
                    multiplier = lookups.effectiveness(move["type"], defender_types or ["notype"])
                    super_effective += multiplier > 1
                    resisted += 0 < multiplier < 1
                    no_effect += multiplier == 0
                    powers.append(move["base_power"])
            features[f"{side}_super_effective"] = super_effective
            features[f"{side}_resisted"] = resisted
            features[f"{side}_no_effect"] = no_effect
            features[f"{side}_mean_power"] = np.mean(powers) if powers else 0.0

        # C. Moves used and remaining HP of the most frequent species
        for side in ("p1", "p2"):
            used = collections.Counter(turn[f"{side}_move_details"]["name"] for turn in timeline
                                       if turn[f"{side}_move_details"])
            for name in lookups.frequent_moves:
                features[f"{side}_move_{name}"] = used[name]
            last_hp = last_seen_hp(timeline, side)
            team = set(p1_types) if side == "p1" else set(last_hp)   # for Player 2, only the Pokemon seen so far
            for name in lookups.frequent_species:
                features[f"{side}_species_{name}"] = last_hp.get(name, 1.0) if name in team else 0.0

        rows.append(features)

    return pd.concat([base_df, pd.DataFrame(rows, index=base_df.index)], axis=1)


# ---------------------------------------------------------------- both versions
def _feature_columns(df):
    return [col for col in df.columns if col not in NON_FEATURES]


def build_forest_features(data, lookups):
    """Feature table with all the columns of versions 1 and 2 for one set of battles.

    Returns ``(table, groups)``. ``table`` keeps ``battle_id`` (and ``player_won`` for the
    training set); ``groups`` holds the column lists, in the order used by our models:
        state   the 7 state features at turn 30;
        v1      the 47 features of version 1;
        dense   version 1 plus groups A and B (98 columns);
        sparse  the 198 columns of group C (moves and species);
        v2      dense columns followed by the sparse ones (296 columns).
    """
    state_df = create_state_features(data)
    v1_df = add_style_features(data, add_damage_features(data, state_df), lookups)
    v2_df = add_version2_features(data, v1_df, lookups)

    v1 = _feature_columns(v1_df)
    v2_new = [col for col in v2_df.columns if col not in v1_df.columns]
    sparse = [col for col in v2_new if "_move_" in col or "_species_" in col]
    dense = v1 + [col for col in v2_new if col not in sparse]
    groups = {"state": _feature_columns(state_df), "v1": v1, "dense": dense,
              "sparse": sparse, "v2": dense + sparse}
    return v2_df, groups


# ==================================================================================================
# 5. MODELS
# ==================================================================================================
# Estimators of our three model families, with the settings we validated.
#
# All of them are scikit-learn estimators, so preprocessing (standardisation) and stacking are
# refitted inside every training fold of the cross-validation: no information from a validation
# fold reaches the model trained on the other folds.

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


# ==================================================================================================
# 6. VALIDATION
# ==================================================================================================
# The cross-validation shared by all our experiments.
#
# Every model in every notebook was compared on the same 5 stratified folds (seed 42), fold by
# fold: a change was kept only if it raised the mean accuracy and won in most folds.

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
