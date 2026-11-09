"""Feature versions 1-7 of the logistic regression notebook (also used by gradient boosting).

Each version adds one group of features to the previous one, and each group tests one idea
about what decides a battle. Every function works on one battle at a time (plus the lookup
tables built on the training set), so it can be applied to the test set without leakage.

    v1  pre-battle stats: mean base stats of Player 1's team, base stats of Player 2's lead
    v2  + HP of the two active Pokemon at turn 30
    v3  + number of Pokemon lost by Player 1
    v4  + Player 2's losses, status conditions of each team, stat boosts of the active Pokemon
    v5  + Pokemon "in good shape" (at least 50% HP and no status) for each player
    v6  + whole-team HP and its trend from turn 20 to 30, type effectiveness, 58 frequent moves
    v7  + immunities (attacks wasted on immune Pokemon, immune matchups at turn 30)
"""

import collections

import numpy as np
import pandas as pd

from .lookups import STATUSES

NON_FEATURES = ["battle_id", "player_won"]


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
