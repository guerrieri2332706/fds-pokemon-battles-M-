"""Features of the random forest notebook (versions 1 and 2).

Version 1 (47 features)
    * state of the battle at turn 30 (7): team HP, losses, HP of the active Pokemon;
    * momentum and damage: HP and loss differences at turns 10 and 20, damage dealt and recovered;
    * play style: switches, status and priority moves, stat boosts gained, share of STAB attacks,
      attacks without visible damage, turns spent asleep, paralysed, behind a substitute,
      seeded or confused; team information (Pokemon used and revealed, Leftovers, mean base stats).

Version 2 (296 features) adds three groups:
    A. condition of each team at turn 30: sorted HP of the six Pokemon, HP bands, "usable" HP
       (alive and not asleep or frozen), HP weighted by base stats, status counts, active status and boosts;
    B. type effectiveness of the attacks: super effective, resisted, no effect, mean base power;
    C. sparse columns: uses of the 58 frequent moves and HP of the 41 frequent species (198 columns).

Every value depends on a single battle plus the lookup tables built on the training set.
"""

import collections

import numpy as np
import pandas as pd

from .lookups import STATS, STATUSES

CONDITIONS = {"turns_asleep": "slp", "turns_paralyzed": "par"}
EFFECTS = {"turns_substitute": "substitute", "turns_seeded": "leechseed", "turns_confused": "confusion"}
NON_FEATURES = ["battle_id", "player_won"]


# ---------------------------------------------------------------- helpers
def last_seen_hp(timeline, side):
    """Last recorded HP of every Pokemon of one side that appeared in the given turns."""
    hp = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        hp[state["name"]] = state["hp_pct"]
    return hp


def team_hp(timeline, side, p1_team):
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
            "p1_team_hp": team_hp(timeline, "p1", p1_team),
            "p2_team_hp": team_hp(timeline, "p2", p1_team),
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
            features[f"team_hp_diff_t{t}"] = team_hp(window, "p1", p1_team) - team_hp(window, "p2", p1_team)
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
