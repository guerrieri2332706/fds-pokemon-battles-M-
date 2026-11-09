"""Lookup tables, built from the training set only.

None of these tables uses the target ``player_won``: they only describe which species, moves
and types appear in the training battles. Test battles never contribute to them, so applying
the feature functions to the test set cannot leak information.
"""

import collections
from dataclasses import dataclass

import numpy as np

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
