"""Team-quality scoring: given every Pokémon the player currently has access
to (active party + PC box), score how good the *current* 6-slot party is
versus the best possible 6 out of that pool. Pure functions, no RAM/PyBoy
access - environment.py is responsible for gathering the inputs (via
ram_map.party_stats() / RedGymEnv.get_box_pokemon()) and feeding them in.

Deliberately NOT wired to perform any swap itself - see the Phase 4 plan
note: the agent has to learn to act on this signal through its own action
space, this module only tells it how far from optimal it currently is.
"""
from itertools import combinations

from . import gen1_data, pokedex_cache

WEIGHT_STATS = 1.0
WEIGHT_OFFENSE_COVERAGE = 8.0
WEIGHT_DEFENSE_COVERAGE = 6.0
WEIGHT_NEXT_OPPONENT = 10.0

ALL_GEN1_TYPES = [
    "normal", "fighting", "flying", "poison", "ground", "rock", "bug",
    "ghost", "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
]


def _species_types(internal_index: int) -> list[str]:
    entry = pokedex_cache.species(internal_index)
    return entry["types"] if entry else []


def _estimate_box_mon_stats(internal_index: int, level: int) -> dict:
    """Box-mon RAM doesn't carry pre-computed stats (see the note in
    environment.get_box_pokemon()); estimate them from the PokeAPI base
    stats using the standard Gen 1 stat formula with a flat assumed DV of 8
    (roughly the population average of 0-15) and zero stat experience, since
    neither is readable for a boxed mon without withdrawing it."""
    entry = pokedex_cache.species(internal_index)
    if entry is None:
        return {"attack": 0, "defense": 0, "speed": 0, "special": 0, "hp": 1, "max_hp": 1}
    dv = 8
    def stat(base):
        return (base + dv) * 2 * level // 100 + 5
    def hp_stat(base):
        return (base + dv) * 2 * level // 100 + level + 10
    return {
        "attack": stat(entry["attack"]),
        "defense": stat(entry["defense"]),
        "speed": stat(entry["speed"]),
        "special": stat(entry["special_approx"]),
        "hp": hp_stat(entry["hp"]),
        "max_hp": hp_stat(entry["hp"]),
    }


def _candidate(internal_index: int, level: int, stats: dict | None = None) -> dict:
    """Normalizes a party-slot dict or a (species, level) box entry into one
    shape the scorer works with."""
    if stats is None:
        stats = _estimate_box_mon_stats(internal_index, level)
    return {
        "species": internal_index,
        "level": level,
        "stats": stats,
        "types": _species_types(internal_index),
    }


def party_stats_to_candidates(party_stats: list[dict]) -> list[dict]:
    return [
        _candidate(
            s["species"], s["level"],
            stats={"attack": s["attack"], "defense": s["defense"], "speed": s["speed"],
                   "special": s["special"], "hp": s["hp"], "max_hp": s["max_hp"]},
        )
        for s in party_stats
    ]


def box_pokemon_to_candidates(box_pokemon: list[dict]) -> list[dict]:
    return [_candidate(b["species"], b["level"]) for b in box_pokemon]


def _stat_total(candidate: dict) -> float:
    s = candidate["stats"]
    return s["attack"] + s["defense"] + s["speed"] + s["special"] + s["max_hp"]


def _offensive_coverage(team: list[dict]) -> float:
    """How many of the 15 Gen 1 types the team can hit for at least neutral
    (not resisted, not immune) damage with at least one team member's STAB
    types - counts double-effective hits extra since actually knowing the
    move matters more than just sharing a type, but movesets aren't modeled
    here yet, so type-based STAB is the proxy."""
    chart = pokedex_cache.type_chart()
    covered = set()
    for target_type in ALL_GEN1_TYPES:
        for member in team:
            for atk_type in member["types"]:
                relations = chart.get(atk_type)
                if relations is None:
                    continue
                if target_type in relations["no_damage_to"]:
                    continue
                if target_type in relations["half_damage_to"]:
                    continue
                covered.add(target_type)
                break
            if target_type in covered:
                break
    return len(covered)


def _defensive_coverage(team: list[dict]) -> float:
    """Counts, for each type, how many team members are simultaneously
    weak to it - summed and inverted (fewer shared weaknesses = better),
    normalized so it's on a roughly comparable scale to offensive coverage."""
    chart = pokedex_cache.type_chart()
    total_shared_weakness = 0
    for atk_type in ALL_GEN1_TYPES:
        relations = chart.get(atk_type)
        if relations is None:
            continue
        weak_count = sum(
            1 for member in team
            if any(t in relations["double_damage_to"] for t in member["types"])
        )
        if weak_count >= 3:  # 3+ team members sharing a weakness is the real risk
            total_shared_weakness += weak_count
    max_possible = len(ALL_GEN1_TYPES) * len(team)
    return max_possible - total_shared_weakness


def _next_opponent_coverage(team: list[dict], badge_count: int) -> float:
    upcoming = gen1_data.NEXT_TRAINER_BY_BADGE_COUNT.get(badge_count)
    if upcoming is None:
        return 0.0
    opponent_types = set()
    for species_name, _level in upcoming["team"]:
        matches = [
            idx for idx, name in gen1_data.INTERNAL_INDEX_TO_NAME.items()
            if name == species_name
        ]
        if matches:
            opponent_types.update(_species_types(matches[0]))
    if not opponent_types:
        return 0.0
    hits = 0
    chart = pokedex_cache.type_chart()
    for opp_type in opponent_types:
        for member in team:
            if any(
                opp_type in chart.get(atk_type, {}).get("double_damage_to", [])
                for atk_type in member["types"]
            ):
                hits += 1
                break
    return hits / len(opponent_types)


def score_team(team: list[dict], badge_count: int = 0) -> float:
    if not team:
        return 0.0
    stats_score = sum(_stat_total(m) for m in team) / len(team)
    return (
        WEIGHT_STATS * stats_score
        + WEIGHT_OFFENSE_COVERAGE * _offensive_coverage(team)
        + WEIGHT_DEFENSE_COVERAGE * _defensive_coverage(team)
        + WEIGHT_NEXT_OPPONENT * _next_opponent_coverage(team, badge_count)
    )


def current_team_score(party_stats: list[dict], badge_count: int = 0) -> float:
    return score_team(party_stats_to_candidates(party_stats), badge_count)


def best_team_score(
    party_stats: list[dict], box_pokemon: list[dict], badge_count: int = 0
) -> tuple[float, list[dict]]:
    """Best-scoring 6-Pokemon combination from the full pool (party + box).
    Exhaustive search - fine as long as the pool stays in the tens, which it
    will for a single Gen 1 playthrough (max ~30ish catches before the PC
    box itself fills up)."""
    pool = party_stats_to_candidates(party_stats) + box_pokemon_to_candidates(box_pokemon)
    if len(pool) <= 6:
        return score_team(pool, badge_count), pool
    if len(pool) > 18:
        # C(18,6) is ~18.5k combos, comfortably fast; beyond that it grows
        # too quickly to run inline during training (C(30,6) ~ 593k). Falls
        # back to a greedy per-member add, which is not guaranteed optimal
        # but stays cheap regardless of pool size.
        pool = sorted(pool, key=_stat_total, reverse=True)
        team: list[dict] = []
        remaining = pool[:]
        for _ in range(6):
            remaining.sort(key=lambda c: score_team(team + [c], badge_count), reverse=True)
            team.append(remaining.pop(0))
        return score_team(team, badge_count), team
    best = (-1.0, None)
    for combo in combinations(pool, 6):
        s = score_team(list(combo), badge_count)
        if s > best[0]:
            best = (s, list(combo))
    return best


def team_quality_gap(
    party_stats: list[dict], box_pokemon: list[dict], badge_count: int = 0
) -> float:
    """>= 0. How much score is being left on the table by the current party
    versus the best achievable team from everything caught so far - the
    learnable signal for Phase 4."""
    current = current_team_score(party_stats, badge_count)
    best, _ = best_team_score(party_stats, box_pokemon, badge_count)
    return max(best - current, 0.0)
