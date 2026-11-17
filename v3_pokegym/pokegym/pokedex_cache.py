"""Loads the offline PokeAPI cache built by scripts/build_pokedex_cache.py.

Read once per process, kept in memory - no network I/O at runtime. Used by
reward calculation (environment.py) and the team optimizer
(team_optimizer.py). Keyed by the *internal Gen 1 index* (the byte pokegym
reads straight out of party RAM), not the National Pokedex number, so
callers can go straight from `ram_map.party_stats()`'s `species` field to a
cache entry without an extra lookup.
"""
import json
from pathlib import Path

_CACHE_PATH = Path(__file__).parent / "data" / "pokedex_cache.json"

_raw = None


def _load():
    global _raw
    if _raw is None:
        if not _CACHE_PATH.exists():
            raise FileNotFoundError(
                f"{_CACHE_PATH} not found - run `python3 scripts/build_pokedex_cache.py` "
                "once before training (see v3_pokegym/scripts/build_pokedex_cache.py)."
            )
        _raw = json.loads(_CACHE_PATH.read_text())
    return _raw


def species(internal_index: int) -> dict | None:
    """Cached PokeAPI data for one species, keyed by internal Gen 1 index.
    Returns None for indices with no real species (e.g. MissingNo slots)."""
    return _load()["species"].get(str(internal_index))


def move(move_name: str) -> dict | None:
    return _load()["moves"].get(move_name)


def move_by_id(move_id: int) -> dict | None:
    if "move_ids" not in _load():
        raise KeyError(f"{_CACHE_PATH} has no move_ids - run `python scripts/build_pokedex_cache.py --move-ids`.")
    return _raw["move_ids"].get(str(move_id))


def type_chart() -> dict:
    """Gen 1 type-effectiveness relations, keyed by attacking type name."""
    return _load()["type_chart"]


def all_species() -> dict:
    return _load()["species"]
