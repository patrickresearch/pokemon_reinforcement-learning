"""One-time offline sync: pull everything the reward/team-optimizer code
needs from PokeAPI for the 151 Gen 1 species, and write it to a single local
JSON file. Run this once (or whenever you want to refresh it) - training
itself must never hit the network, so pokegym/pokedex_cache.py only ever
reads the file this script produces.

Usage (from v3_pokegym/, with the venv active):
    python3 scripts/build_pokedex_cache.py
"""
import json
import multiprocessing
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# pokegym/__init__.py -> environment.py creates a multiprocessing.Manager()
# at class-body/import time, which only works with 'fork' (macOS defaults to
# 'spawn'). This script doesn't touch the emulator at all, but importing
# pokegym.gen1_data still runs pokegym/__init__.py first, so this is needed
# just to make the import succeed - see baseline_fast_pokegym.py for the
# same fix applied where it actually matters (env vectorization).
multiprocessing.set_start_method('fork', force=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from pokegym.gen1_data import INTERNAL_INDEX_TO_NAME, to_pokeapi_slug

API_BASE = "https://pokeapi.co/api/v2"
OUT_PATH = Path(__file__).parent.parent / "pokegym" / "data" / "pokedex_cache.json"
REQUEST_DELAY_S = 0.15  # be polite to the free, shared API


def fetch_json(url: str) -> dict:
    # PokeAPI's CDN returns 403 for requests with no/blank User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "pokemon-rl-cache-builder/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def build_type_chart() -> dict:
    """Gen 1 type-effectiveness chart. PokeAPI's `type` endpoint carries
    `past_damage_relations` for generations where a type's matchups differed
    from the current games - Gen 1 lacks Dark/Steel entirely and had several
    now-corrected relations (e.g. Ice was not resisted by Fire in Gen 1;
    Poison still hit Bug super-effectively, etc.). Falls back to the
    current-gen relations when no Gen 1 override is present for a type.
    """
    chart = {}
    types_index = fetch_json(f"{API_BASE}/type?limit=30")
    for entry in types_index["results"]:
        type_data = fetch_json(entry["url"])
        name = type_data["name"]
        if name in ("unknown", "shadow", "stellar"):
            continue  # not part of Gen 1
        relations = type_data["damage_relations"]
        for past in type_data.get("past_damage_relations", []):
            if past["generation"]["name"] == "generation-i":
                relations = past["damage_relations"]
                break
        chart[name] = {
            "double_damage_to": sorted(t["name"] for t in relations["double_damage_to"]),
            "half_damage_to": sorted(t["name"] for t in relations["half_damage_to"]),
            "no_damage_to": sorted(t["name"] for t in relations["no_damage_to"]),
        }
        time.sleep(REQUEST_DELAY_S)
    return chart


def build_species_cache() -> dict:
    species_cache = {}
    move_cache = {}
    for internal_index, name in sorted(INTERNAL_INDEX_TO_NAME.items()):
        slug = to_pokeapi_slug(name)
        data = fetch_json(f"{API_BASE}/pokemon/{slug}")
        time.sleep(REQUEST_DELAY_S)

        stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
        level_up_moves = []
        for move_entry in data["moves"]:
            for version_detail in move_entry["version_group_details"]:
                if (
                    version_detail["move_learn_method"]["name"] == "level-up"
                    and version_detail["version_group"]["name"] == "red-blue"
                ):
                    move_name = move_entry["move"]["name"]
                    level_up_moves.append(
                        {"move": move_name, "level": version_detail["level_learned_at"]}
                    )
                    if move_name not in move_cache:
                        move_cache[move_name] = None  # resolved in a second pass below
        level_up_moves.sort(key=lambda m: m["level"])

        species_cache[internal_index] = {
            "name": name,
            "pokeapi_slug": slug,
            "national_dex_number": data["id"],
            "types": [t["type"]["name"] for t in data["types"]],
            # hp/attack/defense/speed are unchanged since Gen 1; "special" is
            # only an approximation (see gen1_data.py docstring) until the
            # true original single-stat value is sourced.
            "hp": stats["hp"],
            "attack": stats["attack"],
            "defense": stats["defense"],
            "speed": stats["speed"],
            "special_approx": round((stats["special-attack"] + stats["special-defense"]) / 2),
            "level_up_moves": level_up_moves,
        }
        print(f"  [{internal_index:3d}/151] {name} ({slug})")

    for move_name in move_cache:
        move_data = fetch_json(f"{API_BASE}/move/{move_name}")
        time.sleep(REQUEST_DELAY_S)
        move_cache[move_name] = {
            "type": move_data["type"]["name"],
            "power": move_data["power"],
            "damage_class": move_data["damage_class"]["name"] if move_data["damage_class"] else None,
        }

    return species_cache, move_cache


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)
    print("Fetching Gen 1 type effectiveness chart...")
    type_chart = build_type_chart()
    print(f"  {len(type_chart)} types cached")

    print("Fetching species + level-up movesets for 151 species...")
    species_cache, move_cache = build_species_cache()

    payload = {
        "type_chart": type_chart,
        "species": species_cache,
        "moves": move_cache,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH} ({len(species_cache)} species, {len(move_cache)} moves, {len(type_chart)} types)")


if __name__ == "__main__":
    main()
