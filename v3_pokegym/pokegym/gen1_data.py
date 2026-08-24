"""Hand-curated Gen 1 (Red/Blue) facts that PokeAPI does not model directly.

PokeAPI is species/move/type data for the modern games; it has no concept of
a specific game's fixed trainer rosters, and its per-species "stats" reflect
the current games (hp/attack/defense/speed are unchanged since Gen 1, but the
single Gen 1 "Special" stat was later split into Special Attack/Special
Defense - not always as a simple average). Both gaps are filled here instead
of guessing them into the PokeAPI cache.
"""

from .pokemon_data import pokemon_data

# pokegym reads Pokemon by their internal Gen 1 index byte (POKE[i] in
# ram_map.py), which is NOT the National Pokedex number - pokemon_data.py
# already carries that exact internal-index -> English-name mapping (used
# by ram_map.pokemon()/pokemon_l()), so it's reused here as the bridge to
# PokeAPI species names rather than re-deriving/duplicating it.
INTERNAL_INDEX_TO_NAME = {
    int(entry["decimal"]): entry["name"]
    for entry in pokemon_data
    if "MissingNo" not in entry["name"]
}
assert len(INTERNAL_INDEX_TO_NAME) == 151, (
    f"expected 151 real Gen 1 species in pokemon_data.py, got {len(INTERNAL_INDEX_TO_NAME)}"
)

_SLUG_OVERRIDES = {
    "Nidoran♂": "nidoran-m",
    "Nidoran♀": "nidoran-f",
    "Mr. Mime": "mr-mime",
    "Farfetch'd": "farfetchd",
}


def to_pokeapi_slug(name: str) -> str:
    """English display name (as used in pokemon_data.py) -> PokeAPI slug."""
    if name in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[name]
    return name.lower().replace(" ", "-").replace(".", "").replace("'", "")


# Gen 1's internal type-ID byte (as stored in TYPE1/TYPE2 party RAM) is a
# separate, game-internal encoding that has nothing to do with PokeAPI - it
# has to be mapped to PokeAPI's lowercase type names by hand. IDs 0x09-0x13
# and 0x1B+ are unused in a real, non-glitched Gen 1 save.
GEN1_TYPE_ID_TO_NAME = {
    0x00: "normal", 0x01: "fighting", 0x02: "flying", 0x03: "poison",
    0x04: "ground", 0x05: "rock", 0x07: "bug", 0x08: "ghost",
    0x14: "fire", 0x15: "water", 0x16: "grass", 0x17: "electric",
    0x18: "psychic", 0x19: "ice", 0x1A: "dragon",
}


# Deterministic mandatory trainer battles, keyed by badge count at the time
# they become the *next* fight (0 badges = about to fight Brock, etc.).
# Species+level pulled from the well-documented, version-shared core Gen 1
# gym leader / rival teams (Red version; Blue swaps a handful of species but
# levels match). This is intentionally a coarse "what's coming up" signal
# for the team optimizer's opponent-aware scoring, not a battle simulator -
# minor species/level drift against the exact ROM is fine, refine here if it
# turns out to matter empirically.
NEXT_TRAINER_BY_BADGE_COUNT = {
    0: {"name": "Brock", "team": [("Geodude", 12), ("Onix", 14)]},
    1: {"name": "Misty", "team": [("Staryu", 18), ("Starmie", 21)]},
    2: {"name": "Lt. Surge", "team": [("Voltorb", 21), ("Pikachu", 18), ("Raichu", 24)]},
    3: {"name": "Erika", "team": [("Victreebel", 29), ("Tangela", 24), ("Vileplume", 29)]},
    4: {"name": "Koga", "team": [("Koffing", 37), ("Muk", 39), ("Koffing", 37), ("Weezing", 43)]},
    5: {"name": "Sabrina", "team": [("Kadabra", 38), ("Mr. Mime", 37), ("Venomoth", 38), ("Alakazam", 43)]},
    6: {"name": "Blaine", "team": [("Growlithe", 42), ("Ponyta", 40), ("Rapidash", 42), ("Arcanine", 47)]},
    7: {"name": "Giovanni (Viridian)", "team": [
        ("Rhyhorn", 45), ("Dugtrio", 42), ("Nidoqueen", 44), ("Nidoking", 45), ("Rhydon", 50),
    ]},
}
