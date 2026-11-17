import numpy as np

from . import pokedex_cache
from .gen1_data import GEN1_TYPE_ID_TO_NAME

PARTY_COUNT = 0xD163
PARTY_BASE = 0xD16B
PARTY_STRIDE = 44
BATTLE_FLAG = 0xD057
MENU_ITEM = 0xCC26
ENEMY_SPECIES = 0xCFE5
ENEMY_HP = 0xCFE6
ENEMY_TYPES = (0xCFEA, 0xCFEB)
ENEMY_LEVEL = 0xCFF3
ENEMY_MAX_HP = 0xCFF4
PLAYER_HP = 0xD015
PLAYER_TYPES = (0xD019, 0xD01A)
PLAYER_MOVES = 0xD01C
PLAYER_MAX_HP = 0xD023
PLAYER_PP = 0xD02D
MAX_MOVE_POWER = 200
BATTLE_FEATURES = 1 + 4 * 4 + 2 + 4


def u16(game, addr):
    return 256 * game.get_memory_value(addr) + game.get_memory_value(addr + 1)


def party_identities(game):
    # OT ID + DVs survive evolution and party reordering, species and slot do not.
    mons = []
    for i in range(min(game.get_memory_value(PARTY_COUNT), 6)):
        base = PARTY_BASE + i * PARTY_STRIDE
        mons.append(((u16(game, base + 0x0C), u16(game, base + 0x1B)), game.get_memory_value(base + 0x21)))
    return mons


def trained_levels(game, catch_levels):
    return sum(level - catch_levels.setdefault(key, level) for key, level in party_identities(game))


def type_names(game, addrs):
    ids = {game.get_memory_value(a) for a in addrs}
    return [GEN1_TYPE_ID_TO_NAME[t] for t in ids if t in GEN1_TYPE_ID_TO_NAME]


def type_multiplier(attack_type, defender_types):
    relations = pokedex_cache.type_chart()[attack_type]
    multiplier = 1.0
    for t in defender_types:
        if t in relations["no_damage_to"]:
            multiplier *= 0.0
        elif t in relations["double_damage_to"]:
            multiplier *= 2.0
        elif t in relations["half_damage_to"]:
            multiplier *= 0.5
    return multiplier


def battle_features(game):
    feats = np.zeros(BATTLE_FEATURES, dtype=np.float32)
    if game.get_memory_value(BATTLE_FLAG) == 0:
        return feats
    own, enemy = type_names(game, PLAYER_TYPES), type_names(game, ENEMY_TYPES)
    feats[0] = 1.0
    for i in range(4):
        move = pokedex_cache.move_by_id(game.get_memory_value(PLAYER_MOVES + i))
        if move is None:
            continue
        power = move["power"] or 0
        slot = feats[1 + 4 * i:5 + 4 * i]
        slot[0] = type_multiplier(move["type"], enemy) / 4 if power else 0.0
        slot[1] = min(power / MAX_MOVE_POWER, 1.0)
        slot[2] = move["type"] in own
        slot[3] = (game.get_memory_value(PLAYER_PP + i) & 0x3F) > 0
    feats[17] = min(u16(game, PLAYER_HP) / max(u16(game, PLAYER_MAX_HP), 1), 1.0)
    feats[18] = min(u16(game, ENEMY_HP) / max(u16(game, ENEMY_MAX_HP), 1), 1.0)
    feats[19 + min(game.get_memory_value(MENU_ITEM), 3)] = 1.0
    return feats


def enemy_snapshot(game):
    return (
        game.get_memory_value(BATTLE_FLAG),
        game.get_memory_value(ENEMY_SPECIES),
        u16(game, ENEMY_HP),
        game.get_memory_value(ENEMY_LEVEL),
    )


def enemy_fainted(prev, cur):
    # Fleeing or catching ends the battle with the enemy's HP above zero.
    return prev[0] > 0 and cur[0] > 0 and prev[1] == cur[1] != 0 and prev[2] > 0 and cur[2] == 0
