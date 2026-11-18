import multiprocessing

multiprocessing.set_start_method("fork", force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from pokegym import game_state as gs
from pokegym.config import EnvConfig

FIRE, WATER, NORMAL = 0x14, 0x15, 0x00
MOVES = {33: {"type": "normal", "power": 35}, 55: {"type": "water", "power": 40}, 45: {"type": "normal", "power": None}}


class FakeGame:
    def __init__(self, mem=None):
        self.mem = dict(mem or {})

    def get_memory_value(self, addr):
        return self.mem.get(addr, 0)


def party(*mons):
    mem = {gs.PARTY_COUNT: len(mons)}
    for i, (ot, dvs, level) in enumerate(mons):
        base = gs.PARTY_BASE + i * gs.PARTY_STRIDE
        mem.update({base + 0x0C: ot >> 8, base + 0x0D: ot & 255, base + 0x1B: dvs >> 8, base + 0x1C: dvs & 255, base + 0x21: level})
    return FakeGame(mem)


def test_trained_levels_ignore_catches_and_survive_reordering():
    catch_levels = {}
    assert gs.trained_levels(party((1, 0xABCD, 5)), catch_levels) == 0
    assert gs.trained_levels(party((1, 0xABCD, 8)), catch_levels) == 3
    assert gs.trained_levels(party((1, 0x1111, 3), (1, 0xABCD, 8)), catch_levels) == 3
    assert gs.trained_levels(party((1, 0x2222, 4), (1, 0xABCD, 8), (1, 0x1111, 5)), catch_levels) == 5


def test_capped_levels_stop_at_cap_and_unlock_with_badges():
    assert gs.level_cap(0) == 19 and gs.level_cap(1) == 26 and gs.level_cap(8) is None
    catch_levels = {}
    gs.trained_levels(party((1, 0xABCD, 6), (1, 0x1111, 20)), catch_levels)
    game = party((1, 0xABCD, 30), (1, 0x1111, 25))
    assert gs.trained_levels(game, catch_levels, cap=16) == 10
    assert gs.trained_levels(game, catch_levels, cap=23) == 17 + 3
    assert gs.trained_levels(game, catch_levels) == 24 + 5


def battle(moves, pp=(10, 10, 10, 10), menu=0):
    mem = {gs.BATTLE_FLAG: 1, gs.PLAYER_TYPES[0]: WATER, gs.PLAYER_TYPES[1]: WATER,
           gs.ENEMY_TYPES[0]: FIRE, gs.ENEMY_TYPES[1]: FIRE, gs.MENU_ITEM: menu,
           gs.PLAYER_HP + 1: 10, gs.PLAYER_MAX_HP + 1: 20, gs.ENEMY_HP + 1: 5, gs.ENEMY_MAX_HP + 1: 20}
    for i, (move, p) in enumerate(zip(moves, pp)):
        mem[gs.PLAYER_MOVES + i] = move
        mem[gs.PLAYER_PP + i] = p
    return FakeGame(mem)


@pytest.fixture
def moves(monkeypatch):
    monkeypatch.setattr(gs.pokedex_cache, "move_by_id", lambda i: MOVES.get(i))


def test_battle_features_encode_effectiveness_stab_and_pp(moves):
    feats = gs.battle_features(battle((33, 55, 45, 0), pp=(10, 0xC0, 10, 0), menu=1))
    tackle, water_gun, growl, empty = feats[1:5], feats[5:9], feats[9:13], feats[13:17]
    assert feats[0] == 1
    assert list(tackle) == pytest.approx([0.25, 35 / 200, 0, 1])
    assert list(water_gun) == pytest.approx([0.5, 40 / 200, 1, 0])
    assert list(growl) == [0, 0, 0, 1]
    assert list(empty) == [0, 0, 0, 0]
    assert feats[17:19] == pytest.approx([0.5, 0.25])
    assert list(feats[19:23]) == [0, 1, 0, 0]


def test_battle_features_are_zero_outside_battle():
    assert not gs.battle_features(FakeGame()).any()


@pytest.mark.parametrize("prev, cur, fainted", [
    ((1, 36, 12, 5), (1, 36, 0, 5), True),
    ((1, 36, 12, 5), (0, 36, 12, 5), False),
    ((1, 36, 12, 5), (1, 36, 9, 5), False),
    ((1, 36, 12, 5), (1, 112, 0, 5), False),
    ((0, 36, 12, 5), (1, 36, 0, 5), False),
])
def test_enemy_fainted_only_on_ko_within_battle(prev, cur, fainted):
    assert gs.enemy_fainted(prev, cur) is fainted


def test_observation_matches_space():
    if not Path("pokemon_red.gb").exists():
        pytest.skip("pokemon_red.gb not in CWD")
    from pokegym.environment import Environment
    for cfg in (EnvConfig(go_explore="off", traj_stride=0), EnvConfig.r1(go_explore="off", traj_stride=0)):
        env = Environment(headless=True, config=cfg)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        assert ("battle" in obs) == cfg.battle_obs
        env.close()
