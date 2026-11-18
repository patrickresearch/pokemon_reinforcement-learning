import multiprocessing

multiprocessing.set_start_method("fork", force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from pokegym.config import EnvConfig
from pokegym.environment import Environment

B_BUTTON = 5
EPISODE_STEPS = 8


class FakeGame:
    def __init__(self, mem):
        self.mem = mem

    def get_memory_value(self, addr):
        return self.mem.get(addr, 0)


def test_box_struct_stride_and_level_offset():
    env = Environment.__new__(Environment)
    env.box_stride, env.box_level_offset = 33, 0x03
    env.game = FakeGame({0xDA80: 2, 0xDA96: 0x99, 0xDA99: 7, 0xDA96 + 33: 0xB0, 0xDA99 + 33: 12})
    assert env.get_box_pokemon() == [{"species": 0x99, "level": 7}, {"species": 0xB0, "level": 12}]


def run_over_time_limit(cfg):
    if not Path("pokemon_red.gb").exists():
        pytest.skip("pokemon_red.gb not in CWD")
    env = Environment(headless=True, config=cfg)
    env.reset()
    for _ in range(EPISODE_STEPS):
        _, _, terminated, truncated, _ = env.step(B_BUTTON)
    env.reset()
    _, reward, _, _, info = env.step(B_BUTTON)
    env.close()
    return terminated, truncated, reward, info["reward_components"]["catch_bonus"]


def stationary_loop_penalty(cfg, steps=40):
    if not Path("pokemon_red.gb").exists():
        pytest.skip("pokemon_red.gb not in CWD")
    env = Environment(headless=True, config=cfg)
    env.reset()
    for _ in range(steps):
        _, _, _, _, info = env.step(B_BUTTON)
    env.close()
    return info["reward_components"]["loop_penalty"]


def test_moves_only_ignores_button_presses():
    assert stationary_loop_penalty(EnvConfig(go_explore="off", traj_stride=0, loop_penalty="moves_only")) == 0


@pytest.mark.parametrize("cfg", [EnvConfig(go_explore="off", traj_stride=0), EnvConfig.baseline(traj_stride=0)])
def test_standing_still_outside_battle_is_penalised(cfg):
    assert stationary_loop_penalty(cfg) < 0


def test_trajectory_is_flushed_on_close(tmp_path):
    if not Path("pokemon_red.gb").exists():
        pytest.skip("pokemon_red.gb not in CWD")
    env = Environment(headless=True, config=EnvConfig(go_explore="off", traj_stride=2, run_dir=str(tmp_path), rank=3))
    env.reset()
    for _ in range(10):
        env.step(B_BUTTON)
    env.close()
    data = (tmp_path / "traj" / "env3.u8").read_bytes()
    assert len(data) == 20
    assert all(b >> 2 == B_BUTTON for b in data[3::4])


def test_fixed_time_limit_truncates_without_refarm():
    cfg = EnvConfig(max_episode_steps=EPISODE_STEPS, go_explore="off", traj_stride=0)
    terminated, truncated, reward, catch_bonus = run_over_time_limit(cfg)
    assert truncated and not terminated
    assert catch_bonus == 0
    assert reward == 0


def test_baseline_reproduces_catch_refarm():
    cfg = EnvConfig.baseline(max_episode_steps=EPISODE_STEPS, traj_stride=0)
    terminated, truncated, _, catch_bonus = run_over_time_limit(cfg)
    assert terminated and truncated
    assert catch_bonus > 0


def test_frontier_prefers_unused_then_newest_cells_on_any_map():
    from pokegym.environment import frontier_target
    archive = {
        (12, 0, 0): (None, 0, 0, 0, 100, 0),
        (51, 2, 3): (None, 0, 0, 0, 900, 0),
        (51, 2, 4): (None, 0, 0, 0, 950, 1),
    }
    assert frontier_target(archive, (12, 0, 0)) == (51, 2, 3)
    assert frontier_target(archive, (51, 2, 3)) == (12, 0, 0)
    assert frontier_target({(12, 0, 0): archive[(12, 0, 0)]}, (12, 0, 0)) is None
