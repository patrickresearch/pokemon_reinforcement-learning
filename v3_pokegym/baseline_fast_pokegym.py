import multiprocessing

# pokegym.environment creates a multiprocessing.Manager at import time, which
# only works with fork; macOS defaults to spawn.
multiprocessing.set_start_method("fork", force=True)

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from pokegym.config import FIXES, EnvConfig
from pokegym.environment import Environment
from pokegym.rnd_wrapper import RNDRewardWrapper
from pokegym_callback import RewardComponentCallback, RunLogCallback


# SB3's SubprocVecEnv worker requires a gymnasium.Env subclass.
class GymnasiumEnvAdapter(gym.Env):
    def __init__(self, inner):
        self.inner = inner
        self.action_space = inner.action_space
        self.observation_space = inner.observation_space

    def reset(self, **kwargs):
        return self.inner.reset(**kwargs)

    def step(self, action):
        return self.inner.step(action)

    def render(self):
        return None

    def close(self):
        self.inner.close()


def make_env(rank, cfg):
    def _init():
        return GymnasiumEnvAdapter(Environment(headless=True, config=replace(cfg, rank=rank)))
    return _init


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain", "--", "."))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--rnd-coef", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--go-explore", choices=("frontier", "latest", "legacy", "off"))
    parser.add_argument("--loop-penalty", choices=("no_battle", "moves_only", "legacy", "off"))
    parser.add_argument("--level-reward", choices=("capped", "trained", "legacy"))
    parser.add_argument("--battle-reward", choices=("ko", "legacy"))
    parser.add_argument("--team-reward", choices=("off", "legacy"))
    parser.add_argument("--battle-obs", choices=("on", "off"))
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--baseline", action="store_true")
    preset.add_argument("--r1", action="store_true")
    parser.add_argument("--ablate", action="append", choices=FIXES, default=[])
    args = parser.parse_args()

    run_dir = Path("runs_pokegym") / args.name
    run_dir.mkdir(parents=True)

    cfg = EnvConfig.baseline() if args.baseline else EnvConfig.r1() if args.r1 else EnvConfig()
    cfg = replace(cfg, run_dir=str(run_dir), **{name: False for name in args.ablate})
    for name in ("go_explore", "loop_penalty", "level_reward", "battle_reward", "team_reward"):
        if getattr(args, name):
            cfg = replace(cfg, **{name: getattr(args, name)})
    if args.battle_obs:
        cfg = replace(cfg, battle_obs=args.battle_obs == "on")

    ppo_kwargs = dict(n_steps=2048, batch_size=512, n_epochs=3, gamma=0.998, ent_coef=0.01, device=args.device)
    (run_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "env": asdict(cfg), "ppo": ppo_kwargs, "git": git_state()}, indent=2
    ))

    set_random_seed(args.seed)
    env = SubprocVecEnv([make_env(i, cfg) for i in range(args.num_envs)])
    if args.rnd_coef > 0:
        env = RNDRewardWrapper(env, intrinsic_coef=args.rnd_coef, device="cpu")

    model = PPO("MultiInputPolicy", env, verbose=1, seed=args.seed, **ppo_kwargs)
    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))

    callbacks = CallbackList([
        CheckpointCallback(save_freq=args.save_freq, save_path=run_dir / "checkpoints", name_prefix="poke"),
        RewardComponentCallback(log_freq=1000),
        RunLogCallback(run_dir),
    ])
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks)
    finally:
        model.save(run_dir / "checkpoints" / "poke_final")
        env.close()
