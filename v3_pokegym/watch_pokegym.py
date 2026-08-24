"""Run a single windowed (non-headless) env so you can watch it play live.

Usage (from v3_pokegym/, with the venv active):
    python3 watch_pokegym.py                          # random actions
    python3 watch_pokegym.py --model runs_pokegym/poke_pokegym_500000_steps.zip
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
from pokegym.environment import Environment


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to an SB3 PPO .zip checkpoint")
    args = parser.parse_args()

    # headless=False opens a real SDL2 window (pokegym/pyboy_binding.py's
    # make_env() also bumps emulation speed to 6x automatically in this case).
    env = GymnasiumEnvAdapter(Environment(headless=False))
    obs, info = env.reset()

    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded checkpoint: {args.model}")
    else:
        model = None
        print("No --model given, taking random actions. Ctrl+C to stop.")

    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
