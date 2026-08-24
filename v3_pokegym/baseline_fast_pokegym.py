import multiprocessing
# pokegym.environment.Base creates a multiprocessing.Manager() at class-body
# (i.e. import) time. That only works with the 'fork' start method. macOS
# defaults to 'spawn', which crashes with an unpicklable-closure error as
# soon as SubprocVecEnv tries to hand env_fn to worker processes. Must be
# set before pokegym (or anything importing it) is imported, and before
# any Process/Pool is created.
multiprocessing.set_start_method('fork', force=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
from pokegym.environment import Environment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from pokegym_callback import RewardComponentCallback
from pokegym.rnd_wrapper import RNDRewardWrapper


# pokegym.environment.Base implements the Gymnasium duck-type (reset/step/
# action_space/observation_space) but does not actually subclass gymnasium.Env.
# SB3 2.3.2's SubprocVecEnv worker does an isinstance(env, gymnasium.Env) check
# and rejects it otherwise, so wrap it.
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


# pokegym/pyboy_binding.py's make_env() currently hardcodes the ROM filename
# to "pokemon_red.gb" relative to the working directory (it ignores the
# rom_path argument entirely - see comment in pokegym/pyboy_binding.py).
# Run this script with CWD = v3_pokegym/, which has pokemon_red.gb symlinked
# to the real ROM in the parent directory.
def make_env(rank, seed=0):
    def _init():
        env = GymnasiumEnvAdapter(Environment(headless=True))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    sess_path = Path("runs_pokegym")
    sess_path.mkdir(exist_ok=True)

    num_cpu = 12  # leave a couple of the 14 M4 Pro cores for OS/main process
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = RNDRewardWrapper(env, intrinsic_coef=0.5, device="cpu")

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000, save_path=sess_path, name_prefix="poke_pokegym"
    )
    callbacks = CallbackList([checkpoint_callback, RewardComponentCallback(log_freq=1000)])

    model = PPO(
        "MultiInputPolicy",  # observation_space is a Dict (screen + team_quality/gap scalars)
        env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        n_epochs=3,
        gamma=0.998,
        ent_coef=0.01,
        device="cpu",  # no CUDA on Apple Silicon; try "mps" once this is confirmed stable
        tensorboard_log=sess_path,
    )

    model.learn(total_timesteps=100_000_000, callback=callbacks, tb_log_name="poke_pokegym_ppo")
