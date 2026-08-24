"""Random Network Distillation intrinsic exploration reward.

A frozen, randomly-initialized "target" CNN and a trainable "predictor" CNN
both look at the (downsampled) screen; the predictor is trained to match the
target's output, and its prediction error is the intrinsic reward - novel
frames (which the predictor hasn't learned to reconstruct yet) score high,
familiar/repeated frames score low. Unlike the coordinate-based
exploration_reward in environment.py, this needs no game-state bookkeeping
at all and can pick up on subtler novelty (menus, battle screens, NPC
sprites) that (r, c, map_n) alone can't distinguish.

Implemented as a VecEnvWrapper around the SubprocVecEnv rather than inside
environment.py: the two small CNNs need a single batched forward/backward
pass across all parallel envs each step, which is far cheaper done once in
the main process than duplicated inside every PyBoy subprocess.
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnvWrapper

# Screen obs channel 0 downsampled by this stride before hitting the CNNs -
# RND only needs enough resolution to distinguish "seen this frame before"
# from "genuinely new", not full detail.
DOWNSAMPLE_STRIDE = 2
EMBED_DIM = 128


class _RNDNet(nn.Module):
    def __init__(self, in_h: int, in_w: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        conv_h, conv_w = (in_h + 3) // 4, (in_w + 3) // 4
        self.fc = nn.Linear(32 * conv_h * conv_w, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class RNDRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, intrinsic_coef: float = 0.5, learning_rate: float = 1e-4, device: str = "cpu"):
        super().__init__(venv)
        self.intrinsic_coef = intrinsic_coef
        self.device = torch.device(device)

        h, w = self.observation_space["screen"].shape[:2]
        self.in_h = (h + DOWNSAMPLE_STRIDE - 1) // DOWNSAMPLE_STRIDE
        self.in_w = (w + DOWNSAMPLE_STRIDE - 1) // DOWNSAMPLE_STRIDE

        self.target = _RNDNet(self.in_h, self.in_w).to(self.device)
        self.predictor = _RNDNet(self.in_h, self.in_w).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.reward_rms = RunningMeanStd(shape=())
        self._epsilon = 1e-8

    def _preprocess(self, screens: np.ndarray) -> torch.Tensor:
        # screens: (N, H, W, C) uint8 -> (N, 1, H//stride, W//stride) float32 in [0, 1]
        gray = screens[:, ::DOWNSAMPLE_STRIDE, ::DOWNSAMPLE_STRIDE, 0]
        t = torch.from_numpy(gray).to(self.device, dtype=torch.float32) / 255.0
        return t.unsqueeze(1)

    def _intrinsic_reward(self, obs) -> np.ndarray:
        screens = obs["screen"] if isinstance(obs, dict) else obs
        x = self._preprocess(screens)

        with torch.no_grad():
            target_out = self.target(x)
        predictor_out = self.predictor(x)

        per_env_mse = (predictor_out.detach() - target_out).pow(2).mean(dim=1).cpu().numpy()

        loss = nn.functional.mse_loss(predictor_out, target_out.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return per_env_mse

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        intrinsic = self._intrinsic_reward(obs)

        self.reward_rms.update(intrinsic)
        normalized_intrinsic = intrinsic / np.sqrt(self.reward_rms.var + self._epsilon)

        for i, info in enumerate(infos):
            info.setdefault("reward_components", {})["rnd_intrinsic"] = float(
                self.intrinsic_coef * normalized_intrinsic[i]
            )

        rewards = rewards + self.intrinsic_coef * normalized_intrinsic
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()
