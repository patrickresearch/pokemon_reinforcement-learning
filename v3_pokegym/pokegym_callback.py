import csv
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

TB_STATS = (
    "badges", "maps_explored", "levels_sum", "highest_pokemon_level", "party_size",
    "caught_pokemon", "seen_pokemon", "deaths", "event", "money", "go_explore_resets",
    "kos", "trained_levels",
)


class RewardComponentCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._sums = {}
        self._count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            components = info.get("reward_components")
            if not components:
                continue
            self._count += 1
            for key, value in components.items():
                self._sums[key] = self._sums.get(key, 0.0) + value

        if self._count >= self.log_freq:
            for key, total in self._sums.items():
                self.logger.record(f"reward_components/{key}", total / self._count)
            self._sums = {}
            self._count = 0
        return True


class RunLogCallback(BaseCallback):
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = Path(run_dir)

    def _on_training_start(self):
        self.t0 = time.time()
        (self.run_dir / "heatmaps").mkdir(exist_ok=True)
        self.milestone_file = open(self.run_dir / "milestones.csv", "a", newline="")
        self.milestones = csv.writer(self.milestone_file)
        if self.milestone_file.tell() == 0:
            self.milestones.writerow(["wall_s", "global_step", "env", "env_step", "milestone"])
        self.stats_file = open(self.run_dir / "stats.jsonl", "a")

    def _on_step(self) -> bool:
        wall = round(time.time() - self.t0, 1)
        for env, info in enumerate(self.locals["infos"]):
            for name, env_step in info.get("milestones", ()):
                self.milestones.writerow([wall, self.num_timesteps, env, env_step, name])
                self.milestone_file.flush()
            stats = info.get("stats")
            if not stats:
                continue
            row = {
                k: float(v) for k, v in stats.items()
                if isinstance(v, (int, float, np.integer, np.floating))
            }
            for key in TB_STATS:
                if key in row:
                    self.logger.record_mean(f"game/{key}", row[key])
            row.update(wall_s=wall, global_step=self.num_timesteps, env=env)
            self.stats_file.write(json.dumps(row) + "\n")
            self.stats_file.flush()
            np.save(
                self.run_dir / "heatmaps" / f"env{env}.npy",
                np.asarray(info["pokemon_exploration_map"], dtype=np.int32),
            )
        return True

    def _on_training_end(self):
        self.milestone_file.close()
        self.stats_file.close()
