from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentCallback(BaseCallback):
    """Averages the per-step `reward_components` dict (set in
    pokegym/environment.py's step()) across every step since the last log
    and writes each component to TensorBoard under reward_components/*, so
    catch/level-up/battle-win/etc. bonuses are visible as separate curves
    instead of only the single combined scalar reward SB3 logs by default.
    """

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
