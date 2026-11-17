from dataclasses import dataclass

FIXES = ("fix_catch_refarm", "truncate_time_limit", "fix_box_struct")


@dataclass
class EnvConfig:
    fix_catch_refarm: bool = True
    truncate_time_limit: bool = True
    fix_box_struct: bool = True
    go_explore: str = "latest"
    loop_penalty: str = "no_battle"
    level_reward: str = "trained"
    battle_reward: str = "ko"
    team_reward: str = "off"
    battle_obs: bool = True
    max_episode_steps: int = 20480
    traj_stride: int = 4
    run_dir: str = ""
    rank: int = 0

    @classmethod
    def baseline(cls, **kwargs):
        return cls(
            **{name: False for name in FIXES},
            go_explore="legacy",
            loop_penalty="legacy",
            level_reward="legacy",
            battle_reward="legacy",
            team_reward="legacy",
            battle_obs=False,
            **kwargs,
        )

    @classmethod
    def r1(cls, **kwargs):
        return cls(level_reward="legacy", battle_reward="legacy", team_reward="legacy", battle_obs=False, **kwargs)
