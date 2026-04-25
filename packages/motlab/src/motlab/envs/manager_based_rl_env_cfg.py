"""Config dataclass for :class:`ManagerBasedRLEnv`."""

from __future__ import annotations

from typing import Any

from motlab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from motlab.utils.configclass import configclass


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """RL env: adds rewards, terminations, episode length, optional commands."""

    rewards: Any = None
    terminations: Any = None
    commands: Any = None
    curriculum: Any = None
    episode_length_s: float = 5.0
    is_finite_horizon: bool = False
