"""RL env: adds rewards / terminations / commands / curriculum."""

from __future__ import annotations

import math

import torch

from motlab.envs.manager_based_env import ManagerBasedEnv
from motlab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from motlab.managers.command_manager import CommandManager
from motlab.managers.curriculum_manager import CurriculumManager
from motlab.managers.reward_manager import RewardManager
from motlab.managers.termination_manager import TerminationManager


class ManagerBasedRLEnv(ManagerBasedEnv):
    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, device: torch.device | str | None = None) -> None:
        super().__init__(cfg, device=device)

        self.max_episode_length_s: float = cfg.episode_length_s
        self.max_episode_length: int = int(math.ceil(cfg.episode_length_s / self.step_dt))

        self.command_manager = (
            CommandManager(cfg.commands, self) if cfg.commands is not None else None
        )
        self.reward_manager = (
            RewardManager(cfg.rewards, self) if cfg.rewards is not None else None
        )
        self.termination_manager = (
            TerminationManager(cfg.terminations, self) if cfg.terminations is not None else None
        )
        self.curriculum_manager = (
            CurriculumManager(cfg.curriculum, self) if cfg.curriculum is not None else None
        )

        # ObservationManager is built last because its terms may reference
        # commands / rewards.
        self._finalize_managers()

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        info: dict = {"log": {}}
        # Reward bookkeeping (per episode means) before zeroing.
        if self.reward_manager is not None:
            info["log"].update(self.reward_manager.reset(env_ids))
        if self.termination_manager is not None:
            info["log"].update(self.termination_manager.reset(env_ids))
        if self.command_manager is not None:
            info["log"].update(self.command_manager.reset(env_ids))
        if self.curriculum_manager is not None:
            info["log"].update(self.curriculum_manager.reset(env_ids))

        obs, _ = super().reset(env_ids)
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, _ = super().step(action)

        if self.command_manager is not None:
            self.command_manager.compute(self.step_dt)

        # Termination + reward
        if self.termination_manager is not None:
            self.termination_manager.compute()
            terminated = self.termination_manager.terminated
            truncated = self.termination_manager.truncated
        else:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros_like(terminated)

        # Time-out via episode length
        time_out = self.episode_length_buf >= self.max_episode_length
        truncated = truncated | time_out

        if self.reward_manager is not None:
            reward = self.reward_manager.compute(dt=self.step_dt)
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        info: dict = {"log": {}}
        done = terminated | truncated
        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            _, reset_info = self.reset(reset_ids)
            info["log"].update(reset_info.get("log", {}))
            if self.curriculum_manager is not None:
                info["log"].update(self.curriculum_manager.compute(reset_ids))
            obs = self._compute_obs()

        return obs, reward, terminated, truncated, info
