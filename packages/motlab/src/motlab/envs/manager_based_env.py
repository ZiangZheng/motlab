"""Non-RL manager-based env (used as a base for :class:`ManagerBasedRLEnv`)."""

from __future__ import annotations

import torch

from motlab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from motlab.managers.action_manager import ActionManager
from motlab.managers.event_manager import EventManager
from motlab.managers.observation_manager import ObservationManager
from motlab.scene.interactive_scene import InteractiveScene


class ManagerBasedEnv:
    """Owns the scene + actions/observations/events managers.  Provides
    :meth:`step`, :meth:`reset`, :meth:`close`."""

    cfg: ManagerBasedEnvCfg

    def __init__(self, cfg: ManagerBasedEnvCfg, device: torch.device | str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.sim.device)
        if cfg.scene is None:
            raise ValueError("ManagerBasedEnvCfg.scene must be set")
        self.num_envs = int(cfg.scene.num_envs)

        self.scene = InteractiveScene(cfg.scene, device=self.device)

        self.physics_dt: float = float(cfg.sim.dt)
        self.step_dt: float = self.physics_dt * int(cfg.decimation)
        self.common_step_counter: int = 0

        self.action_manager = ActionManager(cfg.actions, self) if cfg.actions is not None else None
        self.event_manager = EventManager(cfg.events, self) if cfg.events is not None else None
        if self.event_manager is not None:
            self.event_manager.apply("startup", env_ids=None)

        self.observation_manager: ObservationManager | None = None  # filled by _finalize_managers
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Subclasses (RL env) construct extra managers here, then call _finalize_managers.
        if type(self) is ManagerBasedEnv:
            self._finalize_managers()

    def _finalize_managers(self) -> None:
        if self.cfg.observations is not None:
            self.observation_manager = ObservationManager(self.cfg.observations, self)

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self.action_manager.total_action_dim if self.action_manager else 0

    @property
    def observation_dims(self) -> dict[str, tuple[int, ...]]:
        return self.observation_manager.group_obs_dim if self.observation_manager else {}

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        if self.event_manager is not None:
            self.event_manager.apply("reset", env_ids=env_ids)
        if self.action_manager is not None:
            self.action_manager.reset(env_ids)

        return self._compute_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict]:
        action = action.to(self.device)
        if self.action_manager is not None:
            self.action_manager.process_action(action)

        for _ in range(int(self.cfg.decimation)):
            if self.action_manager is not None:
                self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.scene.step()
            self.scene.update(self.physics_dt)

        self.common_step_counter += 1
        self.episode_length_buf += 1

        if self.event_manager is not None:
            self.event_manager.apply("interval", env_ids=None, dt=self.step_dt)

        return self._compute_obs(), {}

    # ------------------------------------------------------------------
    def _compute_obs(self) -> dict[str, torch.Tensor]:
        if self.observation_manager is None:
            return {}
        return self.observation_manager.compute()

    def close(self) -> None:
        self.scene = None  # type: ignore[assignment]
