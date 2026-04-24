"""Manager-based batched torch env.

Complements :class:`motlab_envs.env.TorchEnv` (direct style — tasks
override ``apply_action`` / ``update_state`` / ``reset`` themselves).

:class:`ManagerBasedTorchEnv` reads declarative term configs from the
task's ``EnvCfg`` (``observations`` / ``rewards`` / ``terminations`` /
``commands`` / ``actions``), builds lite managers from them, and runs a
standard compute flow every step. Subclasses typically only have to:

- Set ``_freejoint_pos_dim`` / ``_freejoint_vel_dim`` class attributes so
  MDP helpers can slice ``dof_pos`` / ``dof_vel``.
- Optionally provide an ``init_pose_fn`` callable in the cfg (otherwise
  the base broadcasts ``model.compute_init_dof_pos()``).

Mirrors isaaclab's ``ManagerBasedRLEnv``, mjlab's ``ManagerBasedRlEnv``
and genesislab — with torch tensors end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import gymnasium as gym
import torch

from motlab_envs.base import EnvCfg
from motlab_envs.engine import motrix
from motlab_envs.env import TensorEnvState, TorchEnv
from motlab_envs.managers import (
    ActionsCfg,
    CommandsCfgMB,
    ObservationManager,
    ObservationsCfg,
    PDActionCfg,
    RewardManager,
    RewardsCfg,
    TerminationManager,
    TerminationsCfg,
)
from motlab_envs.sim2real.actuator import PDActuator


# ---------------------------------------------------------------------------
# Base manager-based EnvCfg — tasks subclass and fill in the term dicts.
# ---------------------------------------------------------------------------
@dataclass
class ManagerBasedEnvCfg(EnvCfg):
    """Config shape expected by :class:`ManagerBasedTorchEnv`."""

    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    commands: CommandsCfgMB = field(default_factory=CommandsCfgMB)
    actions: ActionsCfg = field(default_factory=ActionsCfg)

    # Optional hook to populate the initial ``dof_pos`` vector at reset.
    # Signature: ``init_pose_fn(env, n) -> (n, num_dof_pos)`` torch tensor.
    # Defaults to broadcasting ``model.compute_init_dof_pos()``.
    init_pose_fn: Optional[Callable] = None

    # Reset noise applied on top of ``init_pose_fn``.
    reset_joint_noise: float = 0.0
    reset_base_pos_noise: float = 0.0


# ---------------------------------------------------------------------------
# Env class
# ---------------------------------------------------------------------------
class ManagerBasedTorchEnv(TorchEnv):
    """TorchEnv whose step logic is assembled from manager term configs."""

    _cfg: ManagerBasedEnvCfg

    # Subclasses override to describe DOF layout. Defaults = fixed-base.
    _freejoint_pos_dim: int = 0
    _freejoint_vel_dim: int = 0

    def __init__(self, cfg: ManagerBasedEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        # DOF layout — assume num_joints = all non-freejoint dofs unless the
        # action cfg specifies a narrower joint list.
        pd_cfg: PDActionCfg = cfg.actions.joint_pd
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_joints: int = len(pd_cfg.joint_names) or (
            self._num_dof_pos - self._freejoint_pos_dim
        )

        # Optional PD stack.
        self._pd: Optional[PDActuator] = None
        if pd_cfg.stiffness is not None:
            self._pd = PDActuator(
                num_envs=num_envs,
                num_actuators=self._num_joints,
                stiffness=pd_cfg.stiffness,
                damping=pd_cfg.damping,
                default_angles=(
                    pd_cfg.default_angles if pd_cfg.default_angles is not None else 0.0
                ),
                action_scale=pd_cfg.action_scale,
                latency_steps=pd_cfg.latency_steps,
                action_noise_scale=pd_cfg.action_noise_scale,
                torque_limit=pd_cfg.torque_limit,
                device=self._device,
            )

        # Per-step buffers (all on the task device).
        self._last_actions = torch.zeros(
            (num_envs, self._num_joints), dtype=torch.float32, device=self._device,
        )
        self._prev_actions = torch.zeros_like(self._last_actions)
        self._last_torques = torch.zeros_like(self._last_actions)
        self._last_joint_vel = torch.zeros_like(self._last_actions)
        self._step_counter = 0
        # Set by ``_reset_done_envs`` so ``reset`` can target the right slots.
        self._current_reset_mask: Optional[torch.Tensor] = None

        # Commands: generator funcs run per-env on reset / resample.
        self._commands: dict[str, torch.Tensor] = {}
        self._command_resample_steps: dict[str, int] = {}
        self._init_commands_storage()

        # Managers.
        self._obs_mgr = self._build_observation_manager(cfg.observations)
        self._rew_mgr = self._build_reward_manager(cfg.rewards)
        self._term_mgr = self._build_termination_manager(cfg.terminations)

        # Action space.
        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._num_joints,), dtype="float32")

        # Observation space depends on term outputs — resolved lazily on
        # first ``init_state`` call.
        self._observation_space: gym.spaces.Box = gym.spaces.Box(
            -float("inf"), float("inf"), (0,), dtype="float32",
        )
        self._obs_dim_resolved: bool = False

    # ---- Gym spaces -----------------------------------------------------
    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    # ---- DOF accessors used by MDP functions ---------------------------
    def _dof_pos_tensor(self, data=None) -> torch.Tensor:
        data = self._state.data if data is None else data
        view = motrix.dof_pos_view(data)
        if view.device != self._device:
            view = view.to(self._device)
        return view

    def _dof_vel_tensor(self, data=None) -> torch.Tensor:
        data = self._state.data if data is None else data
        view = motrix.dof_vel_view(data)
        if view.device != self._device:
            view = view.to(self._device)
        return view

    def base_pos(self, data=None) -> torch.Tensor:
        return self._dof_pos_tensor(data)[..., 0:3]

    def base_quat_xyzw(self, data=None) -> torch.Tensor:
        return self._dof_pos_tensor(data)[..., 3:7]

    def base_lin_vel_world(self, data=None) -> torch.Tensor:
        return self._dof_vel_tensor(data)[..., 0:3]

    def base_ang_vel_world(self, data=None) -> torch.Tensor:
        return self._dof_vel_tensor(data)[..., 3:6]

    def joint_pos(self, data=None) -> torch.Tensor:
        return self._dof_pos_tensor(data)[..., self._freejoint_pos_dim:]

    def joint_vel(self, data=None) -> torch.Tensor:
        return self._dof_vel_tensor(data)[..., self._freejoint_vel_dim:]

    def last_actions(self) -> torch.Tensor:
        return self._last_actions

    def prev_actions(self) -> torch.Tensor:
        return self._prev_actions

    def last_torques(self) -> torch.Tensor:
        return self._last_torques

    def last_joint_vel(self) -> torch.Tensor:
        return self._last_joint_vel

    def pd_actuator(self) -> PDActuator:
        assert self._pd is not None, "No PDActuator configured (cfg.actions.joint_pd.stiffness is None)"
        return self._pd

    def command(self, name: str) -> torch.Tensor:
        return self._commands[name]

    # ---- Command wiring -------------------------------------------------
    def _init_commands_storage(self) -> None:
        """Allocate storage. Values filled in on first reset / resample."""
        for name, term_cfg in self._cfg.commands.terms.items():
            saved_n = self._num_envs
            self._num_envs = 1
            try:
                probe = term_cfg.func(self, **term_cfg.params)
            finally:
                self._num_envs = saved_n
            if not isinstance(probe, torch.Tensor):
                probe = torch.as_tensor(probe, dtype=torch.float32, device=self._device)
            assert probe.ndim == 2 and probe.shape[0] == 1, (
                f"Command {name!r}: expected (num_envs, D), got {tuple(probe.shape)}"
            )
            dim = int(probe.shape[1])
            self._commands[name] = torch.zeros(
                (self._num_envs, dim), dtype=torch.float32, device=self._device,
            )
            resample_steps = max(1, int(term_cfg.resample_seconds / self._cfg.ctrl_dt))
            self._command_resample_steps[name] = resample_steps

    def _resample_commands(self, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        for name, term_cfg in self._cfg.commands.terms.items():
            saved = self._num_envs
            self._num_envs = n
            try:
                values = term_cfg.func(self, **term_cfg.params)
            finally:
                self._num_envs = saved
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values, dtype=torch.float32, device=self._device)
            self._commands[name][mask] = values.to(
                dtype=self._commands[name].dtype, device=self._commands[name].device,
            )

    def _maybe_resample_commands(self) -> None:
        for name, period in self._command_resample_steps.items():
            if self._step_counter > 0 and self._step_counter % period == 0:
                mask = torch.ones((self._num_envs,), dtype=torch.bool, device=self._device)
                self._resample_commands(mask)

    # ---- Manager builders ----------------------------------------------
    def _build_observation_manager(self, cfg: ObservationsCfg) -> ObservationManager:
        mgr = ObservationManager()
        for name, term in cfg.terms.items():
            mgr.add(
                name=name,
                fn=_bind_mdp(term.func, term.params),
                scale=term.scale,
                noise_scale=term.noise_scale,
                clip=term.clip,
            )
        return mgr

    def _build_reward_manager(self, cfg: RewardsCfg) -> RewardManager:
        dt = self._cfg.ctrl_dt
        mgr = RewardManager(clip=cfg.clip, zero_on_termination=cfg.zero_on_termination)
        for name, term in cfg.terms.items():
            weight = term.weight * (dt if term.scale_by_dt else 1.0)
            mgr.add(name=name, fn=_bind_mdp(term.func, term.params), weight=weight)
        return mgr

    def _build_termination_manager(self, cfg: TerminationsCfg) -> TerminationManager:
        mgr = TerminationManager()
        for name, term in cfg.terms.items():
            mgr.add(
                name=name,
                fn=_bind_mdp(term.func, term.params),
                terminal=term.terminal,
            )
        return mgr

    # ---- TorchEnv lifecycle overrides -----------------------------------
    def _reset_done_envs(self) -> None:
        """Capture the done mask so ``reset`` can route per-env buffer clears."""
        if self._state is not None:
            self._current_reset_mask = self._state.done.clone()
        super()._reset_done_envs()
        self._current_reset_mask = None

    def init_state(self) -> TensorEnvState:
        """Resolve obs dim on first call, then delegate to :class:`TorchEnv`."""
        if self._state is not None and self._obs_dim_resolved:
            return self._state

        # First-time init: allocate a 0-dim obs buffer, reset, then re-run
        # obs terms to get the true dim.
        super().init_state()  # creates state.obs with 0 cols; runs reset()

        obs = self._obs_mgr.compute(self).to(torch.float32)
        self._observation_space = gym.spaces.Box(
            -float("inf"), float("inf"), (obs.shape[-1],), dtype="float32",
        )
        self._state.obs = obs
        self._obs_dim_resolved = True
        return self._state

    # ---- TorchEnv hooks -------------------------------------------------
    def apply_action(self, actions: torch.Tensor, state: TensorEnvState) -> TensorEnvState:
        actions = torch.clamp(actions, -1.0, 1.0).to(torch.float32)
        self._prev_actions = self._last_actions.clone()
        self._last_actions = actions.clone()

        if self._pd is not None:
            torque = self._pd.compute(actions, self.joint_pos(state.data), self.joint_vel(state.data))
        else:
            torque = actions
        self._last_torques = torque.clone()
        motrix.set_actuator_ctrls(state.data, torque)

        self._maybe_resample_commands()
        return state

    def update_state(self, state: TensorEnvState) -> TensorEnvState:
        self._step_counter += 1
        if self._term_mgr.terms:
            terminated, _ = self._term_mgr.compute(self)
        else:
            terminated = torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device)
        if self._rew_mgr.terms:
            reward, _ = self._rew_mgr.compute(self, terminated=terminated)
        else:
            reward = torch.zeros((self._num_envs,), dtype=torch.float32, device=self._device)
        obs = self._obs_mgr.compute(self).to(torch.float32)
        state.obs = obs
        state.reward = reward.to(torch.float32)
        state.terminated = terminated
        self._last_joint_vel = self.joint_vel(state.data).clone()
        return state

    def reset(self, data) -> tuple[torch.Tensor, dict]:
        cfg = self._cfg
        n = data.shape[0]
        if cfg.init_pose_fn is not None:
            init = cfg.init_pose_fn(self, n)
            if not isinstance(init, torch.Tensor):
                init = torch.as_tensor(init, dtype=torch.float32, device=self._device)
        else:
            init_vec = motrix.init_dof_pos_tensor(self._model, device=self._device)
            init = init_vec[None, :].expand(n, -1).clone()
        init = init.to(dtype=torch.float32, device=self._device).clone()

        if cfg.reset_joint_noise > 0:
            noise = (torch.rand((n, self._num_joints), device=self._device) * 2 - 1) * cfg.reset_joint_noise
            init[:, self._freejoint_pos_dim:] += noise
        if cfg.reset_base_pos_noise > 0 and self._freejoint_pos_dim >= 3:
            xy_noise = (torch.rand((n, 2), device=self._device) * 2 - 1) * cfg.reset_base_pos_noise
            init[:, 0:2] += xy_noise

        dof_vel = torch.zeros((n, self._num_dof_vel), dtype=torch.float32, device=self._device)

        data.reset(self._model)
        motrix.set_dof_vel(data, dof_vel)
        motrix.set_dof_pos(data, self._model, init)

        # Per-env buffer clears + command resample targeted to the exact
        # done mask when available; full-batch on first init.
        mask = self._current_reset_mask
        if mask is None:
            mask = torch.ones((self._num_envs,), dtype=torch.bool, device=self._device)
        self._last_actions[mask] = 0.0
        self._prev_actions[mask] = 0.0
        self._last_joint_vel[mask] = 0.0
        self._last_torques[mask] = 0.0
        if self._pd is not None and int(mask.sum().item()) == self._num_envs:
            self._pd.reset()
        self._resample_commands(mask)

        # Placeholder obs — base ``init_state`` will overwrite it on first call.
        obs_dim = int(self._observation_space.shape[0])
        return torch.zeros((n, obs_dim), dtype=torch.float32, device=self._device), {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bind_mdp(func: Callable, params: dict) -> Callable[[Any], torch.Tensor]:
    if not params:
        return func
    return lambda env: func(env, **params)
