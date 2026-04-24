"""Go1 velocity-tracking env — showcases motlab's Manager-lite + PDActuator.

Observation (48-d):
  base_lin_vel (3, body frame, scale ``norm.lin_vel``)
  base_ang_vel (3, body frame, scale ``norm.ang_vel``)
  projected_gravity (3)
  velocity_command (3)
  joint_pos - default (12, scale ``norm.dof_pos``)
  joint_vel (12, scale ``norm.dof_vel``)
  last_actions (12)

Actions: 12 raw policy outputs → PDActuator → joint torques.

dof_pos layout (MotrixSim free-joint convention): [xyz | quat_xyzw | 12 joints].
"""

from __future__ import annotations

import gymnasium as gym
import torch

from motlab_envs import registry
from motlab_envs.asset_zoo.robots.unitree_go1 import (
    ACTION_SCALE,
    DAMPING,
    DEFAULT_JOINT_ANGLES,
    INIT_BASE_POS,
    INIT_BASE_QUAT_XYZW,
    STIFFNESS,
    TORQUE_LIMIT,
)
from motlab_envs.engine import motrix
from motlab_envs.env import TensorEnvState, TorchEnv
from motlab_envs.managers import (
    ObservationManager,
    RewardManager,
    TerminationManager,
    VelocityCommandManager,
)
from motlab_envs.math import quaternion as quat
from motlab_envs.sim2real.actuator import PDActuator

from .cfg import Go1VelocityEnvCfg

_NUM_JOINTS = 12
_FREEJOINT_POS_DIM = 7  # xyz + quat(xyzw)
_FREEJOINT_VEL_DIM = 6  # lin + ang


@registry.env("go1-velocity", sim_backend="torch")
class Go1VelocityEnv(TorchEnv):
    """Velocity-tracking PPO target for the Unitree Go1 quadruped."""

    _cfg: Go1VelocityEnvCfg

    def __init__(self, cfg: Go1VelocityEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        dev = self._device

        self._num_dof_pos = self._model.num_dof_pos  # 19
        self._num_dof_vel = self._model.num_dof_vel  # 18
        assert self._num_dof_pos == _FREEJOINT_POS_DIM + _NUM_JOINTS, (
            f"Unexpected dof_pos layout: {self._num_dof_pos}"
        )

        self._default_joint_angles = torch.as_tensor(
            DEFAULT_JOINT_ANGLES, dtype=torch.float32, device=dev,
        )
        self._default_dof_pos_vec = torch.as_tensor(
            (*INIT_BASE_POS, *INIT_BASE_QUAT_XYZW, *DEFAULT_JOINT_ANGLES),
            dtype=torch.float32, device=dev,
        )

        self._pd = PDActuator(
            num_envs=num_envs,
            num_actuators=_NUM_JOINTS,
            stiffness=STIFFNESS,
            damping=DAMPING,
            default_angles=DEFAULT_JOINT_ANGLES,
            action_scale=ACTION_SCALE,
            latency_steps=cfg.domain_rand.action_latency_steps,
            action_noise_scale=cfg.domain_rand.action_noise_scale,
            torque_limit=TORQUE_LIMIT,
            device=dev,
        )

        self._cmd = VelocityCommandManager(
            num_envs=num_envs,
            ctrl_dt=cfg.ctrl_dt,
            resample_seconds=cfg.commands.resample_seconds,
            low=cfg.commands.vel_limit[0],
            high=cfg.commands.vel_limit[1],
            device=dev,
        )
        self._cmd.reset_all()

        self._last_actions = torch.zeros((num_envs, _NUM_JOINTS), dtype=torch.float32, device=dev)
        self._prev_actions = torch.zeros_like(self._last_actions)
        self._last_joint_vel = torch.zeros_like(self._last_actions)
        self._last_torques = torch.zeros_like(self._last_actions)
        self._step_counter = 0

        self._obs_mgr = self._build_obs_manager()
        self._rew_mgr = self._build_reward_manager()
        self._term_mgr = self._build_termination_manager()

        obs_dim = self._probe_obs_dim()
        self._observation_space = gym.spaces.Box(
            -float("inf"), float("inf"), (obs_dim,), dtype="float32",
        )
        self._action_space = gym.spaces.Box(-1.0, 1.0, (_NUM_JOINTS,), dtype="float32")

    # ---- Spaces ---------------------------------------------------------
    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    # ---- Low-level indexing helpers ------------------------------------
    def _dof_pos(self, data) -> torch.Tensor:
        v = motrix.dof_pos_view(data)
        return v if v.device == self._device else v.to(self._device)

    def _dof_vel(self, data) -> torch.Tensor:
        v = motrix.dof_vel_view(data)
        return v if v.device == self._device else v.to(self._device)

    def _base_pos(self, data) -> torch.Tensor:
        return self._dof_pos(data)[..., 0:3]

    def _base_quat_xyzw(self, data) -> torch.Tensor:
        return self._dof_pos(data)[..., 3:7]

    def _base_lin_vel_world(self, data) -> torch.Tensor:
        return self._dof_vel(data)[..., 0:3]

    def _base_ang_vel_world(self, data) -> torch.Tensor:
        return self._dof_vel(data)[..., 3:6]

    def _joint_pos(self, data) -> torch.Tensor:
        return self._dof_pos(data)[..., _FREEJOINT_POS_DIM:]

    def _joint_vel(self, data) -> torch.Tensor:
        return self._dof_vel(data)[..., _FREEJOINT_VEL_DIM:]

    def _projected_gravity(self, data) -> torch.Tensor:
        q_wxyz = quat.xyzw_to_wxyz(self._base_quat_xyzw(data))
        gravity = torch.zeros_like(q_wxyz[..., :3])
        gravity[..., 2] = -1.0
        return quat.rotate_inverse(q_wxyz, gravity)

    def _base_lin_vel_body(self, data) -> torch.Tensor:
        q_wxyz = quat.xyzw_to_wxyz(self._base_quat_xyzw(data))
        return quat.rotate_inverse(q_wxyz, self._base_lin_vel_world(data))

    def _base_ang_vel_body(self, data) -> torch.Tensor:
        q_wxyz = quat.xyzw_to_wxyz(self._base_quat_xyzw(data))
        return quat.rotate_inverse(q_wxyz, self._base_ang_vel_world(data))

    # ---- Manager wiring -------------------------------------------------
    def _build_obs_manager(self) -> ObservationManager:
        norm = self._cfg.normalization
        noise = self._cfg.domain_rand.obs_noise_scale

        def f_lin_vel(self_):
            return self_._base_lin_vel_body(self_._state.data)

        def f_ang_vel(self_):
            return self_._base_ang_vel_body(self_._state.data)

        def f_grav(self_):
            return self_._projected_gravity(self_._state.data)

        def f_cmd(self_):
            return self_._cmd.commands

        def f_joint_pos(self_):
            return self_._joint_pos(self_._state.data) - self_._default_joint_angles

        def f_joint_vel(self_):
            return self_._joint_vel(self_._state.data)

        def f_actions(self_):
            return self_._last_actions

        return (
            ObservationManager()
            .add("base_lin_vel", f_lin_vel, scale=norm.lin_vel, noise_scale=noise)
            .add("base_ang_vel", f_ang_vel, scale=norm.ang_vel, noise_scale=noise)
            .add("projected_gravity", f_grav, noise_scale=noise)
            .add("velocity_command", f_cmd, scale=norm.lin_vel)
            .add("joint_pos", f_joint_pos, scale=norm.dof_pos, noise_scale=noise)
            .add("joint_vel", f_joint_vel, scale=norm.dof_vel, noise_scale=noise)
            .add("last_actions", f_actions)
        )

    def _build_reward_manager(self) -> RewardManager:
        scales = self._cfg.reward.scales
        sigma = self._cfg.reward.tracking_sigma
        dt = self._cfg.ctrl_dt

        def r_track_lin_xy(self_):
            cmd = self_._cmd.commands[:, 0:2]
            lin = self_._base_lin_vel_body(self_._state.data)[..., 0:2]
            err = torch.sum((cmd - lin) ** 2, dim=-1)
            return torch.exp(-err / sigma).to(torch.float32)

        def r_track_ang_z(self_):
            cmd = self_._cmd.commands[:, 2]
            ang = self_._base_ang_vel_body(self_._state.data)[..., 2]
            err = (cmd - ang) ** 2
            return torch.exp(-err / sigma).to(torch.float32)

        def r_lin_z(self_):
            return self_._base_lin_vel_body(self_._state.data)[..., 2] ** 2

        def r_ang_xy(self_):
            ang = self_._base_ang_vel_body(self_._state.data)[..., 0:2]
            return torch.sum(ang ** 2, dim=-1)

        def r_orientation(self_):
            g = self_._projected_gravity(self_._state.data)[..., 0:2]
            return torch.sum(g ** 2, dim=-1)

        def r_joint_torques(self_):
            return torch.sum(self_._last_torques ** 2, dim=-1)

        def r_joint_accel(self_):
            qdot = self_._joint_vel(self_._state.data)
            accel = (qdot - self_._last_joint_vel) / dt
            return torch.sum(accel ** 2, dim=-1)

        def r_action_rate(self_):
            d = self_._last_actions - self_._prev_actions
            return torch.sum(d ** 2, dim=-1)

        def r_alive(self_):
            return torch.ones((self_._num_envs,), dtype=torch.float32, device=self_._device)

        return (
            RewardManager()
            .add("track_lin_vel_xy", r_track_lin_xy, weight=scales["track_lin_vel_xy"])
            .add("track_ang_vel_z", r_track_ang_z, weight=scales["track_ang_vel_z"])
            .add("lin_vel_z", r_lin_z, weight=scales["lin_vel_z"])
            .add("ang_vel_xy", r_ang_xy, weight=scales["ang_vel_xy"])
            .add("orientation", r_orientation, weight=scales["orientation"])
            .add("joint_torques", r_joint_torques, weight=scales["joint_torques"])
            .add("joint_accel", r_joint_accel, weight=scales["joint_accel"])
            .add("action_rate", r_action_rate, weight=scales["action_rate"])
            .add("alive", r_alive, weight=scales["alive"])
        )

    def _build_termination_manager(self) -> TerminationManager:
        min_h = self._cfg.base_min_height
        tilt_gz = self._cfg.gravity_z_terminate_threshold

        def t_base_low(self_):
            return self_._base_pos(self_._state.data)[..., 2] < min_h

        def t_fallen(self_):
            gz = self_._projected_gravity(self_._state.data)[..., 2]
            return gz > tilt_gz

        return (
            TerminationManager()
            .add("base_low", t_base_low)
            .add("fallen", t_fallen)
        )

    # ---- obs dim probe via a single-env batched data -------------------
    def _probe_obs_dim(self) -> int:
        probe_data = motrix.make_batched_data(self._model, 1)
        motrix.set_dof_pos(probe_data, self._model, self._default_dof_pos_vec[None, :])

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx._state = type("S", (), {"data": probe_data})()
        ctx._cmd = type("C", (), {"commands": torch.zeros((1, 3), dtype=torch.float32, device=self._device)})()
        ctx._last_actions = torch.zeros((1, _NUM_JOINTS), dtype=torch.float32, device=self._device)
        ctx._default_joint_angles = self._default_joint_angles
        ctx._device = self._device
        # Bind helper methods so they see ``ctx``'s state.
        for name in (
            "_base_lin_vel_body", "_base_ang_vel_body", "_projected_gravity",
            "_joint_pos", "_joint_vel", "_base_quat_xyzw", "_base_pos",
            "_base_lin_vel_world", "_base_ang_vel_world", "_dof_pos", "_dof_vel",
        ):
            setattr(ctx, name, getattr(self, name).__get__(ctx))
        return self._obs_mgr.dim(ctx)

    # ---- TorchEnv hooks -------------------------------------------------
    def apply_action(self, actions: torch.Tensor, state: TensorEnvState) -> TensorEnvState:
        actions = torch.clamp(actions, -1.0, 1.0).to(torch.float32)
        self._prev_actions = self._last_actions.clone()
        self._last_actions = actions.clone()

        joint_pos = self._joint_pos(state.data)
        joint_vel = self._joint_vel(state.data)
        torque = self._pd.compute(actions, joint_pos, joint_vel)
        self._last_torques = torque.clone()
        motrix.set_actuator_ctrls(state.data, torque)

        self._cmd.maybe_resample(self._step_counter)
        return state

    def update_state(self, state: TensorEnvState) -> TensorEnvState:
        self._step_counter += 1

        obs = self._obs_mgr.compute(self).to(torch.float32)
        terminated, _ = self._term_mgr.compute(self)
        reward, _ = self._rew_mgr.compute(self, terminated=terminated)

        state.obs = obs
        state.reward = reward.to(torch.float32)
        state.terminated = terminated
        self._last_joint_vel = self._joint_vel(state.data).clone()
        return state

    def reset(self, data) -> tuple[torch.Tensor, dict]:
        cfg = self._cfg
        n = data.shape[0]
        dev = self._device

        init = self._default_dof_pos_vec[None, :].expand(n, -1).clone()
        init[:, 0:2] += (torch.rand((n, 2), device=dev) * 2 - 1) * cfg.reset_base_pos_noise
        init[:, _FREEJOINT_POS_DIM:] += (
            torch.rand((n, _NUM_JOINTS), device=dev) * 2 - 1
        ) * cfg.reset_joint_noise

        dof_vel = torch.zeros((n, self._num_dof_vel), dtype=torch.float32, device=dev)

        data.reset(self._model)
        motrix.set_dof_vel(data, dof_vel)
        motrix.set_dof_pos(data, self._model, init)

        if n == self._num_envs:
            self._pd.reset()
            self._cmd.reset_all()
            self._last_actions.zero_()
            self._prev_actions.zero_()
            self._last_joint_vel.zero_()
            self._last_torques.zero_()

        obs_dim = int(self._observation_space.shape[0])
        return torch.zeros((n, obs_dim), dtype=torch.float32, device=dev), {}
