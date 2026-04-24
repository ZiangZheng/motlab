"""Reward MDP functions for locomotion tasks (torch).

All returns are shape ``(num_envs,)`` float32 on ``env.device``. Functions
are pure — no internal state; the ``env`` arg provides accessors. The
declarative :class:`motlab_envs.managers.cfg.RewardTermCfg` supplies
weights, so these return *unweighted* signals.
"""

from __future__ import annotations

from typing import Any

import torch

from motlab_envs.math import quaternion as quat


# ---- Shared helpers -----------------------------------------------------
def _quat_wxyz(env: Any) -> torch.Tensor:
    return quat.xyzw_to_wxyz(env.base_quat_xyzw())


def _base_lin_vel_body(env: Any) -> torch.Tensor:
    return quat.rotate_inverse(_quat_wxyz(env), env.base_lin_vel_world())


def _base_ang_vel_body(env: Any) -> torch.Tensor:
    return quat.rotate_inverse(_quat_wxyz(env), env.base_ang_vel_world())


# ---- Velocity-tracking rewards -----------------------------------------
def track_lin_vel_xy_exp(env: Any, command_name: str = "twist", std: float = 0.5) -> torch.Tensor:
    """Exponential of negative xy-plane linear-velocity tracking error."""
    cmd = env.command(command_name)[..., :2]
    lin_xy = _base_lin_vel_body(env)[..., :2]
    err = torch.sum((cmd - lin_xy) ** 2, dim=-1)
    return torch.exp(-err / (std * std)).to(torch.float32)


def track_ang_vel_z_exp(env: Any, command_name: str = "twist", std: float = 0.5) -> torch.Tensor:
    """Exponential of negative yaw angular-velocity tracking error."""
    cmd = env.command(command_name)[..., 2]
    ang_z = _base_ang_vel_body(env)[..., 2]
    err = (cmd - ang_z) ** 2
    return torch.exp(-err / (std * std)).to(torch.float32)


# ---- Penalties on base motion ------------------------------------------
def lin_vel_z_l2(env: Any) -> torch.Tensor:
    """Squared vertical linear velocity (penalty — weight should be negative)."""
    v = _base_lin_vel_body(env)[..., 2]
    return (v * v).to(torch.float32)


def ang_vel_xy_l2(env: Any) -> torch.Tensor:
    """Squared roll + pitch angular velocity."""
    w = _base_ang_vel_body(env)[..., :2]
    return torch.sum(w * w, dim=-1).to(torch.float32)


def flat_orientation_l2(env: Any) -> torch.Tensor:
    """Squared deviation of gravity's x/y components (0 when upright)."""
    q_wxyz = _quat_wxyz(env)
    gravity = torch.zeros_like(q_wxyz[..., :3])
    gravity[..., 2] = -1.0
    proj = quat.rotate_inverse(q_wxyz, gravity)
    return torch.sum(proj[..., :2] ** 2, dim=-1).to(torch.float32)


# ---- Action / joint smoothness penalties -------------------------------
def action_rate_l2(env: Any) -> torch.Tensor:
    """Squared change in action between the last two steps."""
    delta = env.last_actions() - env.prev_actions()
    return torch.sum(delta * delta, dim=-1).to(torch.float32)


def joint_torques_l2(env: Any) -> torch.Tensor:
    """Squared joint torques."""
    t = env.last_torques()
    return torch.sum(t * t, dim=-1).to(torch.float32)


def joint_acc_l2(env: Any, ctrl_dt: float | None = None) -> torch.Tensor:
    """Squared joint acceleration, approximated via finite differences."""
    dt = ctrl_dt if ctrl_dt is not None else env.cfg.ctrl_dt
    accel = (env.joint_vel() - env.last_joint_vel()) / dt
    return torch.sum(accel * accel, dim=-1).to(torch.float32)


# ---- Liveness -----------------------------------------------------------
def is_alive(env: Any) -> torch.Tensor:
    """Constant 1.0 every step the env is still running."""
    return torch.ones((env.num_envs,), dtype=torch.float32, device=env.device)
