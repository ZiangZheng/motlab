"""Observation MDP functions for locomotion tasks (torch).

All returns are shape ``(num_envs, D)`` float32 on ``env.device``.
``env`` is expected to be a
:class:`~motlab_envs.manager_env.ManagerBasedTorchEnv` subclass (or any
env exposing ``base_*`` / ``joint_*`` / ``command()`` accessors).
"""

from __future__ import annotations

from typing import Any

import torch

from motlab_envs.math import quaternion as quat


# ---- Quaternion convention shim (MotrixSim is xyzw) ---------------------
def _quat_wxyz(env: Any) -> torch.Tensor:
    return quat.xyzw_to_wxyz(env.base_quat_xyzw())


# ---- Observation terms ---------------------------------------------------
def base_lin_vel_body(env: Any) -> torch.Tensor:
    """Linear velocity of the base in the base frame."""
    return quat.rotate_inverse(_quat_wxyz(env), env.base_lin_vel_world()).to(torch.float32)


def base_ang_vel_body(env: Any) -> torch.Tensor:
    """Angular velocity of the base in the base frame."""
    return quat.rotate_inverse(_quat_wxyz(env), env.base_ang_vel_world()).to(torch.float32)


def projected_gravity(env: Any) -> torch.Tensor:
    """World gravity rotated into the base frame (useful for balance)."""
    q_wxyz = _quat_wxyz(env)
    gravity = torch.zeros_like(q_wxyz[..., :3])
    gravity[..., 2] = -1.0
    return quat.rotate_inverse(q_wxyz, gravity).to(torch.float32)


def joint_pos_rel(env: Any, default_angles: Any) -> torch.Tensor:
    """Joint positions minus their default pose."""
    default = torch.as_tensor(default_angles, dtype=torch.float32, device=env.device)
    return (env.joint_pos().to(torch.float32) - default).to(torch.float32)


def joint_vel(env: Any) -> torch.Tensor:
    return env.joint_vel().to(torch.float32)


def last_actions(env: Any) -> torch.Tensor:
    return env.last_actions().to(torch.float32)


def velocity_command(env: Any, name: str = "twist") -> torch.Tensor:
    return env.command(name).to(torch.float32)
