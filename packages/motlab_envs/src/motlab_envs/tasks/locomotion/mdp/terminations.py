"""Termination MDP functions for locomotion tasks (torch).

All returns are bool tensors of shape ``(num_envs,)``. Multiple terms
are OR-combined by :class:`motlab_envs.managers.termination.TerminationManager`.
Time-limit truncation is handled by ``TorchEnv`` itself, not here.
"""

from __future__ import annotations

from typing import Any

import torch

from motlab_envs.math import quaternion as quat


def _quat_wxyz(env: Any) -> torch.Tensor:
    return quat.xyzw_to_wxyz(env.base_quat_xyzw())


def base_height_below_minimum(env: Any, minimum_height: float = 0.2) -> torch.Tensor:
    """True when the base has dropped below ``minimum_height`` (world z)."""
    z = env.base_pos()[..., 2]
    return z < minimum_height


def bad_orientation(env: Any, gravity_z_threshold: float = -0.5) -> torch.Tensor:
    """True when the base is tipped beyond a gravity-z threshold.

    ``gravity_z_threshold`` is in body-frame gravity coordinates: upright is
    ``-1.0``; a value of ``-0.5`` fires once the base is tipped past ~60°.
    """
    q_wxyz = _quat_wxyz(env)
    gravity = torch.zeros_like(q_wxyz[..., :3])
    gravity[..., 2] = -1.0
    proj = quat.rotate_inverse(q_wxyz, gravity)
    return proj[..., 2] > gravity_z_threshold
