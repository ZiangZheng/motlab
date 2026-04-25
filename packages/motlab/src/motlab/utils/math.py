"""Torch quaternion / frame-transform utilities (IsaacLab conventions).

Quaternion layout throughout motlab is ``(w, x, y, z)`` with ``w >= 0``
ambiguity resolution left to the caller. MotrixSim uses ``(x, y, z, w)`` —
convert at the engine boundary via :func:`convert_quat`.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Conventions + small helpers
# ---------------------------------------------------------------------------
def normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def convert_quat(quat: torch.Tensor, to: str = "wxyz") -> torch.Tensor:
    """Reorder a quaternion between ``wxyz`` and ``xyzw`` layouts."""
    if to == "wxyz":
        return torch.stack([quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]], dim=-1)
    if to == "xyzw":
        return torch.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], dim=-1)
    raise ValueError(f"Unknown quaternion layout '{to}' (use 'wxyz' or 'xyzw').")


def quat_unique(quat: torch.Tensor) -> torch.Tensor:
    """Flip sign so ``w >= 0`` (quat and -quat represent the same rotation)."""
    return torch.where(quat[..., 0:1] < 0, -quat, quat)


# ---------------------------------------------------------------------------
# Quaternion algebra (wxyz)
# ---------------------------------------------------------------------------
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (shape ``(..., 3)``) by unit quaternion ``q`` (wxyz)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of unit quaternion ``q``."""
    return quat_apply(quat_conjugate(q), v)


# Compatibility aliases used by isaaclab MDP code.
quat_rotate = quat_apply
quat_rotate_inverse = quat_apply_inverse


# ---------------------------------------------------------------------------
# Euler conversions
# ---------------------------------------------------------------------------
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def euler_xyz_from_quat(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angles), torch.cos(angles))


# ---------------------------------------------------------------------------
# Gravity helpers
# ---------------------------------------------------------------------------
def project_gravity_b(
    root_quat_w: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gravity vector expressed in the base frame given ``root_quat_w`` (wxyz)."""
    if gravity_w is None:
        gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], dtype=root_quat_w.dtype, device=root_quat_w.device
        ).expand_as(root_quat_w[..., :3])
    return quat_apply_inverse(root_quat_w, gravity_w)


# ---------------------------------------------------------------------------
# Random sampling helpers
# ---------------------------------------------------------------------------
def sample_uniform(
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    shape: Tuple[int, ...],
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    u = torch.rand(shape, device=device, generator=generator)
    return low + (high - low) * u


PI = math.pi
