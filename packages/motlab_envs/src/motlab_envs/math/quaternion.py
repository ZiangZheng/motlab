"""Batched quaternion helpers (torch). Quaternions stored as ``(w, x, y, z)``.

All functions accept tensors with a trailing dim of 4 for quaternions and 3
for vectors. Leading dims broadcast; output device/dtype follow the inputs.
"""

from __future__ import annotations

import torch


def normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -q[..., 1:]
    return out


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by quaternion ``q`` (wxyz)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse (conjugate, for unit ``q``) of ``q``."""
    return rotate(conjugate(q), v)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Reorder quaternion from xyzw (MotrixSim / MuJoCo) to wxyz (motlab)."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Reorder quaternion from wxyz (motlab) to xyzw (MotrixSim / MuJoCo)."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)
