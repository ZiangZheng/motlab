"""Command generators for locomotion tasks (torch).

Each command function takes ``(env, **params)`` and returns a
``(num_envs, D)`` float32 tensor on ``env.device``. The
:class:`motlab_envs.manager_env.ManagerBasedTorchEnv` calls these on
reset and on each resample tick.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


def uniform_velocity(
    env: Any,
    low: Sequence[float] = (-1.0, -1.0, -1.0),
    high: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Sample ``(vx, vy, wz)`` uniformly from ``[low, high]`` per env."""
    low_t = torch.as_tensor(low, dtype=torch.float32, device=env.device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=env.device)
    if low_t.shape != (3,) or high_t.shape != (3,):
        raise ValueError(
            f"uniform_velocity expects length-3 bounds, got {tuple(low_t.shape)} / {tuple(high_t.shape)}"
        )
    u = torch.rand((env.num_envs, 3), dtype=torch.float32, device=env.device)
    return low_t + u * (high_t - low_t)
