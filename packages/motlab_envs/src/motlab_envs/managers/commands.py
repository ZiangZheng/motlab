"""Command manager: sample + resample commands per env (e.g. velocity tracking)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class VelocityCommandManager:
    """Sample ``(vx, vy, wz)`` commands per env and resample on a schedule.

    Usage::

        cmds = VelocityCommandManager(num_envs=N, ctrl_dt=0.02,
                                      resample_seconds=10.0,
                                      low=(-1.0, -1.0, -1.0),
                                      high=(1.0, 1.0, 1.0),
                                      device="cpu")
        cmds.reset_all()
        cmds.reset(done_mask)         # bool tensor (N,)
        cmds.maybe_resample(step=t)   # each step; resamples expired envs
        cmd = cmds.commands            # (N, 3) float32 tensor
    """

    num_envs: int
    ctrl_dt: float
    resample_seconds: float
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    device: torch.device | str = "cpu"
    generator: Optional[torch.Generator] = None

    commands: torch.Tensor = field(init=False)
    _next_resample_step: torch.Tensor = field(init=False)
    _low: torch.Tensor = field(init=False)
    _high: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self._low = torch.as_tensor(self.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(self.high, dtype=torch.float32, device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._next_resample_step = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

    @property
    def resample_steps(self) -> int:
        return max(1, int(self.resample_seconds / self.ctrl_dt))

    def _sample(self, n: int) -> torch.Tensor:
        u = torch.rand((n, 3), device=self.device, dtype=torch.float32, generator=self.generator)
        return self._low + u * (self._high - self._low)

    def reset_all(self) -> None:
        self.commands[:] = self._sample(self.num_envs)
        self._next_resample_step[:] = self.resample_steps

    def reset(self, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        self.commands[mask] = self._sample(n)
        self._next_resample_step[mask] = self.resample_steps

    def maybe_resample(self, step: int) -> None:
        due = step >= self._next_resample_step
        n = int(due.sum().item())
        if n == 0:
            return
        self.commands[due] = self._sample(n)
        self._next_resample_step[due] = step + self.resample_steps

    def set(self, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            self.commands[:] = values
        else:
            self.commands[mask] = values
