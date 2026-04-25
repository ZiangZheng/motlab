"""Uniform velocity command term — base ``[lin_x, lin_y, ang_z]`` sampler."""

from __future__ import annotations

import torch

from motlab.managers.command_manager import CommandTerm
from motlab.managers.manager_term_cfg import CommandTermCfg
from motlab.utils.configclass import configclass


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Per-axis uniform ranges; sampled per resample event."""

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        heading: tuple[float, float] | None = None

    ranges: "UniformVelocityCommandCfg.Ranges" = Ranges()
    rel_standing_envs: float = 0.0  # fraction of envs receiving zero command
    heading_command: bool = False
    heading_control_stiffness: float = 0.5


class UniformVelocityCommand(CommandTerm):
    cfg: UniformVelocityCommandCfg

    @property
    def command_dim(self) -> int:
        return 3

    def _resample(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = self.cfg.ranges
        cmd = torch.empty(n, 3, device=self._device)
        cmd[:, 0].uniform_(*r.lin_vel_x)
        cmd[:, 1].uniform_(*r.lin_vel_y)
        cmd[:, 2].uniform_(*r.ang_vel_z)
        if self.cfg.rel_standing_envs > 0.0:
            stand_mask = torch.rand(n, device=self._device) < self.cfg.rel_standing_envs
            cmd[stand_mask] = 0.0
        self._command[env_ids] = cmd
