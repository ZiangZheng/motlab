"""Simulation timing + device configuration."""

from __future__ import annotations

from motlab.utils.configclass import configclass


@configclass
class SimulationCfg:
    """Engine-level config: timestep, device, gravity (informational)."""

    dt: float = 0.005
    device: str = "cpu"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
