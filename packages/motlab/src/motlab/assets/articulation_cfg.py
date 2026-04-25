"""Articulation configuration: spawn pose, defaults, actuator groups."""

from __future__ import annotations

from typing import Type

from motlab.actuators.actuator_cfg import ActuatorBaseCfg
from motlab.utils.configclass import configclass


@configclass
class InitialStateCfg:
    """Default state used when ``Articulation.reset`` is called without overrides."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # base position
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos: dict[str, float] = {}  # regex name → value (missing names → 0)
    joint_vel: dict[str, float] = {}


@configclass
class ArticulationCfg:
    """Spawn config for a multi-joint robot."""

    class_type: Type | None = None
    asset_path: str = ""  # MJCF or scene XML on disk
    init_state: InitialStateCfg = InitialStateCfg()
    soft_joint_pos_limit_factor: float = 1.0
    actuators: dict[str, ActuatorBaseCfg] = {}
