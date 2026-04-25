"""Actuator configuration dataclasses."""

from __future__ import annotations

from typing import Type

from motlab.utils.configclass import configclass


@configclass
class ActuatorBaseCfg:
    """Common actuator config fields. ``joint_names_expr`` is a list of
    regex patterns that select which joints this actuator controls."""

    class_type: Type | None = None
    joint_names_expr: list[str] = []
    effort_limit: float | dict[str, float] | None = None
    velocity_limit: float | dict[str, float] | None = None
    stiffness: float | dict[str, float] | None = None
    damping: float | dict[str, float] | None = None
    armature: float | dict[str, float] | None = None
    friction: float | dict[str, float] | None = None


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Ideal joint-space PD actuator: ``tau = kp*(q* - q) - kd*qd``,
    clipped to ``effort_limit``."""

    pass
