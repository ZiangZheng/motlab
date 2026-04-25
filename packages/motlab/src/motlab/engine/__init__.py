"""MotrixSim adapter — single import boundary for the engine."""

from motlab.engine.motrix import (
    MotrixEngine,
    load_model,
    resolve_actuator_indices,
    resolve_joint_indices,
    resolve_link_indices,
)

__all__ = [
    "MotrixEngine",
    "load_model",
    "resolve_actuator_indices",
    "resolve_joint_indices",
    "resolve_link_indices",
]
