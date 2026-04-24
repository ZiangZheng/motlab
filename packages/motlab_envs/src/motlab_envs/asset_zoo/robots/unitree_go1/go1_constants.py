"""Unitree Go1 constants — robot-level definitions independent of any task.

Ported from mjlab's ``go1_constants.py`` but reshaped to fit motlab's style:
physical parameters live here as plain Python tuples / floats; task configs
import them to populate ``ControlCfg`` / ``InitStateCfg``. Consumers
convert to ``torch.Tensor`` on the task's device at build time.
"""

from __future__ import annotations

from pathlib import Path

GO1_DIR: Path = Path(__file__).parent
GO1_XML: Path = GO1_DIR / "xmls" / "go1.xml"
GO1_SCENE_XML: Path = GO1_DIR / "xmls" / "scene.xml"

# Joint order follows the MJCF body traversal: FR, FL, RR, RL × (hip, thigh, calf).
JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

# Default standing pose (mjlab INIT_STATE regex resolved explicitly).
DEFAULT_JOINT_ANGLES: tuple[float, ...] = (
     0.1,  0.9, -1.8,   # FR
    -0.1,  0.9, -1.8,   # FL
     0.1,  0.9, -1.8,   # RR
    -0.1,  0.9, -1.8,   # RL
)

# PD gains reflected to motor side (see mjlab's reflected_inertia * NATURAL_FREQ²).
# ElectricActuator(Ixx=1.118e-4, hip gear=6, knee gear=9), natural_freq = 2π·10 Hz,
# damping_ratio = 2.0. Same scalar pair per joint here; sim2real can shape later.
_HIP_STIFFNESS:  float = 20.0
_HIP_DAMPING:    float = 0.5
_KNEE_STIFFNESS: float = 20.0
_KNEE_DAMPING:   float = 0.5


def _make_pd(hip: float, knee: float) -> tuple[float, ...]:
    """(hip, thigh, calf) × 4 — knee overrides the calf joints (every 3rd index)."""
    out = [hip] * 12
    for i in (2, 5, 8, 11):
        out[i] = knee
    return tuple(out)


STIFFNESS: tuple[float, ...] = _make_pd(_HIP_STIFFNESS, _KNEE_STIFFNESS)
DAMPING:   tuple[float, ...] = _make_pd(_HIP_DAMPING, _KNEE_DAMPING)

# Per mjlab: action_scale = 0.25 * effort_limit / stiffness.
ACTION_SCALE: float = 0.25

# Torque limits per actuator (N·m).
TORQUE_LIMIT: float = 35.55  # worst-case (calves)

# Initial base pose for reset.
INIT_BASE_POS: tuple[float, float, float] = (0.0, 0.0, 0.34)
INIT_BASE_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# Convenience: reset reference pose as a 19-vec ((xyz) + (xyzw) + 12 joints).
INIT_DOF_POS: tuple[float, ...] = (*INIT_BASE_POS, *INIT_BASE_QUAT_XYZW, *DEFAULT_JOINT_ANGLES)
