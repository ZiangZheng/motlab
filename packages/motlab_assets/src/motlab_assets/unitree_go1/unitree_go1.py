"""Unitree Go1 quadruped articulation cfg with two PD actuator groups."""

from __future__ import annotations

from pathlib import Path

from motlab.actuators.actuator_cfg import IdealPDActuatorCfg
from motlab.assets.articulation_cfg import ArticulationCfg, InitialStateCfg

_GO1_SCENE = str(Path(__file__).parent / "xmls" / "scene.xml")

# Standing default joint angles
_DEFAULT_JOINT_POS = {
    "F[LR]_hip_joint": 0.0,
    "R[LR]_hip_joint": 0.0,
    "F[LR]_thigh_joint": 0.9,
    "R[LR]_thigh_joint": 0.9,
    "F[LR]_calf_joint": -1.8,
    "R[LR]_calf_joint": -1.8,
}

GO1_CFG = ArticulationCfg(
    asset_path=_GO1_SCENE,
    init_state=InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_DEFAULT_JOINT_POS,
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.7,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)
