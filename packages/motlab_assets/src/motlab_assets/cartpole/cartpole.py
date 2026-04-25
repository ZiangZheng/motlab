"""Cartpole articulation cfg: slider + hinge with a single force actuator."""

from __future__ import annotations

from pathlib import Path

from motlab.actuators.actuator_cfg import IdealPDActuatorCfg
from motlab.assets.articulation_cfg import ArticulationCfg, InitialStateCfg

_CARTPOLE_XML = str(Path(__file__).parent / "xmls" / "cartpole.xml")

CARTPOLE_CFG = ArticulationCfg(
    asset_path=_CARTPOLE_XML,
    init_state=InitialStateCfg(joint_pos={"slider": 0.0, "hinge": 0.0}),
    actuators={
        "cart": IdealPDActuatorCfg(
            joint_names_expr=["slider"],
            effort_limit=3.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
