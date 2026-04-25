"""Config dataclass for :class:`ManagerBasedEnv`."""

from __future__ import annotations

from typing import Any

from motlab.scene.interactive_scene_cfg import InteractiveSceneCfg
from motlab.sim.simulation_cfg import SimulationCfg
from motlab.utils.configclass import configclass


@configclass
class ManagerBasedEnvCfg:
    """Common (non-RL) env config: scene, sim, actions/observations/events."""

    sim: SimulationCfg = SimulationCfg()
    scene: InteractiveSceneCfg | None = None
    decimation: int = 1  # number of physics steps per env.step()

    actions: Any = None
    observations: Any = None
    events: Any = None

    seed: int | None = None
