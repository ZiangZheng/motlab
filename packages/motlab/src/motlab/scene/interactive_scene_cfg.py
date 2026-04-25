"""Scene config: declares which assets are spawned + scene-level options.

Subclasses declare assets as class attributes typed as
:class:`ArticulationCfg`. The :class:`InteractiveScene` constructor
introspects the cfg to spawn each one.
"""

from __future__ import annotations

import dataclasses

from motlab.utils.configclass import configclass


@configclass
class InteractiveSceneCfg:
    """Base scene cfg.  Subclass and add ``robot: ArticulationCfg = ...``."""

    num_envs: int = 1
    env_spacing: float = 2.0
    replicate_physics: bool = True

    def asset_items(self) -> list[tuple[str, object]]:
        """Return ``(name, cfg)`` pairs for every dataclass field that is an
        :class:`ArticulationCfg`."""
        from motlab.assets.articulation_cfg import ArticulationCfg

        out: list[tuple[str, object]] = []
        for f in dataclasses.fields(self):
            val = getattr(self, f.name, None)
            if isinstance(val, ArticulationCfg):
                out.append((f.name, val))
        return out
