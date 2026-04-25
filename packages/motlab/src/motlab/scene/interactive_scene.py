"""InteractiveScene: holds the spawned assets and exposes them by name.

In motlab there is exactly one :class:`Articulation` per scene cfg field,
each backed by its own :class:`MotrixEngine`. (MotrixSim batches at the
data level rather than the model level — multiple robots in one scene
would require a single combined MJCF, which we don't support yet.)
"""

from __future__ import annotations

from typing import Iterator

import torch

from motlab.assets.articulation import Articulation
from motlab.assets.articulation_cfg import ArticulationCfg
from motlab.scene.interactive_scene_cfg import InteractiveSceneCfg


class InteractiveScene:
    def __init__(
        self,
        cfg: InteractiveSceneCfg,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = torch.device(device)
        self._articulations: dict[str, Articulation] = {}

        for name, asset_cfg in cfg.asset_items():
            assert isinstance(asset_cfg, ArticulationCfg)
            asset_class = asset_cfg.class_type or Articulation
            self._articulations[name] = asset_class(
                cfg=asset_cfg, num_envs=self.num_envs, device=self.device
            )

        if not self._articulations:
            raise ValueError("InteractiveScene requires at least one ArticulationCfg field")

    # ------------------------------------------------------------------
    @property
    def articulations(self) -> dict[str, Articulation]:
        return self._articulations

    def __getitem__(self, key: str) -> Articulation:
        return self._articulations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._articulations)

    def keys(self) -> list[str]:
        return list(self._articulations.keys())

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for a in self._articulations.values():
            a.reset(env_ids)

    def write_data_to_sim(self) -> None:
        for a in self._articulations.values():
            a.write_data_to_sim()

    def step(self) -> None:
        for a in self._articulations.values():
            a.engine.step()

    def update(self, dt: float) -> None:
        for a in self._articulations.values():
            a.update(dt)
