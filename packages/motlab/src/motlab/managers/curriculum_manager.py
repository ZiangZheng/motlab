"""Curriculum manager: lets terms inspect / modify the env over time."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import CurriculumTermCfg, SceneEntityCfg


class CurriculumManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, CurriculumTermCfg] = {}
        for name, term in self._term_items():
            assert isinstance(term, CurriculumTermCfg)
            for k, v in list(term.params.items()):
                if isinstance(v, SceneEntityCfg):
                    v.resolve(env.scene)
                    term.params[k] = v
            self._terms[name] = term

    def compute(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        info: dict[str, float] = {}
        for name, term in self._terms.items():
            value = term.func(self._env, env_ids=env_ids, **term.params)
            if value is not None:
                info[f"Curriculum/{name}"] = float(value)
        return info

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        return {}
