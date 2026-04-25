"""Reward manager: weighted sum of named reward terms."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import RewardTermCfg, SceneEntityCfg


class RewardManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, RewardTermCfg] = {}
        for name, term in self._term_items():
            assert isinstance(term, RewardTermCfg)
            for k, v in list(term.params.items()):
                if isinstance(v, SceneEntityCfg):
                    v.resolve(env.scene)
                    term.params[k] = v
            self._terms[name] = term
        self._step_dt = float(getattr(env, "step_dt", 1.0))
        self._episode_sums: dict[str, torch.Tensor] = {
            n: torch.zeros(self.num_envs, device=self.device) for n in self._terms
        }

    @property
    def active_terms(self) -> list[str]:
        return list(self._terms.keys())

    def compute(self, dt: float | None = None) -> torch.Tensor:
        if dt is None:
            dt = self._step_dt
        total = torch.zeros(self.num_envs, device=self.device)
        for name, term in self._terms.items():
            r = term.func(self._env, **term.params) * term.weight * dt
            self._episode_sums[name] += r
            total += r
        return total

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        info: dict[str, float] = {}
        if env_ids is None:
            for name, sums in self._episode_sums.items():
                info[f"Episode_Reward/{name}"] = float(sums.mean().item())
                sums.zero_()
        else:
            for name, sums in self._episode_sums.items():
                if len(env_ids) > 0:
                    info[f"Episode_Reward/{name}"] = float(sums[env_ids].mean().item())
                sums[env_ids] = 0.0
        return info
