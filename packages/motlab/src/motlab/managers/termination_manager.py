"""Termination manager: collects MDP-end / time-out signals."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import SceneEntityCfg, TerminationTermCfg


class TerminationManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, TerminationTermCfg] = {}
        for name, term in self._term_items():
            assert isinstance(term, TerminationTermCfg)
            for k, v in list(term.params.items()):
                if isinstance(v, SceneEntityCfg):
                    v.resolve(env.scene)
                    term.params[k] = v
            self._terms[name] = term

        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros_like(self._terminated)
        self._term_dones: dict[str, torch.Tensor] = {
            n: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for n in self._terms
        }

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated

    @property
    def truncated(self) -> torch.Tensor:
        return self._truncated

    @property
    def dones(self) -> torch.Tensor:
        return self._terminated | self._truncated

    def compute(self) -> torch.Tensor:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, term in self._terms.items():
            d = term.func(self._env, **term.params).bool()
            self._term_dones[name] = d
            if term.time_out:
                truncated |= d
            else:
                terminated |= d
        self._terminated = terminated
        self._truncated = truncated
        return self.dones

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        if env_ids is None:
            self._terminated.zero_()
            self._truncated.zero_()
        else:
            self._terminated[env_ids] = False
            self._truncated[env_ids] = False
        return {}
