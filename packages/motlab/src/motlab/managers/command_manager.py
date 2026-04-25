"""Command manager: holds resampled command vectors (e.g. velocity commands)."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import CommandTermCfg


class CommandTerm:
    """Base command term. Subclasses sample commands and expose them as a
    tensor of shape ``(num_envs, command_dim)``."""

    cfg: CommandTermCfg

    def __init__(self, cfg: CommandTermCfg, env) -> None:
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs
        self._command = torch.zeros(self._num_envs, self.command_dim, device=self._device)
        self._time_left = torch.zeros(self._num_envs, device=self._device)

    @property
    def command_dim(self) -> int:
        raise NotImplementedError

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        self._time_left -= dt
        due = (self._time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if due.numel() > 0:
            self._resample(due)
            lo, hi = self.cfg.resampling_time_range
            self._time_left[due] = lo + (hi - lo) * torch.rand(len(due), device=self._device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._resample(env_ids)
        lo, hi = self.cfg.resampling_time_range
        self._time_left[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self._device)


class CommandManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, CommandTerm] = {}
        for name, term_cfg in self._term_items():
            assert isinstance(term_cfg, CommandTermCfg)
            term_class = term_cfg.class_type
            if term_class is None:
                raise ValueError(f"CommandTerm {name!r}: cfg.class_type is None")
            self._terms[name] = term_class(term_cfg, env)

    @property
    def terms(self) -> dict[str, CommandTerm]:
        return self._terms

    def get_command(self, name: str) -> torch.Tensor:
        return self._terms[name].command

    def compute(self, dt: float) -> None:
        for term in self._terms.values():
            term.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        for term in self._terms.values():
            term.reset(env_ids)
        return {}
