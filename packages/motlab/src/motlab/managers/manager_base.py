"""Common helpers for managers — term iteration via dataclass introspection."""

from __future__ import annotations

import dataclasses
from typing import Iterator


class ManagerBase:
    """Provides ``_term_items()`` over a dataclass cfg, returning
    ``(name, term_cfg)`` pairs in declaration order."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self):
        return self._device

    def _term_items(self) -> Iterator[tuple[str, object]]:
        if self.cfg is None:
            return
        for f in dataclasses.fields(self.cfg):
            val = getattr(self.cfg, f.name, None)
            if val is None:
                continue
            yield f.name, val
