"""Event manager: hooks fired on startup, reset, or interval."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import EventTermCfg, SceneEntityCfg


class EventManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, EventTermCfg] = {}
        for name, term in self._term_items():
            assert isinstance(term, EventTermCfg)
            for k, v in list(term.params.items()):
                if isinstance(v, SceneEntityCfg):
                    v.resolve(env.scene)
                    term.params[k] = v
            self._terms[name] = term
        # interval bookkeeping
        self._interval_time = {
            n: torch.zeros(self.num_envs, device=self.device)
            for n, t in self._terms.items() if t.mode == "interval"
        }

    def apply(self, mode: str, env_ids: torch.Tensor | None = None, dt: float = 0.0) -> None:
        for name, term in self._terms.items():
            if term.mode != mode:
                continue
            if mode == "interval":
                if term.interval_range_s is None:
                    continue
                self._interval_time[name] += dt
                lo, hi = term.interval_range_s
                fire = self._interval_time[name] >= hi
                if fire.any():
                    fids = torch.nonzero(fire, as_tuple=False).flatten()
                    term.func(self._env, env_ids=fids, **term.params)
                    # resample interval
                    self._interval_time[name][fids] = 0.0
            else:
                term.func(self._env, env_ids=env_ids, **term.params)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        for buf in self._interval_time.values():
            if env_ids is None:
                buf.zero_()
            else:
                buf[env_ids] = 0.0
        return {}
