"""Observation manager: groups of named observation terms → flat tensor.

The observation cfg is a dataclass whose fields are
:class:`ObservationGroupCfg` subclasses. Each group is itself a dataclass
whose fields are :class:`ObservationTermCfg` instances. Within a group,
each term's function is called with ``func(env, **params)`` and must
return a 2D tensor of shape ``(num_envs, term_dim)``.
"""

from __future__ import annotations

import dataclasses

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import (
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)


def _resolve_entity_params(params: dict, scene) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, SceneEntityCfg):
            v.resolve(scene)
        out[k] = v
    return out


class ObservationManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._groups: dict[str, ObservationGroupCfg] = {}
        self._group_terms: dict[str, dict[str, ObservationTermCfg]] = {}
        self._group_dims: dict[str, int] = {}

        if cfg is None:
            return
        for f in dataclasses.fields(cfg):
            grp = getattr(cfg, f.name, None)
            if not isinstance(grp, ObservationGroupCfg):
                continue
            self._groups[f.name] = grp
            terms: dict[str, ObservationTermCfg] = {}
            for tf in dataclasses.fields(grp):
                tval = getattr(grp, tf.name, None)
                if isinstance(tval, ObservationTermCfg):
                    tval.params = _resolve_entity_params(tval.params, env.scene)
                    terms[tf.name] = tval
            self._group_terms[f.name] = terms
            # Compute term dim by calling once with the current env state.
            dim = 0
            for tname, tcfg in terms.items():
                t = self._compute_term(tcfg)
                dim += t.shape[-1]
            self._group_dims[f.name] = dim

    # ------------------------------------------------------------------
    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...]]:
        return {k: (v,) for k, v in self._group_dims.items()}

    def _compute_term(self, tcfg: ObservationTermCfg) -> torch.Tensor:
        out = tcfg.func(self._env, **tcfg.params)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if tcfg.scale is not None:
            out = out * tcfg.scale
        if tcfg.clip is not None:
            out = out.clamp(tcfg.clip[0], tcfg.clip[1])
        if tcfg.noise is not None:
            out = tcfg.noise(out)
        return out

    def compute(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for gname, terms in self._group_terms.items():
            parts = [self._compute_term(t) for t in terms.values()]
            grp = self._groups[gname]
            if grp.concatenate_terms:
                result[gname] = torch.cat(parts, dim=-1) if parts else torch.empty(self.num_envs, 0, device=self.device)
            else:
                result[gname] = parts[0] if parts else torch.empty(self.num_envs, 0, device=self.device)
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        return {}
