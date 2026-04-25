"""Action manager + base ``ActionTerm`` (e.g. JointPositionAction)."""

from __future__ import annotations

import torch

from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import ActionTermCfg


class ActionTerm:
    """Base class for an action term: maps raw policy output → asset commands."""

    cfg: ActionTermCfg

    def __init__(self, cfg: ActionTermCfg, env) -> None:
        self.cfg = cfg
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = env.num_envs

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def raw_actions(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def processed_actions(self) -> torch.Tensor:
        raise NotImplementedError

    def process_actions(self, actions: torch.Tensor) -> None:
        raise NotImplementedError

    def apply_actions(self) -> None:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class ActionManager(ManagerBase):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._terms: dict[str, ActionTerm] = {}
        for name, term_cfg in self._term_items():
            assert isinstance(term_cfg, ActionTermCfg), f"{name!r}: not an ActionTermCfg"
            term_class = term_cfg.class_type
            if term_class is None:
                raise ValueError(f"ActionTerm {name!r}: cfg.class_type is None")
            self._terms[name] = term_class(term_cfg, env)
        self._action_dim = sum(t.action_dim for t in self._terms.values())
        self._action = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    @property
    def total_action_dim(self) -> int:
        return self._action_dim

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def terms(self) -> dict[str, ActionTerm]:
        return self._terms

    def process_action(self, actions: torch.Tensor) -> None:
        self._action = actions
        offset = 0
        for term in self._terms.values():
            d = term.action_dim
            term.process_actions(actions[:, offset : offset + d])
            offset += d

    def apply_action(self) -> None:
        for term in self._terms.values():
            term.apply_actions()

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        for term in self._terms.values():
            term.reset(env_ids)
        if env_ids is None:
            self._action.zero_()
        else:
            self._action[env_ids] = 0.0
        return {}
