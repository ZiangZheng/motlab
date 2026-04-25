"""Joint-level action terms: position target and direct effort."""

from __future__ import annotations

import re

import torch

from motlab.managers.action_manager import ActionTerm
from motlab.managers.manager_term_cfg import ActionTermCfg
from motlab.utils.configclass import configclass
from motlab.utils.string_utils import resolve_matching_names


def _broadcast(spec, names, fallback) -> list[float]:
    if isinstance(spec, (int, float)):
        return [float(spec)] * len(names)
    if isinstance(spec, dict):
        out = [float(fallback)] * len(names)
        seen = [False] * len(names)
        for pat, val in spec.items():
            cre = re.compile(f"^{pat}$")
            for i, n in enumerate(names):
                if not seen[i] and cre.match(n):
                    out[i] = float(val)
                    seen[i] = True
        return out
    return [float(fallback)] * len(names)


@configclass
class JointPositionActionCfg(ActionTermCfg):
    pass


class JointPositionAction(ActionTerm):
    """Action = scaled + offset position target written to the asset's
    joint position target buffer.  When ``use_default_offset=True`` the
    offset is the asset's default joint position."""

    cfg: JointPositionActionCfg

    def __init__(self, cfg: JointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        ids, names = resolve_matching_names(
            cfg.joint_names, self._asset.joint_names, preserve_order=cfg.preserve_order
        )
        if not ids:
            raise ValueError(f"JointPositionAction matched no joints: {cfg.joint_names}")
        self._joint_ids = torch.tensor(ids, device=self._device, dtype=torch.long)
        self._joint_names = names
        scale = torch.tensor(_broadcast(cfg.scale, names, 1.0), device=self._device)
        self._scale = scale.unsqueeze(0).expand(self._num_envs, -1).clone()
        offset = torch.tensor(_broadcast(cfg.offset, names, 0.0), device=self._device)
        self._offset = offset.unsqueeze(0).expand(self._num_envs, -1).clone()
        if cfg.use_default_offset:
            default = self._asset.data.default_joint_pos.index_select(-1, self._joint_ids)
            self._offset = default.clone()
        self._raw = torch.zeros(self._num_envs, len(ids), device=self._device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self) -> int:
        return int(self._joint_ids.numel())

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw = actions
        self._processed = actions * self._scale + self._offset

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed, joint_ids=self._joint_ids)


@configclass
class JointEffortActionCfg(ActionTermCfg):
    pass


class JointEffortAction(ActionTerm):
    """Direct joint effort (ignores actuator PD). Used for cartpole-style envs."""

    cfg: JointEffortActionCfg

    def __init__(self, cfg: JointEffortActionCfg, env) -> None:
        super().__init__(cfg, env)
        ids, names = resolve_matching_names(
            cfg.joint_names, self._asset.joint_names, preserve_order=cfg.preserve_order
        )
        if not ids:
            raise ValueError(f"JointEffortAction matched no joints: {cfg.joint_names}")
        self._joint_ids = torch.tensor(ids, device=self._device, dtype=torch.long)
        scale = torch.tensor(_broadcast(cfg.scale, names, 1.0), device=self._device)
        self._scale = scale.unsqueeze(0).expand(self._num_envs, -1).clone()
        self._raw = torch.zeros(self._num_envs, len(ids), device=self._device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self) -> int:
        return int(self._joint_ids.numel())

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw = actions
        self._processed = actions * self._scale

    def apply_actions(self) -> None:
        # Write directly to MotrixSim ctrl buffer for the matching actuator(s).
        # We assume the matching actuator name == joint name (true for cartpole).
        ctrl = torch.zeros(self._num_envs, self._asset.engine.num_actuators, device=self._device)
        # Build joint→actuator inverse map via name match.
        for k, jname in enumerate(self._asset.joint_names):
            if jname in self._asset.model.actuator_names:
                aid = self._asset.model.actuator_names.index(jname)
                jid_in_processed = (self._joint_ids == k).nonzero(as_tuple=False).flatten()
                if jid_in_processed.numel() > 0:
                    ctrl[:, aid] = self._processed[:, jid_in_processed.item()]
        self._asset.engine.set_actuator_ctrls(ctrl)
