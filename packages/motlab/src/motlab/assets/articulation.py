"""Articulation: torch-facing wrapper over a MotrixSim model + actuator groups.

The asset takes ownership of one :class:`MotrixEngine` and exposes:

- joint-ordered state via :class:`ArticulationData`
- a :meth:`set_joint_position_target` write path
- a :meth:`write_data_to_sim` step hook that runs each actuator group's
  PD controller and writes torques back to the engine
- :meth:`reset` / :meth:`write_*_to_sim` reset hooks

The joint order seen from outside this module is :attr:`joint_names`
(== ``model.joint_names``). Actuator order in MotrixSim differs — we
remap inside :meth:`write_data_to_sim`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import torch

from motlab.actuators.actuator_base import ActuatorBase
from motlab.actuators.actuator_cfg import IdealPDActuatorCfg
from motlab.actuators.actuator_pd import IdealPDActuator
from motlab.assets.articulation_cfg import ArticulationCfg
from motlab.assets.articulation_data import ArticulationData
from motlab.engine.motrix import MotrixEngine, load_model
from motlab.utils.math import convert_quat
from motlab.utils.string_utils import resolve_matching_names, resolve_matching_names_values


def _broadcast_to_joint_dict(
    spec: float | dict[str, float] | None,
    names: Sequence[str],
    fallback: float,
) -> list[float]:
    if spec is None:
        return [float(fallback)] * len(names)
    if isinstance(spec, (int, float)):
        return [float(spec)] * len(names)
    out = [float(fallback)] * len(names)
    seen = [False] * len(names)
    for pat, val in spec.items():
        cre = re.compile(f"^{pat}$")
        for i, n in enumerate(names):
            if not seen[i] and cre.match(n):
                out[i] = float(val)
                seen[i] = True
    return out


class Articulation:
    """Wraps one MotrixSim model with default state, actuator groups, and
    torch-tensor accessors mirroring IsaacLab's :class:`Articulation`."""

    def __init__(
        self,
        cfg: ArticulationCfg,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        if not cfg.asset_path:
            raise ValueError("ArticulationCfg.asset_path must be set")
        path = Path(cfg.asset_path)
        if not path.exists():
            raise FileNotFoundError(f"Articulation asset not found: {path}")

        model = load_model(path)
        self.engine = MotrixEngine(model, num_envs=self.num_envs, device=self.device)
        self.model = model

        self.joint_names: list[str] = list(model.joint_names)
        self.body_names: list[str] = list(model.body_names)
        self.link_names: list[str] = list(model.link_names)
        self.num_joints: int = len(self.joint_names)
        self.has_floating_base: bool = self.engine.has_floating_base

        # ---- Default joint state -----------------------------------------
        default_pos = torch.zeros(self.num_joints, device=self.device)
        if cfg.init_state.joint_pos:
            ids, _, vals = resolve_matching_names_values(cfg.init_state.joint_pos, self.joint_names)
            default_pos[ids] = torch.tensor(vals, device=self.device)
        default_vel = torch.zeros(self.num_joints, device=self.device)
        if cfg.init_state.joint_vel:
            ids, _, vals = resolve_matching_names_values(cfg.init_state.joint_vel, self.joint_names)
            default_vel[ids] = torch.tensor(vals, device=self.device)
        self._default_joint_pos = default_pos.unsqueeze(0)  # (1, J)
        self._default_joint_vel = default_vel.unsqueeze(0)

        # Soft joint pos limits — pulled from the model's joint_limits table.
        jl = torch.tensor(model.joint_limits, device=self.device, dtype=torch.float32)
        # joint_limits is shape (num_dof_pos, 2) = (low, high) per dof
        # For free-base robots the first 7 entries are floating base; we want
        # the joint-only entries which start at engine._fb_pos_start + 7.
        if self.has_floating_base:
            jl_joints = jl[7:]
        else:
            jl_joints = jl
        # Each hinge/slide joint has num_dof_pos==1, so this lines up by joint.
        self._soft_joint_pos_limits = jl_joints.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        if cfg.soft_joint_pos_limit_factor != 1.0:
            mid = 0.5 * (self._soft_joint_pos_limits[..., 0] + self._soft_joint_pos_limits[..., 1])
            half = 0.5 * (self._soft_joint_pos_limits[..., 1] - self._soft_joint_pos_limits[..., 0])
            half = half * cfg.soft_joint_pos_limit_factor
            self._soft_joint_pos_limits[..., 0] = mid - half
            self._soft_joint_pos_limits[..., 1] = mid + half

        # ---- Actuator groups --------------------------------------------
        self.actuators: dict[str, ActuatorBase] = {}
        self._joint_actuator_idx = torch.full((self.num_joints,), -1, device=self.device, dtype=torch.long)
        for group_name, act_cfg in cfg.actuators.items():
            ids, names = resolve_matching_names(act_cfg.joint_names_expr, self.joint_names)
            if not ids:
                raise ValueError(
                    f"Actuator group {group_name!r} matches no joints "
                    f"(patterns={act_cfg.joint_names_expr})"
                )
            stiffness = torch.tensor(_broadcast_to_joint_dict(act_cfg.stiffness, names, 0.0), device=self.device)
            damping = torch.tensor(_broadcast_to_joint_dict(act_cfg.damping, names, 0.0), device=self.device)
            effort = torch.tensor(_broadcast_to_joint_dict(act_cfg.effort_limit, names, 1e6), device=self.device)
            vlim = torch.tensor(_broadcast_to_joint_dict(act_cfg.velocity_limit, names, 1e6), device=self.device)

            act_class = act_cfg.class_type or IdealPDActuator
            self.actuators[group_name] = act_class(
                cfg=act_cfg,
                joint_names=names,
                joint_ids=ids,
                num_envs=self.num_envs,
                device=self.device,
                stiffness=stiffness.unsqueeze(0).expand(self.num_envs, -1).clone(),
                damping=damping.unsqueeze(0).expand(self.num_envs, -1).clone(),
                effort_limit=effort.unsqueeze(0).expand(self.num_envs, -1).clone(),
                velocity_limit=vlim.unsqueeze(0).expand(self.num_envs, -1).clone(),
            )
            for j in ids:
                if self._joint_actuator_idx[j] != -1:
                    raise ValueError(f"Joint {self.joint_names[j]!r} matched by multiple actuator groups")
                self._joint_actuator_idx[j] = list(self.actuators).index(group_name)

        # If user provided no actuator config, fall back to a default zero-PD
        # group (useful for cartpole-style direct-torque tasks).
        if not self.actuators:
            default_cfg = IdealPDActuatorCfg(joint_names_expr=[".*"])
            self.actuators["__default__"] = IdealPDActuator(
                cfg=default_cfg,
                joint_names=self.joint_names,
                joint_ids=list(range(self.num_joints)),
                num_envs=self.num_envs,
                device=self.device,
                stiffness=torch.zeros(self.num_envs, self.num_joints, device=self.device),
                damping=torch.zeros(self.num_envs, self.num_joints, device=self.device),
                effort_limit=torch.full((self.num_envs, self.num_joints), 1e6, device=self.device),
                velocity_limit=torch.full((self.num_envs, self.num_joints), 1e6, device=self.device),
            )
            self._joint_actuator_idx[:] = 0

        # ---- Internal state buffers --------------------------------------
        self._joint_pos_target = self._default_joint_pos.expand(self.num_envs, -1).clone()
        self._applied_torque = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self._computed_torque = torch.zeros_like(self._applied_torque)
        self._joint_vel_prev = torch.zeros_like(self._applied_torque)
        self._joint_acc = torch.zeros_like(self._applied_torque)

        # Engine-side joint slice tables (positions and velocities).
        self._joint_pos_indices = torch.tensor(
            self.engine.joint_pos_idx, device=self.device, dtype=torch.long
        )
        self._joint_vel_indices = torch.tensor(
            self.engine.joint_vel_idx, device=self.device, dtype=torch.long
        )

        # Engine-side actuator → joint reorder index for ctrl writes.
        self._actuator_joint_ids = torch.tensor(
            self.engine.actuator_joint_idx, device=self.device, dtype=torch.long
        )

        # Lazy data view
        self.data = ArticulationData(self)

        # Initialise the simulation state from defaults so observations are valid before any reset.
        self._initialize_state()

    # ------------------------------------------------------------------
    # Internal joint-space readers (pull from engine each call).
    # ------------------------------------------------------------------
    def _full_dof_pos(self) -> torch.Tensor:
        return self.engine.dof_pos()

    def _full_dof_vel(self) -> torch.Tensor:
        return self.engine.dof_vel()

    def _joint_pos(self) -> torch.Tensor:
        return self._full_dof_pos().index_select(-1, self._joint_pos_indices)

    def _joint_vel(self) -> torch.Tensor:
        return self._full_dof_vel().index_select(-1, self._joint_vel_indices)

    # ---- Floating-base accessors -------------------------------------
    def _root_pos_w(self) -> torch.Tensor:
        if self.has_floating_base:
            return self._full_dof_pos()[..., 0:3]
        # No floating base — return zeros (matches IsaacLab fixed-base behaviour).
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _root_quat_w(self) -> torch.Tensor:
        """Returns root quaternion in ``wxyz`` (motlab convention)."""
        if self.has_floating_base:
            xyzw = self._full_dof_pos()[..., 3:7]
            return convert_quat(xyzw, to="wxyz")
        wxyz = torch.zeros(self.num_envs, 4, device=self.device)
        wxyz[..., 0] = 1.0
        return wxyz

    def _root_lin_vel_w(self) -> torch.Tensor:
        if self.has_floating_base:
            return self._full_dof_vel()[..., 0:3]
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _root_ang_vel_w(self) -> torch.Tensor:
        if self.has_floating_base:
            return self._full_dof_vel()[..., 3:6]
        return torch.zeros(self.num_envs, 3, device=self.device)

    # ---- Per-link --------------------------------------------------
    def _link_pos_w(self) -> torch.Tensor:
        return self.engine.link_poses()[..., 0:3]

    def _link_quat_w(self) -> torch.Tensor:
        xyzw = self.engine.link_poses()[..., 3:7]
        return convert_quat(xyzw, to="wxyz")

    # ------------------------------------------------------------------
    # Action writes
    # ------------------------------------------------------------------
    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Stage a joint position target. Applied to sim by :meth:`write_data_to_sim`."""
        if joint_ids is None:
            self._joint_pos_target = target
        else:
            ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
            self._joint_pos_target.index_copy_(-1, ids, target)

    def write_data_to_sim(self) -> None:
        """Run actuator PD, write torques to engine.actuator_ctrls."""
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        torque = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        for group in self.actuators.values():
            ids = torch.tensor(group.joint_ids, device=self.device, dtype=torch.long)
            tau = group.compute(
                control_action=self._joint_pos_target.index_select(-1, ids),
                joint_pos=joint_pos.index_select(-1, ids),
                joint_vel=joint_vel.index_select(-1, ids),
            )
            torque.index_copy_(-1, ids, tau)
        self._applied_torque = torque
        self._computed_torque = torch.stack(
            [g.computed_effort for g in self.actuators.values()], dim=0
        ).sum(dim=0) if False else self._applied_torque

        # MotrixSim actuator order ≠ joint order — remap by name.
        ctrl = torque.index_select(-1, self._actuator_joint_ids)
        self.engine.set_actuator_ctrls(ctrl)

    def update(self, dt: float) -> None:
        """Bookkeeping after an engine step (joint accelerations)."""
        cur_vel = self._joint_vel()
        if dt > 0:
            self._joint_acc = (cur_vel - self._joint_vel_prev) / dt
        self._joint_vel_prev = cur_vel.clone()

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------
    def _initialize_state(self) -> None:
        """Write default state into the engine's batched data."""
        dof_pos = torch.zeros(self.num_envs, self.engine.num_dof_pos, device=self.device)
        dof_vel = torch.zeros(self.num_envs, self.engine.num_dof_vel, device=self.device)

        if self.has_floating_base:
            pos = torch.tensor(self.cfg.init_state.pos, device=self.device, dtype=torch.float32)
            quat_wxyz = torch.tensor(self.cfg.init_state.rot, device=self.device, dtype=torch.float32)
            quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
            dof_pos[:, 0:3] = pos
            dof_pos[:, 3:7] = quat_xyzw
            lvel = torch.tensor(self.cfg.init_state.lin_vel, device=self.device, dtype=torch.float32)
            avel = torch.tensor(self.cfg.init_state.ang_vel, device=self.device, dtype=torch.float32)
            dof_vel[:, 0:3] = lvel
            dof_vel[:, 3:6] = avel

        # Joint defaults
        dof_pos.index_copy_(
            -1, self._joint_pos_indices, self._default_joint_pos.expand(self.num_envs, -1)
        )
        dof_vel.index_copy_(
            -1, self._joint_vel_indices, self._default_joint_vel.expand(self.num_envs, -1)
        )

        self.engine.set_dof_pos(dof_pos)
        self.engine.set_dof_vel(dof_vel)

        self._joint_pos_target = self._default_joint_pos.expand(self.num_envs, -1).clone()
        self._applied_torque.zero_()
        self._joint_acc.zero_()
        self._joint_vel_prev = self._joint_vel().clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all envs (the only mode currently supported)."""
        if env_ids is None or len(env_ids) == self.num_envs:
            self._initialize_state()
            for g in self.actuators.values():
                g.reset(None)
            return
        # Per-env reset: simulate by reading current full state, overwriting
        # selected envs, and re-writing.
        dof_pos = torch.from_numpy(self.engine.dof_pos_np().copy()).to(self.device)
        dof_vel = torch.from_numpy(self.engine.dof_vel_np().copy()).to(self.device)
        if self.has_floating_base:
            pos = torch.tensor(self.cfg.init_state.pos, device=self.device, dtype=torch.float32)
            quat_wxyz = torch.tensor(self.cfg.init_state.rot, device=self.device, dtype=torch.float32)
            quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
            dof_pos[env_ids, 0:3] = pos
            dof_pos[env_ids, 3:7] = quat_xyzw
            lvel = torch.tensor(self.cfg.init_state.lin_vel, device=self.device, dtype=torch.float32)
            avel = torch.tensor(self.cfg.init_state.ang_vel, device=self.device, dtype=torch.float32)
            dof_vel[env_ids, 0:3] = lvel
            dof_vel[env_ids, 3:6] = avel

        defaults_pos = self._default_joint_pos.expand(len(env_ids), -1)
        defaults_vel = self._default_joint_vel.expand(len(env_ids), -1)
        for k, j in enumerate(self._joint_pos_indices.tolist()):
            dof_pos[env_ids, j] = defaults_pos[:, k]
        for k, j in enumerate(self._joint_vel_indices.tolist()):
            dof_vel[env_ids, j] = defaults_vel[:, k]

        self.engine.set_dof_pos(dof_pos)
        self.engine.set_dof_vel(dof_vel)
        self._joint_pos_target[env_ids] = self._default_joint_pos.expand(len(env_ids), -1)
        self._applied_torque[env_ids] = 0.0
        self._joint_acc[env_ids] = 0.0
        for g in self.actuators.values():
            g.reset(env_ids)
        self._joint_vel_prev = self._joint_vel().clone()

    # ------------------------------------------------------------------
    # Direct state writes (used by event terms)
    # ------------------------------------------------------------------
    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """``root_pose``: (N, 7) ``[x,y,z, qw,qx,qy,qz]``."""
        if not self.has_floating_base:
            return
        dof_pos = torch.from_numpy(self.engine.dof_pos_np().copy()).to(self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        dof_pos[env_ids, 0:3] = root_pose[:, 0:3]
        dof_pos[env_ids, 3:7] = convert_quat(root_pose[:, 3:7], to="xyzw")
        self.engine.set_dof_pos(dof_pos)

    def write_root_velocity_to_sim(self, root_vel: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """``root_vel``: (N, 6) ``[vx,vy,vz, wx,wy,wz]``."""
        if not self.has_floating_base:
            return
        dof_vel = torch.from_numpy(self.engine.dof_vel_np().copy()).to(self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        dof_vel[env_ids, 0:6] = root_vel
        self.engine.set_dof_vel(dof_vel)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        dof_pos = torch.from_numpy(self.engine.dof_pos_np().copy()).to(self.device)
        dof_vel = torch.from_numpy(self.engine.dof_vel_np().copy()).to(self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for k, j in enumerate(self._joint_pos_indices.tolist()):
            dof_pos[env_ids, j] = position[:, k]
        for k, j in enumerate(self._joint_vel_indices.tolist()):
            dof_vel[env_ids, j] = velocity[:, k]
        self.engine.set_dof_pos(dof_pos)
        self.engine.set_dof_vel(dof_vel)

    # ------------------------------------------------------------------
    # Name lookups (regex-aware) — used by SceneEntityCfg.
    # ------------------------------------------------------------------
    def find_joints(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, self.joint_names, preserve_order=preserve_order)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, self.link_names, preserve_order=preserve_order)
