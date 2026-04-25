"""MotrixSim adapter — the *only* place that imports ``motrixsim`` and the
*only* place numpy is allowed (to bridge MotrixSim's NDArray buffers).

Everything above this layer (assets, scene, managers, MDP, tasks, RL) is
torch-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import motrixsim  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(path: str | Path) -> motrixsim.SceneModel:
    """Load a MotrixSim ``SceneModel`` from MJCF/USD on disk."""
    return motrixsim.load_model(str(path))


def load_mjcf_str(mjcf: str) -> motrixsim.SceneModel:
    return motrixsim.load_mjcf_str(mjcf)


# ---------------------------------------------------------------------------
# Name → index helpers (regex-aware)
# ---------------------------------------------------------------------------
def _resolve(model_names: Sequence[str], names: Iterable[str]) -> list[int]:
    """Resolve each name in ``names`` to its index in ``model_names``."""
    out: list[int] = []
    for n in names:
        if n not in model_names:
            raise KeyError(f"name {n!r} not in model (available: {list(model_names)})")
        out.append(model_names.index(n))
    return out


def resolve_joint_indices(model: motrixsim.SceneModel, names: Iterable[str]) -> list[int]:
    return _resolve(model.joint_names, names)


def resolve_actuator_indices(model: motrixsim.SceneModel, names: Iterable[str]) -> list[int]:
    return _resolve(model.actuator_names, names)


def resolve_link_indices(model: motrixsim.SceneModel, names: Iterable[str]) -> list[int]:
    return _resolve(model.link_names, names)


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------
class MotrixEngine:
    """Thin torch-facing wrapper around ``(SceneModel, SceneData)``.

    Buffers are exposed as ``torch.Tensor`` views over MotrixSim's numpy
    arrays. On CPU these views are zero-copy; on GPU we pre-allocate device
    buffers and copy in :meth:`fetch` / :meth:`flush`.
    """

    # -- Construction -------------------------------------------------------
    def __init__(
        self,
        model: motrixsim.SceneModel,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.data = motrixsim.SceneData(model, batch=[self.num_envs])

        # Static counts
        self.num_dof_pos: int = int(model.num_dof_pos)
        self.num_dof_vel: int = int(model.num_dof_vel)
        self.num_joints: int = int(model.num_joints)
        self.num_actuators: int = int(model.num_actuators)
        self.num_links: int = int(model.num_links)
        self.num_bodies: int = int(model.num_bodies)

        # Floating base layout
        self.has_floating_base: bool = len(model.floating_bases) > 0
        if self.has_floating_base:
            fb = model.floating_bases[0]
            # dof_pos: [tx,ty,tz, qx,qy,qz,qw, joints...]
            # dof_vel: [vx,vy,vz, wx,wy,wz, joints...]
            self._fb_pos_start = int(fb.dof_pos_start)  # usually 0
            self._fb_vel_start = int(fb.dof_vel_start)  # usually 0
        else:
            self._fb_pos_start = -1
            self._fb_vel_start = -1

        # Joint slice tables (indices into dof_pos / dof_vel for each joint)
        self.joint_pos_idx: list[int] = list(model.joint_dof_pos_indices)
        self.joint_pos_nums: list[int] = list(model.joint_dof_pos_nums)
        self.joint_vel_idx: list[int] = list(model.joint_dof_vel_indices)
        self.joint_vel_nums: list[int] = list(model.joint_dof_vel_nums)

        # Actuator → joint mapping by name (for joint-space PD on hinge actuators)
        self.actuator_joint_idx: list[int] = []
        for aname in model.actuator_names:
            jname = aname
            if jname not in model.joint_names:
                # fall back: actuator target via actuator object
                act = model.get_actuator(aname)
                jname = getattr(act, "target_name", aname)
            if jname in model.joint_names:
                self.actuator_joint_idx.append(model.joint_names.index(jname))
            else:
                self.actuator_joint_idx.append(-1)

    # -- Buffer access ------------------------------------------------------
    def dof_pos_np(self) -> np.ndarray:
        return self.data.dof_pos

    def dof_vel_np(self) -> np.ndarray:
        return self.data.dof_vel

    def actuator_ctrls_np(self) -> np.ndarray:
        return self.data.actuator_ctrls

    def link_poses_np(self) -> np.ndarray:
        """Per-link poses (B, num_links, 7) with quaternion as ``xyzw``."""
        return self.model.get_link_poses(self.data)

    # ---- torch-tensor views ---------------------------------------------
    def dof_pos(self) -> torch.Tensor:
        return torch.from_numpy(self.dof_pos_np()).to(self.device)

    def dof_vel(self) -> torch.Tensor:
        return torch.from_numpy(self.dof_vel_np()).to(self.device)

    def link_poses(self) -> torch.Tensor:
        """Returns (B, num_links, 7) with quat in ``xyzw``."""
        return torch.from_numpy(self.link_poses_np()).to(self.device)

    # -- State writes ------------------------------------------------------
    def set_dof_pos(self, dof_pos: torch.Tensor) -> None:
        arr = dof_pos.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        self.data.set_dof_pos(arr, self.model)

    def set_dof_vel(self, dof_vel: torch.Tensor) -> None:
        arr = dof_vel.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        self.data.set_dof_vel(arr)

    def set_actuator_ctrls(self, ctrls: torch.Tensor) -> None:
        """Write actuator commands.  ``ctrls`` shape: (B, num_actuators)."""
        # The setter prefers writing through the property to keep numpy view consistent.
        arr = ctrls.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        self.data.actuator_ctrls = arr

    def reset(self) -> None:
        self.data.reset(self.model)

    # -- Stepping ----------------------------------------------------------
    def step(self) -> None:
        motrixsim.step(self.model, self.data)


__all__ = [
    "MotrixEngine",
    "load_model",
    "load_mjcf_str",
    "resolve_actuator_indices",
    "resolve_joint_indices",
    "resolve_link_indices",
]
