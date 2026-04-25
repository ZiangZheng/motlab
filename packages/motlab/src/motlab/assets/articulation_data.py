"""Read-only torch tensor view over an :class:`Articulation`'s state.

Mirrors IsaacLab's :class:`ArticulationData` field names so MDP functions
written against IsaacLab can be ported with minimal edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from motlab.utils.math import (
    convert_quat,
    project_gravity_b,
    quat_apply_inverse,
    yaw_from_quat,
)

if TYPE_CHECKING:  # pragma: no cover
    from motlab.assets.articulation import Articulation


class ArticulationData:
    """Lazy accessors over an :class:`Articulation`'s engine buffers.

    All tensors are computed on demand — calling sites should cache the
    results within a step if they hit them more than once.
    """

    def __init__(self, articulation: "Articulation") -> None:
        self._a = articulation

    # ------------------------------------------------------------------
    # Joints (IsaacLab order = motlab joint order)
    # ------------------------------------------------------------------
    @property
    def joint_names(self) -> list[str]:
        return self._a.joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._a._joint_pos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._a._joint_vel()

    @property
    def joint_acc(self) -> torch.Tensor:
        return self._a._joint_acc

    @property
    def joint_pos_target(self) -> torch.Tensor:
        return self._a._joint_pos_target

    @property
    def applied_torque(self) -> torch.Tensor:
        return self._a._applied_torque

    @property
    def computed_torque(self) -> torch.Tensor:
        return self._a._computed_torque

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._a._default_joint_pos.expand(self._a.num_envs, -1)

    @property
    def default_joint_vel(self) -> torch.Tensor:
        return self._a._default_joint_vel.expand(self._a.num_envs, -1)

    @property
    def soft_joint_pos_limits(self) -> torch.Tensor:
        return self._a._soft_joint_pos_limits

    # ------------------------------------------------------------------
    # Floating base (root link) — wxyz quaternion convention
    # ------------------------------------------------------------------
    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._a._root_pos_w()

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._a._root_quat_w()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._a._root_lin_vel_w()

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._a._root_ang_vel_w()

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.root_lin_vel_w)

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        return quat_apply_inverse(self.root_quat_w, self.root_ang_vel_w)

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        return project_gravity_b(self.root_quat_w)

    @property
    def heading_w(self) -> torch.Tensor:
        return yaw_from_quat(self.root_quat_w)

    # IsaacLab compatibility aliases
    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self.root_pos_w

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self.root_quat_w

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self.root_lin_vel_w

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self.root_ang_vel_w

    # ------------------------------------------------------------------
    # Per-link state
    # ------------------------------------------------------------------
    @property
    def link_pos_w(self) -> torch.Tensor:
        """Shape ``(num_envs, num_links, 3)``."""
        return self._a._link_pos_w()

    @property
    def link_quat_w(self) -> torch.Tensor:
        """Shape ``(num_envs, num_links, 4)`` in ``wxyz``."""
        return self._a._link_quat_w()
