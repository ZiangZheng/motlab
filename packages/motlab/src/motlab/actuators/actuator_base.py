"""Base class for joint actuator models."""

from __future__ import annotations

from typing import Sequence

import torch

from motlab.actuators.actuator_cfg import ActuatorBaseCfg


class ActuatorBase:
    """Base actuator group: a slice of joints sharing PD gains / limits.

    Subclasses implement :meth:`compute` to produce applied joint torques
    given a position target, current joint state, and the desired effort
    range.
    """

    cfg: ActuatorBaseCfg

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: Sequence[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: torch.device | str,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        effort_limit: torch.Tensor,
        velocity_limit: torch.Tensor,
    ) -> None:
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.joint_ids = list(joint_ids)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.num_joints = len(joint_ids)

        # Per-(env, joint) gains and limits — broadcastable to (num_envs, num_joints).
        self.stiffness = stiffness.to(self.device)
        self.damping = damping.to(self.device)
        self.effort_limit = effort_limit.to(self.device)
        self.velocity_limit = velocity_limit.to(self.device)

        # Most recently computed torques (for logging / reward terms).
        self.computed_effort = torch.zeros(num_envs, self.num_joints, device=self.device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.computed_effort.zero_()
            self.applied_effort.zero_()
        else:
            self.computed_effort[env_ids] = 0.0
            self.applied_effort[env_ids] = 0.0

    # ------------------------------------------------------------------
    def compute(
        self,
        control_action: torch.Tensor,  # joint position target, (num_envs, num_joints)
        joint_pos: torch.Tensor,       # current pos,           (num_envs, num_joints)
        joint_vel: torch.Tensor,       # current vel,           (num_envs, num_joints)
    ) -> torch.Tensor:
        raise NotImplementedError
