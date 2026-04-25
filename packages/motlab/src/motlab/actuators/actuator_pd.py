"""Ideal joint-space PD actuator."""

from __future__ import annotations

import torch

from motlab.actuators.actuator_base import ActuatorBase


class IdealPDActuator(ActuatorBase):
    """``tau = kp * (q_target - q) - kd * qdot``, clipped to ``effort_limit``."""

    def compute(
        self,
        control_action: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        error_pos = control_action - joint_pos
        torque = self.stiffness * error_pos - self.damping * joint_vel
        self.computed_effort = torque
        self.applied_effort = torch.clamp(torque, -self.effort_limit, self.effort_limit)
        return self.applied_effort
