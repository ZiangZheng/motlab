"""PD actuator model with optional latency + noise (torch).

Tasks call :meth:`compute` every control step with the raw policy action
(target joint angles relative to ``default_angles``, scaled by
``action_scale``). The actuator returns torques ready to hand to
``engine.motrix.set_actuator_ctrls``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import torch


def _as_tensor(
    value: Any,
    length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    default: float = 0.0,
) -> torch.Tensor:
    """Broadcast scalar/sequence into a ``(length,)`` tensor on ``device``."""
    if value is None:
        return torch.full((length,), default, dtype=dtype, device=device)
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(length).clone()
    if t.shape != (length,):
        raise ValueError(f"Expected shape ({length},), got {tuple(t.shape)}")
    return t.to(device=device, dtype=dtype)


@dataclass
class PDActuator:
    num_envs: int
    num_actuators: int
    stiffness: Any  # scalar / sequence / tensor of length ``num_actuators``
    damping: Any
    default_angles: Any
    action_scale: float = 1.0
    latency_steps: int = 0
    action_noise_scale: float = 0.0
    torque_limit: Optional[float] = None
    device: torch.device | str = "cpu"
    generator: Optional[torch.Generator] = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.stiffness = _as_tensor(self.stiffness, self.num_actuators, self.device)
        self.damping = _as_tensor(self.damping, self.num_actuators, self.device)
        self.default_angles = _as_tensor(self.default_angles, self.num_actuators, self.device)
        self._latency_buffer: deque[torch.Tensor] = deque(maxlen=max(1, self.latency_steps + 1))

    def reset(self) -> None:
        self._latency_buffer.clear()

    def compute(
        self,
        action: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Return PD torques given the latest policy action and joint state."""
        action = action.to(self.device)
        if self.action_noise_scale > 0:
            noise = torch.randn(
                action.shape, device=self.device, dtype=action.dtype, generator=self.generator
            ) * self.action_noise_scale
            action = action + noise

        self._latency_buffer.append(action.clone())
        if self.latency_steps > 0 and len(self._latency_buffer) > self.latency_steps:
            delayed = self._latency_buffer[0]
        else:
            delayed = self._latency_buffer[-1]

        target = delayed * self.action_scale + self.default_angles
        torque = self.stiffness * (target - dof_pos) - self.damping * dof_vel

        if self.torque_limit is not None:
            torque = torch.clamp(torque, -self.torque_limit, self.torque_limit)
        return torque.to(torch.float32)
