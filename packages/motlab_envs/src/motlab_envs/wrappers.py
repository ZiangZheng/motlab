"""Lightweight env wrappers over :class:`TorchEnv` subclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from motlab_envs.base import ABEnv, EnvCfg
from motlab_envs.torch.env import TensorEnvState, TorchEnv


@dataclass
class _WrapperBase:
    env: TorchEnv

    def __getattr__(self, item):
        return getattr(self.env, item)


class ObsNoiseWrapper(_WrapperBase):
    """Add Gaussian noise to the observation after each step."""

    scale: float
    generator: Optional[torch.Generator]

    def __init__(self, env: TorchEnv, scale: float, seed: int | None = None):
        self.env = env
        self.scale = scale
        if seed is not None:
            g = torch.Generator(device=env.device)
            g.manual_seed(seed)
            self.generator = g
        else:
            self.generator = None

    def step(self, actions: torch.Tensor) -> TensorEnvState:
        state = self.env.step(actions)
        if self.scale > 0:
            noise = torch.randn(
                state.obs.shape, device=state.obs.device, dtype=state.obs.dtype,
                generator=self.generator,
            ) * self.scale
            state.obs = state.obs + noise
        return state


class ActionLatencyWrapper(_WrapperBase):
    """Delay actions by ``latency_steps`` control steps."""

    latency_steps: int

    def __init__(self, env: TorchEnv, latency_steps: int):
        self.env = env
        self.latency_steps = latency_steps
        self._buf: list[torch.Tensor] = []

    def step(self, actions: torch.Tensor) -> TensorEnvState:
        if self.latency_steps <= 0:
            return self.env.step(actions)
        self._buf.append(actions)
        if len(self._buf) <= self.latency_steps:
            delayed = torch.zeros_like(actions)
        else:
            delayed = self._buf.pop(0)
        return self.env.step(delayed)


class ClipActionWrapper(_WrapperBase):
    """Clip actions to the env's action-space bounds before stepping."""

    def __init__(self, env: TorchEnv):
        self.env = env
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)
        self._low = low
        self._high = high

    def step(self, actions: torch.Tensor) -> TensorEnvState:
        return self.env.step(torch.clamp(actions, self._low, self._high))


__all__ = [
    "ABEnv",
    "EnvCfg",
    "TorchEnv",
    "TensorEnvState",
    "ActionLatencyWrapper",
    "ClipActionWrapper",
    "ObsNoiseWrapper",
]
