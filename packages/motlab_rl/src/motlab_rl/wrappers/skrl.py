"""SKRL env adapter.

SKRL accepts any gymnasium-like env with:
- ``num_envs``, ``observation_space``, ``action_space``, ``device``
- ``reset() -> (obs, info)``
- ``step(actions) -> (obs, rewards, terminated, truncated, info)``
- ``close()``

MotLab envs are torch-native; this wrapper only reshapes / re-devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from motlab_envs.env import TorchEnv


class SkrlVecEnv:
    def __init__(self, env: "TorchEnv", device: str | torch.device | None = None):
        self.env = env
        self.device = torch.device(device) if device is not None else env.device
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _place(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if t.device != self.device:
            t = t.to(self.device)
        return t

    def reset(self):
        self.env._state = None
        self.env.init_state()
        state = self.env.state
        assert state is not None
        return self._place(state.obs, torch.float32), {}

    def step(self, actions: torch.Tensor):
        actions = actions.detach().to(dtype=torch.float32, device=self.env.device)
        state = self.env.step(actions)
        obs = self._place(state.obs, torch.float32)
        rewards = self._place(state.reward, torch.float32).unsqueeze(-1)
        terminated = self._place(state.terminated, torch.bool).unsqueeze(-1)
        truncated = self._place(state.truncated, torch.bool).unsqueeze(-1)
        return obs, rewards, terminated, truncated, {}

    def render(self, *args, **kwargs):  # pragma: no cover
        return None

    def close(self):  # pragma: no cover
        pass
