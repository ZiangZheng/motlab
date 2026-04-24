"""rsl_rl VecEnv adapter.

rsl_rl's runner expects an object with:
- attributes ``num_envs``, ``num_obs``, ``num_privileged_obs``, ``num_actions``,
  ``max_episode_length``, ``device``
- methods ``get_observations()``, ``reset()``, ``step(actions)``
- step returns ``(obs, rewards, dones, extras)`` with ``extras["time_outs"]``.

The underlying MotLab env is torch-native, so this wrapper is mostly plumbing
— shape adjustments (dones → long, time_outs → long) and device placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from motlab_envs.env import TorchEnv


class RslrlVecEnv:
    def __init__(self, env: "TorchEnv", device: str | torch.device | None = None):
        self.env = env
        self.device = torch.device(device) if device is not None else env.device
        self.num_envs = env.num_envs
        self.num_obs = int(torch.tensor(env.observation_space.shape).prod().item())
        self.num_privileged_obs: Optional[int] = None
        self.num_actions = int(torch.tensor(env.action_space.shape).prod().item())
        self.max_episode_length = env.cfg.max_episode_steps or int(1e9)

        if env.state is None:
            env.init_state()

    # -- internal helpers ---------------------------------------------------
    def _place(self, t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if t.device != self.device:
            t = t.to(self.device)
        return t

    def _obs_tensor(self) -> torch.Tensor:
        assert self.env.state is not None
        return self._place(self.env.state.obs, dtype=torch.float32)

    # -- rsl_rl VecEnv API --------------------------------------------------
    def get_observations(self):
        return self._obs_tensor(), {"observations": {}}

    def reset(self):
        self.env._state = None
        self.env.init_state()
        return self._obs_tensor(), {"observations": {}}

    def step(self, actions: torch.Tensor):
        actions = actions.detach().to(dtype=torch.float32, device=self.env.device)
        state = self.env.step(actions)
        obs = self._place(state.obs, dtype=torch.float32)
        rewards = self._place(state.reward, dtype=torch.float32)
        dones = self._place(state.done.to(torch.long))
        extras = {
            "time_outs": self._place(state.truncated.to(torch.long)),
            "observations": {},
        }
        return obs, rewards, dones, extras
