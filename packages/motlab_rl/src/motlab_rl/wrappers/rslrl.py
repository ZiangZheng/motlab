"""rsl_rl VecEnv adapter for :class:`motlab.envs.ManagerBasedRLEnv`.

rsl_rl's runner expects:
- attributes ``num_envs``, ``num_actions``, ``max_episode_length``,
  ``episode_length_buf``, ``device``, ``cfg``
- methods ``get_observations() -> TensorDict``, ``reset()``,
  ``step(actions) -> (TensorDict, rewards, dones, extras)``
- ``extras["time_outs"]`` holds the truncation mask.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from motlab.envs.manager_based_rl_env import ManagerBasedRLEnv


class RslrlVecEnv:
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        device: str | torch.device | None = None,
    ) -> None:
        self.env = env
        self.device = torch.device(device) if device is not None else env.device
        self.num_envs = env.num_envs
        self.num_actions = int(env.action_dim)
        self.max_episode_length = int(env.max_episode_length)
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf[:] = value

    def _to_tensordict(self, obs: dict[str, torch.Tensor]) -> TensorDict:
        return TensorDict(
            {k: v.to(dtype=torch.float32, device=self.device) for k, v in obs.items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._to_tensordict(self.env._compute_obs())

    def reset(self):
        obs, info = self.env.reset()
        return self._to_tensordict(obs), info

    def step(self, actions: torch.Tensor):
        actions = actions.detach().to(dtype=torch.float32, device=self.env.device)
        obs, reward, terminated, truncated, info = self.env.step(actions)
        dones = (terminated | truncated).to(torch.long)
        extras = {
            "time_outs": truncated.to(torch.long).to(self.device),
            **info,
        }
        return (
            self._to_tensordict(obs),
            reward.to(dtype=torch.float32, device=self.device),
            dones.to(self.device),
            extras,
        )
