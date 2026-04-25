"""skrl Wrapper adapter for :class:`motlab.envs.ManagerBasedRLEnv`.

skrl's agents/trainers consume a :class:`skrl.envs.wrappers.torch.base.Wrapper`
and require gymnasium-style ``observation_space`` / ``action_space`` plus
``reset() -> (obs, info)`` / ``step(actions) -> (obs, rew, term, trunc, info)``
where rewards / terminations / truncations are shaped ``(num_envs, 1)``.

We expose only the ``policy`` obs group as a flat ``(N, obs_dim)`` tensor —
that's the single group every motlab task currently defines.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import torch
from skrl.envs.wrappers.torch.base import Wrapper

from motlab.envs.manager_based_rl_env import ManagerBasedRLEnv

_OBS_GROUP = "policy"


class SkrlVecEnv(Wrapper):
    def __init__(self, env: ManagerBasedRLEnv) -> None:
        super().__init__(env)
        self._mlenv = env
        self._num_envs = env.num_envs
        self._device = torch.device(env.device)

        obs_dim = self._infer_obs_dim()
        act_dim = int(env.action_dim)
        self._observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32
        )

    def _infer_obs_dim(self) -> int:
        obs = self._mlenv._compute_obs()
        if _OBS_GROUP not in obs:
            raise KeyError(
                f"SkrlVecEnv expects obs group {_OBS_GROUP!r}, got {list(obs)}"
            )
        return int(obs[_OBS_GROUP].shape[-1])

    def _flatten_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return obs[_OBS_GROUP].to(dtype=torch.float32, device=self._device)

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._action_space

    @property
    def state_space(self) -> gymnasium.Space | None:
        return None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def state(self) -> torch.Tensor | None:
        return None

    def reset(self):
        obs, info = self._mlenv.reset()
        return self._flatten_obs(obs), info

    def step(self, actions: torch.Tensor):
        actions = actions.detach().to(dtype=torch.float32, device=self._mlenv.device)
        obs, reward, terminated, truncated, info = self._mlenv.step(actions)
        return (
            self._flatten_obs(obs),
            reward.to(dtype=torch.float32, device=self._device).unsqueeze(-1),
            terminated.to(dtype=torch.bool, device=self._device).unsqueeze(-1),
            truncated.to(dtype=torch.bool, device=self._device).unsqueeze(-1),
            info,
        )

    def render(self, *args, **kwargs):
        return None

    def close(self) -> None:
        pass
