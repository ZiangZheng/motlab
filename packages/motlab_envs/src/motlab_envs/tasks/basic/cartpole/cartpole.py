"""CartPole — direct-style reference task (no managers, keep it minimal)."""

from __future__ import annotations

import gymnasium as gym
import torch

from motlab_envs import registry
from motlab_envs.engine import motrix
from motlab_envs.env import TensorEnvState, TorchEnv

from .cfg import CartPoleEnvCfg


@registry.env("cartpole", sim_backend="torch")
class CartPoleEnv(TorchEnv):
    """Classic cart-pole. Keep every env alive while |cart| < 0.8 and |angle| < 0.2."""

    _cfg: CartPoleEnvCfg

    def __init__(self, cfg: CartPoleEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self._action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype="float32")
        self._observation_space = gym.spaces.Box(-float("inf"), float("inf"), (4,), dtype="float32")
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._init_dof_pos = motrix.init_dof_pos_tensor(self._model, device=self._device)
        self._init_dof_vel = torch.zeros((self._num_dof_vel,), dtype=torch.float32, device=self._device)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: TensorEnvState) -> TensorEnvState:
        motrix.set_actuator_ctrls(state.data, actions)
        return state

    def update_state(self, state: TensorEnvState) -> TensorEnvState:
        dof_pos = motrix.dof_pos_view(state.data).to(self._device)
        dof_vel = motrix.dof_vel_view(state.data).to(self._device)

        obs = torch.cat([dof_pos, dof_vel], dim=-1).to(torch.float32)
        reward = torch.ones((self._num_envs,), dtype=torch.float32, device=self._device)

        cart_pos = dof_pos[:, 0]
        angle = dof_pos[:, 1]
        terminated = torch.isnan(angle) | (torch.abs(angle) > 0.2) | (cart_pos < -0.8) | (cart_pos > 0.8)

        state.obs = obs
        state.reward = reward
        state.terminated = terminated
        return state

    def reset(self, data) -> tuple[torch.Tensor, dict]:
        cfg = self._cfg
        n = data.shape[0]
        shape_pos = (n, self._num_dof_pos)
        shape_vel = (n, self._num_dof_vel)

        noise_pos = (torch.rand(shape_pos, device=self._device) * 2 - 1) * cfg.reset_noise_scale
        noise_vel = (torch.rand(shape_vel, device=self._device) * 2 - 1) * cfg.reset_noise_scale
        dof_pos = self._init_dof_pos[None, :].expand(n, -1) + noise_pos
        dof_vel = self._init_dof_vel[None, :].expand(n, -1) + noise_vel

        data.reset(self._model)
        motrix.set_dof_vel(data, dof_vel)
        motrix.set_dof_pos(data, self._model, dof_pos)
        obs = torch.cat([dof_pos, dof_vel], dim=-1).to(torch.float32)
        return obs, {}
