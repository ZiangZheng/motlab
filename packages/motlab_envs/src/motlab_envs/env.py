"""Batched torch-tensor env. Subclass to build concrete tasks.

The underlying physics buffer (``state.data``) is still a MotrixSim
``SceneData``, but task code touches it only through the helpers in
:mod:`motlab_envs.engine.motrix`, which return / accept torch tensors.
All observation / reward / termination state in :class:`TensorEnvState`
is torch-native on ``cfg.device``.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any

import torch

from motlab_envs.base import ABEnv, EnvCfg
from motlab_envs.engine import motrix


@dataclass
class TensorEnvState:
    """All mutable per-step state. ``data`` owns the physics buffers."""

    data: Any  # motrixsim.SceneData
    obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor  # bool
    truncated: torch.Tensor  # bool
    info: dict

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    def replace(self, **updates) -> "TensorEnvState":
        return dataclasses.replace(self, **updates)

    def validate(self) -> None:
        n = self.data.shape[0]
        assert self.reward.shape == (n,), self.reward.shape
        assert self.terminated.shape == (n,), self.terminated.shape
        assert self.truncated.shape == (n,), self.truncated.shape


class TorchEnv(ABEnv):
    """Batched env with a MotrixSim scene underneath.

    Subclasses implement :meth:`apply_action`, :meth:`update_state`, and
    :meth:`reset`. :meth:`step` wires them together and handles
    auto-reset + truncation bookkeeping.
    """

    def __init__(self, cfg: EnvCfg, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self._device: torch.device = torch.device(cfg.device)
        self._model = motrix.load_model(cfg.model_file, sim_dt=cfg.sim_dt)
        self._state: TensorEnvState | None = None

    # -- ABEnv surface ------------------------------------------------------
    @property
    def cfg(self) -> EnvCfg:
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self):
        return self._model

    @property
    def state(self) -> TensorEnvState | None:
        return self._state

    @property
    def render_spacing(self) -> float:
        return self._cfg.render_spacing

    # -- Lifecycle ----------------------------------------------------------
    def init_state(self) -> TensorEnvState:
        obs_dim = int(torch.tensor(self.observation_space.shape).prod().item())
        obs = torch.zeros((self._num_envs, obs_dim), dtype=torch.float32, device=self._device)
        reward = torch.zeros((self._num_envs,), dtype=torch.float32, device=self._device)
        # Start marked done so the first step / init_state triggers a reset.
        terminated = torch.ones((self._num_envs,), dtype=torch.bool, device=self._device)
        truncated = torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device)
        info = {
            "steps": torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device),
        }
        data = motrix.make_batched_data(self._model, self._num_envs)
        self._state = TensorEnvState(data, obs, reward, terminated, truncated, info)
        self._reset_done_envs()
        self._state.validate()
        return self._state

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        state = self._state
        done = state.done
        if not bool(done.any().item()):
            return
        state.info["steps"][done] = 0
        # SceneData indexing uses numpy boolean masks.
        import numpy as np  # localized: engine buffer is numpy-backed
        mask_np = done.detach().cpu().numpy()
        sub_data = state.data[mask_np]
        obs, info1 = self.reset(sub_data)
        state.obs[done] = obs.to(device=state.obs.device, dtype=state.obs.dtype)
        if info1:
            _merge_info(state.info, info1, done)

    def _update_truncate(self) -> None:
        assert self._state is not None
        if not self._cfg.max_episode_steps:
            return
        self._state.truncated = self._state.info["steps"] >= self._cfg.max_episode_steps

    # -- Task hooks ---------------------------------------------------------
    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: TensorEnvState) -> TensorEnvState:
        """Write actions into ``state.data`` (via ``engine.motrix``)."""

    @abc.abstractmethod
    def update_state(self, state: TensorEnvState) -> TensorEnvState:
        """Compute obs / reward / termination after the physics step."""

    @abc.abstractmethod
    def reset(self, data) -> tuple[torch.Tensor, dict]:
        """Reset the sub-batch referenced by ``data``; return ``(obs, info)``."""

    # -- Step loop ----------------------------------------------------------
    def physics_step(self) -> None:
        assert self._state is not None
        for _ in range(self._cfg.sim_substeps):
            motrix.physics_step(self._model, self._state.data)

    def _prev_physics_step(self) -> None:
        assert self._state is not None
        s = self._state
        s.reward.zero_()
        s.terminated.zero_()
        s.truncated.zero_()

    def step(self, actions: torch.Tensor) -> TensorEnvState:
        if self._state is None:
            self.init_state()
        assert self._state is not None
        self._prev_physics_step()
        actions = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(
            actions, dtype=torch.float32, device=self._device
        )
        self._state = self.apply_action(actions, self._state)
        self.physics_step()
        self._state = self.update_state(self._state)
        self._state.info["steps"] += 1
        self._update_truncate()
        self._reset_done_envs()
        return self._state


def _merge_info(dst: dict, new_values: dict, mask: torch.Tensor) -> None:
    for key, value in new_values.items():
        if key not in dst:
            dst[key] = value
            continue
        if isinstance(value, torch.Tensor) and isinstance(dst[key], torch.Tensor):
            dst[key][mask] = value
        elif isinstance(value, dict) and isinstance(dst[key], dict):
            _merge_info(dst[key], value, mask)
        else:
            dst[key] = value
