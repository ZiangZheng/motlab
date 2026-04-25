"""Reward functions — each returns a ``(num_envs,)`` tensor."""

from __future__ import annotations

import torch

from motlab.managers.manager_term_cfg import SceneEntityCfg


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------
def is_alive(env) -> torch.Tensor:
    if env.termination_manager is None:
        return torch.ones(env.num_envs, device=env.device)
    return (~env.termination_manager.terminated).float()


def is_terminated(env) -> torch.Tensor:
    if env.termination_manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return env.termination_manager.terminated.float()


# ---------------------------------------------------------------------------
# Joint penalties
# ---------------------------------------------------------------------------
def _select(asset_data: torch.Tensor, ids: list[int] | None, env) -> torch.Tensor:
    if ids is None:
        return asset_data
    idx = torch.tensor(ids, device=env.device, dtype=torch.long)
    return asset_data.index_select(-1, idx)


def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    t = _select(asset.data.applied_torque, asset_cfg.joint_ids, env)
    return torch.sum(t ** 2, dim=-1)


def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    a = _select(asset.data.joint_acc, asset_cfg.joint_ids, env)
    return torch.sum(a ** 2, dim=-1)


def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    v = _select(asset.data.joint_vel, asset_cfg.joint_ids, env)
    return torch.sum(v ** 2, dim=-1)


def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos - asset.data.default_joint_pos
    diff = _select(diff, asset_cfg.joint_ids, env)
    return torch.sum(torch.abs(diff), dim=-1)


def joint_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    lim = asset.data.soft_joint_pos_limits
    out = torch.clamp_min(lim[..., 0] - pos, 0.0) + torch.clamp_min(pos - lim[..., 1], 0.0)
    out = _select(out, asset_cfg.joint_ids, env)
    return torch.sum(out, dim=-1)


# ---------------------------------------------------------------------------
# Action regularisation
# ---------------------------------------------------------------------------
def action_rate_l2(env) -> torch.Tensor:
    if env.action_manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    cur = env.action_manager.action
    if not hasattr(env, "_prev_action") or env._prev_action.shape != cur.shape:
        env._prev_action = torch.zeros_like(cur)
    diff = cur - env._prev_action
    env._prev_action = cur.clone()
    return torch.sum(diff ** 2, dim=-1)


def action_l2(env) -> torch.Tensor:
    if env.action_manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(env.action_manager.action ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Locomotion: tracking velocity commands
# ---------------------------------------------------------------------------
def track_lin_vel_xy_exp(env, command_name: str = "base_velocity", std: float = 0.5) -> torch.Tensor:
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[..., 0:2]
    vel = asset.data.root_lin_vel_b[..., 0:2]
    err = torch.sum((cmd - vel) ** 2, dim=-1)
    return torch.exp(-err / (std ** 2))


def track_ang_vel_z_exp(env, command_name: str = "base_velocity", std: float = 0.5) -> torch.Tensor:
    asset = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[..., 2]
    vel = asset.data.root_ang_vel_b[..., 2]
    err = (cmd - vel) ** 2
    return torch.exp(-err / (std ** 2))


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_b[..., 2] ** 2


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(env.scene[asset_cfg.name].data.root_ang_vel_b[..., :2] ** 2, dim=-1)


def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.sum(g[..., :2] ** 2, dim=-1)
