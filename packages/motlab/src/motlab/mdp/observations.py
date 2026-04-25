"""Observation functions — return ``(num_envs, term_dim)`` torch tensors."""

from __future__ import annotations

import torch

from motlab.managers.manager_term_cfg import SceneEntityCfg


# ---------------------------------------------------------------------------
# Joint state
# ---------------------------------------------------------------------------
def joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is not None:
        ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        return asset.data.joint_pos.index_select(-1, ids)
    return asset.data.joint_pos


def joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos - asset.data.default_joint_pos
    if asset_cfg.joint_ids is not None:
        ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        return diff.index_select(-1, ids)
    return diff


def joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is not None:
        ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        return asset.data.joint_vel.index_select(-1, ids)
    return asset.data.joint_vel


def joint_vel_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    diff = asset.data.joint_vel - asset.data.default_joint_vel
    if asset_cfg.joint_ids is not None:
        ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        return diff.index_select(-1, ids)
    return diff


# ---------------------------------------------------------------------------
# Root state
# ---------------------------------------------------------------------------
def base_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_b


def base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_ang_vel_b


def projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.projected_gravity_b


def root_pos_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w[..., 2:3]


# ---------------------------------------------------------------------------
# Action / command observations
# ---------------------------------------------------------------------------
def last_action(env) -> torch.Tensor:
    if env.action_manager is None:
        return torch.zeros(env.num_envs, 0, device=env.device)
    return env.action_manager.action


def generated_commands(env, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)
