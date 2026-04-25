"""Termination functions — each returns a boolean ``(num_envs,)`` tensor."""

from __future__ import annotations

import torch

from motlab.managers.manager_term_cfg import SceneEntityCfg


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    h = env.scene[asset_cfg.name].data.root_pos_w[..., 2]
    return h < minimum_height


def joint_pos_out_of_limits(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    lim = asset.data.soft_joint_pos_limits
    if asset_cfg.joint_ids is not None:
        ids = torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        pos = pos.index_select(-1, ids)
        lim = lim.index_select(-2, ids)
    too_low = (pos < lim[..., 0]).any(dim=-1)
    too_high = (pos > lim[..., 1]).any(dim=-1)
    return too_low | too_high


def cartpole_pole_out_of_bounds(env, threshold: float = 0.5) -> torch.Tensor:
    asset = env.scene["robot"]
    pole = asset.data.joint_pos[..., -1]
    return torch.abs(pole) > threshold


def cartpole_cart_out_of_bounds(env, threshold: float = 2.0) -> torch.Tensor:
    asset = env.scene["robot"]
    cart = asset.data.joint_pos[..., 0]
    return torch.abs(cart) > threshold
