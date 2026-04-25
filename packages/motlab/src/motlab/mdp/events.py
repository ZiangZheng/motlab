"""Event functions — fired on startup / reset / interval."""

from __future__ import annotations

import torch

from motlab.managers.manager_term_cfg import SceneEntityCfg


def reset_joints_by_offset(
    env,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return
    nj = asset.num_joints
    pos = asset.data.default_joint_pos[env_ids].clone()
    vel = asset.data.default_joint_vel[env_ids].clone()
    pos += torch.empty(len(env_ids), nj, device=env.device).uniform_(*position_range)
    vel += torch.empty(len(env_ids), nj, device=env.device).uniform_(*velocity_range)
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def reset_root_state_uniform(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    asset = env.scene[asset_cfg.name]
    if not asset.has_floating_base:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return
    n = len(env_ids)

    # ---- Pose -----------------------------------------------------------
    init = asset.cfg.init_state
    pos = torch.tensor(init.pos, device=env.device, dtype=torch.float32).expand(n, -1).clone()
    quat_wxyz = torch.tensor(init.rot, device=env.device, dtype=torch.float32).expand(n, -1).clone()
    if pose_range:
        for axis, key in [(0, "x"), (1, "y"), (2, "z")]:
            if key in pose_range:
                lo, hi = pose_range[key]
                pos[:, axis] += torch.empty(n, device=env.device).uniform_(lo, hi)
    pose = torch.cat([pos, quat_wxyz], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)

    # ---- Velocity -------------------------------------------------------
    vel = torch.zeros(n, 6, device=env.device)
    if velocity_range:
        keymap = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
        for k, idx in keymap.items():
            if k in velocity_range:
                lo, hi = velocity_range[k]
                vel[:, idx] = torch.empty(n, device=env.device).uniform_(lo, hi)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
