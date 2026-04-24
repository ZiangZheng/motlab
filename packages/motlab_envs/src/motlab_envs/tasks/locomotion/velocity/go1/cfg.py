"""Go1 velocity-tracking task config."""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab_envs import registry
from motlab_envs.asset_zoo.robots.unitree_go1 import GO1_SCENE_XML
from motlab_envs.base import (
    AssetCfg,
    CommandsCfg,
    ControlCfg,
    DomainRandCfg,
    EnvCfg,
    NormalizationCfg,
    RewardCfg,
)


@registry.envcfg("go1-velocity")
@dataclass
class Go1VelocityEnvCfg(EnvCfg):
    """Velocity-tracking locomotion for the Unitree Go1 quadruped."""

    sim_dt: float = 0.005
    ctrl_dt: float = 0.02  # 4 sim substeps per control step (50 Hz policy)
    max_episode_seconds: float = 20.0
    render_spacing: float = 1.5

    asset: AssetCfg = field(
        default_factory=lambda: AssetCfg(
            model_file=str(GO1_SCENE_XML),
            body_name="trunk",
        )
    )

    control: ControlCfg = field(
        default_factory=lambda: ControlCfg(
            stiffness=20.0,
            damping=0.5,
            action_scale=0.25,
        )
    )

    commands: CommandsCfg = field(
        default_factory=lambda: CommandsCfg(
            resample_seconds=10.0,
            vel_limit=((-1.0, -0.5, -1.0), (1.0, 0.5, 1.0)),
        )
    )

    normalization: NormalizationCfg = field(
        default_factory=lambda: NormalizationCfg(
            lin_vel=2.0,
            ang_vel=0.25,
            dof_pos=1.0,
            dof_vel=0.05,
        )
    )

    reward: RewardCfg = field(
        default_factory=lambda: RewardCfg(
            scales={
                "track_lin_vel_xy": 1.0,
                "track_ang_vel_z": 0.5,
                "lin_vel_z": -2.0,
                "ang_vel_xy": -0.05,
                "orientation": -5.0,
                "joint_torques": -2.5e-5,
                "joint_accel": -2.5e-7,
                "action_rate": -0.01,
                "alive": 0.1,
            },
            tracking_sigma=0.25,
        )
    )

    domain_rand: DomainRandCfg = field(
        default_factory=lambda: DomainRandCfg(
            obs_noise_scale=0.0,  # bump up once a policy is training
            action_noise_scale=0.0,
            action_latency_steps=0,
        )
    )

    # Termination thresholds.
    base_min_height: float = 0.18
    gravity_z_terminate_threshold: float = -0.5  # projected gravity z above this → fallen

    # Reset pose noise.
    reset_base_pos_noise: float = 0.05
    reset_base_rot_noise: float = 0.1
    reset_joint_noise: float = 0.1
