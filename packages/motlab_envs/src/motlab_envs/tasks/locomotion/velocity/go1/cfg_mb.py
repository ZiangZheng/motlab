"""Go1 velocity-tracking — manager-based variant.

This config assembles the MDP library (``tasks/locomotion/mdp``) into a
:class:`ManagerBasedEnvCfg` and registers it under ``go1-velocity-mb``.
The direct-style counterpart lives in :mod:`.cfg` / :mod:`.go1_velocity`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab_envs import registry
from motlab_envs.asset_zoo.robots.unitree_go1 import (
    ACTION_SCALE,
    DAMPING,
    DEFAULT_JOINT_ANGLES,
    GO1_SCENE_XML,
    STIFFNESS,
    TORQUE_LIMIT,
)
from motlab_envs.base import AssetCfg
from motlab_envs.manager_env import ManagerBasedEnvCfg
from motlab_envs.managers import (
    ActionsCfg,
    CommandsCfgMB,
    CommandTermCfg,
    ObservationsCfg,
    ObservationTermCfg,
    PDActionCfg,
    RewardsCfg,
    RewardTermCfg,
    TerminationsCfg,
    TerminationTermCfg,
)
from motlab_envs.tasks.locomotion.mdp import (
    commands as mdp_commands,
    observations as mdp_obs,
    rewards as mdp_rewards,
    terminations as mdp_terms,
)
from motlab_envs.tasks.locomotion.velocity.velocity_env import LocomotionVelocityEnv


def _default_observations() -> ObservationsCfg:
    return ObservationsCfg(terms={
        "base_lin_vel":       ObservationTermCfg(func=mdp_obs.base_lin_vel_body, scale=2.0),
        "base_ang_vel":       ObservationTermCfg(func=mdp_obs.base_ang_vel_body, scale=0.25),
        "projected_gravity":  ObservationTermCfg(func=mdp_obs.projected_gravity),
        "velocity_command":   ObservationTermCfg(func=mdp_obs.velocity_command, params={"name": "twist"}),
        "joint_pos":          ObservationTermCfg(
            func=mdp_obs.joint_pos_rel, params={"default_angles": DEFAULT_JOINT_ANGLES}, scale=1.0,
        ),
        "joint_vel":          ObservationTermCfg(func=mdp_obs.joint_vel, scale=0.05),
        "last_actions":       ObservationTermCfg(func=mdp_obs.last_actions),
    })


def _default_rewards() -> RewardsCfg:
    sigma_lin = 0.5
    sigma_ang = 0.5
    return RewardsCfg(terms={
        "track_lin_vel_xy":  RewardTermCfg(
            func=mdp_rewards.track_lin_vel_xy_exp,
            params={"command_name": "twist", "std": sigma_lin}, weight=1.0,
        ),
        "track_ang_vel_z":   RewardTermCfg(
            func=mdp_rewards.track_ang_vel_z_exp,
            params={"command_name": "twist", "std": sigma_ang}, weight=0.5,
        ),
        "lin_vel_z":         RewardTermCfg(func=mdp_rewards.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy":        RewardTermCfg(func=mdp_rewards.ang_vel_xy_l2, weight=-0.05),
        "orientation":       RewardTermCfg(func=mdp_rewards.flat_orientation_l2, weight=-5.0),
        "joint_torques":     RewardTermCfg(func=mdp_rewards.joint_torques_l2, weight=-2.5e-5),
        "joint_accel":       RewardTermCfg(func=mdp_rewards.joint_acc_l2, weight=-2.5e-7),
        "action_rate":       RewardTermCfg(func=mdp_rewards.action_rate_l2, weight=-0.01),
        "alive":             RewardTermCfg(func=mdp_rewards.is_alive, weight=0.1),
    })


def _default_terminations() -> TerminationsCfg:
    return TerminationsCfg(terms={
        "base_low":     TerminationTermCfg(
            func=mdp_terms.base_height_below_minimum, params={"minimum_height": 0.18},
        ),
        "bad_orientation": TerminationTermCfg(
            func=mdp_terms.bad_orientation, params={"gravity_z_threshold": -0.5},
        ),
    })


def _default_commands() -> CommandsCfgMB:
    return CommandsCfgMB(terms={
        "twist": CommandTermCfg(
            func=mdp_commands.uniform_velocity,
            params={"low": (-1.0, -0.5, -1.0), "high": (1.0, 0.5, 1.0)},
            resample_seconds=10.0,
        ),
    })


def _default_actions() -> ActionsCfg:
    return ActionsCfg(joint_pd=PDActionCfg(
        default_angles=DEFAULT_JOINT_ANGLES,
        stiffness=STIFFNESS,
        damping=DAMPING,
        action_scale=ACTION_SCALE,
        torque_limit=TORQUE_LIMIT,
    ))


@registry.envcfg("go1-velocity-mb")
@dataclass
class Go1VelocityMBEnvCfg(ManagerBasedEnvCfg):
    """Manager-based velocity tracking for the Unitree Go1."""

    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 20.0
    render_spacing: float = 1.5
    reset_joint_noise: float = 0.1
    reset_base_pos_noise: float = 0.05

    asset: AssetCfg = field(default_factory=lambda: AssetCfg(
        model_file=str(GO1_SCENE_XML), body_name="trunk",
    ))

    observations: ObservationsCfg = field(default_factory=_default_observations)
    rewards: RewardsCfg = field(default_factory=_default_rewards)
    terminations: TerminationsCfg = field(default_factory=_default_terminations)
    commands: CommandsCfgMB = field(default_factory=_default_commands)
    actions: ActionsCfg = field(default_factory=_default_actions)


@registry.env("go1-velocity-mb", sim_backend="torch")
class Go1VelocityMBEnv(LocomotionVelocityEnv):
    """Manager-based Go1 velocity env — thin subclass that picks up the cfg."""

    _cfg: Go1VelocityMBEnvCfg
