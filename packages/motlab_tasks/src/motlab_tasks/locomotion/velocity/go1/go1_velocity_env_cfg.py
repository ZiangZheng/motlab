"""Go1 — track a uniform base velocity command on flat ground."""

from __future__ import annotations

from motlab import mdp
from motlab_assets import GO1_CFG
from motlab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from motlab.managers.manager_term_cfg import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from motlab.mdp.actions.joint_actions import JointPositionActionCfg
from motlab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from motlab.registry import envcfg
from motlab.scene.interactive_scene_cfg import InteractiveSceneCfg
from motlab.sim.simulation_cfg import SimulationCfg
from motlab.utils.configclass import configclass


@configclass
class Go1SceneCfg(InteractiveSceneCfg):
    robot: object = GO1_CFG


@configclass
class CommandsCfg:
    base_velocity: UniformVelocityCommandCfg = UniformVelocityCommandCfg(
        class_type=mdp.commands.UniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos: JointPositionActionCfg = JointPositionActionCfg(
        class_type=mdp.actions.JointPositionAction,
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.base_lin_vel, scale=2.0)
        base_ang_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.base_ang_vel, scale=0.25)
        projected_gravity: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.projected_gravity)
        velocity_commands: ObservationTermCfg = ObservationTermCfg(
            func=mdp.observations.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.joint_pos_rel)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.joint_vel_rel, scale=0.05)
        actions: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.last_action)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_base: EventTermCfg = EventTermCfg(
        func=mdp.events.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints: EventTermCfg = EventTermCfg(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    track_lin_vel_xy: RewardTermCfg = RewardTermCfg(
        func=mdp.rewards.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z: RewardTermCfg = RewardTermCfg(
        func=mdp.rewards.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z: RewardTermCfg = RewardTermCfg(func=mdp.rewards.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy: RewardTermCfg = RewardTermCfg(func=mdp.rewards.ang_vel_xy_l2, weight=-0.05)
    flat_orient: RewardTermCfg = RewardTermCfg(func=mdp.rewards.flat_orientation_l2, weight=-1.0)
    joint_torques: RewardTermCfg = RewardTermCfg(func=mdp.rewards.joint_torques_l2, weight=-1.0e-5)
    joint_acc: RewardTermCfg = RewardTermCfg(func=mdp.rewards.joint_acc_l2, weight=-2.5e-7)
    action_rate: RewardTermCfg = RewardTermCfg(func=mdp.rewards.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.terminations.time_out, time_out=True)
    base_low: TerminationTermCfg = TerminationTermCfg(
        func=mdp.terminations.root_height_below_minimum, params={"minimum_height": 0.2}
    )


@envcfg("go1-velocity")
@configclass
class Go1VelocityEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=0.005, device="cpu")
    scene: Go1SceneCfg = Go1SceneCfg(num_envs=128)
    decimation: int = 4
    episode_length_s: float = 20.0
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
