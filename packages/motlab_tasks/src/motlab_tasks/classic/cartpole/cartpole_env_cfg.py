"""Cartpole — balance the pole upright by sliding the cart."""

from __future__ import annotations

from motlab import mdp
from motlab_assets import CARTPOLE_CFG
from motlab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from motlab.managers.manager_term_cfg import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from motlab.mdp.actions.joint_actions import JointEffortActionCfg
from motlab.registry import envcfg
from motlab.scene.interactive_scene_cfg import InteractiveSceneCfg
from motlab.sim.simulation_cfg import SimulationCfg
from motlab.utils.configclass import configclass


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    robot: object = CARTPOLE_CFG


@configclass
class ActionsCfg:
    joint_effort: JointEffortActionCfg = JointEffortActionCfg(
        class_type=mdp.actions.JointEffortAction,
        asset_name="robot",
        joint_names=["slider"],
        scale=10.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.joint_pos_rel)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.observations.joint_vel_rel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_joints: EventTermCfg = EventTermCfg(
        func=mdp.events.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    alive: RewardTermCfg = RewardTermCfg(func=mdp.rewards.is_alive, weight=1.0)
    pole_pos: RewardTermCfg = RewardTermCfg(
        func=mdp.rewards.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hinge"])},
    )
    cart_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.rewards.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider"])},
    )
    pole_vel: RewardTermCfg = RewardTermCfg(
        func=mdp.rewards.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hinge"])},
    )


@configclass
class TerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.terminations.time_out, time_out=True)
    cart_oob: TerminationTermCfg = TerminationTermCfg(
        func=mdp.terminations.cartpole_cart_out_of_bounds, params={"threshold": 2.0}
    )
    pole_fell: TerminationTermCfg = TerminationTermCfg(
        func=mdp.terminations.cartpole_pole_out_of_bounds, params={"threshold": 1.0}
    )


@envcfg("cartpole")
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=0.005, device="cpu")
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4)
    decimation: int = 4
    episode_length_s: float = 5.0
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
