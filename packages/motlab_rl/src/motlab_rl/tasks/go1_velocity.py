"""Default rsl_rl + skrl PPO configs for the go1 velocity-tracking env."""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab_rl.registry import rlcfg, skrlcfg
from motlab_rl.rslrl.cfg import RslrlAlgorithmCfg, RslrlCfg, RslrlPolicyCfg, RslrlRunnerCfg
from motlab_rl.skrl.cfg import SkrlAgentCfg, SkrlCfg, SkrlPolicyCfg, SkrlRunnerCfg


@rlcfg("go1-velocity")
@dataclass
class Go1VelocityRslrlCfg(RslrlCfg):
    num_envs: int = 128
    algorithm: RslrlAlgorithmCfg = field(
        default_factory=lambda: RslrlAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            entropy_coef=0.005,
            learning_rate=1e-3,
            gamma=0.99,
            lam=0.95,
        )
    )
    policy: RslrlPolicyCfg = field(
        default_factory=lambda: RslrlPolicyCfg(
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            init_noise_std=1.0,
        )
    )
    runner: RslrlRunnerCfg = field(
        default_factory=lambda: RslrlRunnerCfg(
            num_steps_per_env=24,
            max_iterations=2000,
            save_interval=100,
            experiment_name="go1_velocity",
        )
    )


@skrlcfg("go1-velocity")
@dataclass
class Go1VelocitySkrlCfg(SkrlCfg):
    num_envs: int = 128
    agent: SkrlAgentCfg = field(
        default_factory=lambda: SkrlAgentCfg(
            rollouts=24,
            learning_epochs=5,
            mini_batches=4,
            discount_factor=0.99,
            gae_lambda=0.95,
            entropy_loss_scale=0.005,
            learning_rate=1e-3,
        )
    )
    policy: SkrlPolicyCfg = field(
        default_factory=lambda: SkrlPolicyCfg(
            hidden_dims=(512, 256, 128),
            init_noise_std=1.0,
        )
    )
    runner: SkrlRunnerCfg = field(
        default_factory=lambda: SkrlRunnerCfg(
            timesteps=2000 * 24,
            experiment_name="go1_velocity",
        )
    )
