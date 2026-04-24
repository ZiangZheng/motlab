"""Default PPO configs for go1-velocity (SKRL-torch + rsl_rl).

These are starting-point hyperparameters cribbed from mjlab's velocity task.
The solver in MotrixSim panics with ~256 simultaneous falling quadrupeds, so
``num_envs`` is conservatively set to 128 here — raise once the policy is
stable enough that most envs stay upright.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab_rl.registry import rlcfg
from motlab_rl.rslrl.cfg import RslrlAlgorithmCfg, RslrlCfg, RslrlPolicyCfg, RslrlRunnerCfg
from motlab_rl.skrl.config import SkrlCfg, SkrlRunnerCfg


@rlcfg("go1-velocity", backend="torch")
@dataclass
class Go1VelocitySkrlCfg(SkrlCfg):
    num_envs: int = 128
    rollouts: int = 24
    experiment_name: str = "go1_velocity"
    runner: SkrlRunnerCfg = field(
        default_factory=lambda: SkrlRunnerCfg(
            timesteps=500_000,
            rollouts=24,
            learning_rate=1e-3,
            entropy_loss_scale=0.008,
        )
    )


@rlcfg("go1-velocity", backend="torch")
@dataclass
class Go1VelocityRslrlCfg(RslrlCfg):
    num_envs: int = 128
    algorithm: RslrlAlgorithmCfg = field(
        default_factory=lambda: RslrlAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            entropy_coef=0.008,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            value_loss_coef=1.0,
            max_grad_norm=1.0,
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
            max_iterations=1500,
            experiment_name="go1_velocity",
        )
    )
