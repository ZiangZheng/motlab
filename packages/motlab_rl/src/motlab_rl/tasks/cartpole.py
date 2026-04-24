"""Default PPO configs for the cartpole env (SKRL-torch + rsl_rl)."""

from __future__ import annotations

from dataclasses import dataclass, field

from motlab_rl.registry import rlcfg
from motlab_rl.rslrl.cfg import RslrlAlgorithmCfg, RslrlCfg, RslrlPolicyCfg, RslrlRunnerCfg
from motlab_rl.skrl.config import SkrlCfg, SkrlRunnerCfg


@rlcfg("cartpole", backend="torch")
@dataclass
class CartPoleSkrlCfg(SkrlCfg):
    num_envs: int = 1024
    rollouts: int = 16
    experiment_name: str = "cartpole"
    runner: SkrlRunnerCfg = field(
        default_factory=lambda: SkrlRunnerCfg(
            timesteps=50_000,
            rollouts=16,
            learning_rate=3e-4,
            entropy_loss_scale=0.0,
        )
    )


@rlcfg("cartpole", backend="torch")
@dataclass
class CartPoleRslrlCfg(RslrlCfg):
    num_envs: int = 1024
    algorithm: RslrlAlgorithmCfg = field(
        default_factory=lambda: RslrlAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            entropy_coef=0.0,
            learning_rate=1e-3,
        )
    )
    policy: RslrlPolicyCfg = field(
        default_factory=lambda: RslrlPolicyCfg(
            actor_hidden_dims=(128, 64),
            critic_hidden_dims=(128, 64),
            init_noise_std=1.0,
        )
    )
    runner: RslrlRunnerCfg = field(
        default_factory=lambda: RslrlRunnerCfg(
            num_steps_per_env=16,
            max_iterations=150,
            experiment_name="cartpole",
        )
    )
