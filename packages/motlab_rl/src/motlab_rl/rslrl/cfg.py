"""rsl_rl runner cfg schema (PyTorch only).

Fields mirror ``templates/rslrl_config.yaml`` — keep them in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RslrlAlgorithmCfg:
    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class RslrlPolicyCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"


@dataclass
class RslrlRunnerCfg:
    class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 24
    max_iterations: int = 500
    save_interval: int = 50
    experiment_name: str = "motlab-rslrl"
    run_name: str = ""
    logger: str = "tensorboard"
    seed: Optional[int] = 42
    empirical_normalization: bool = False


@dataclass
class RslrlCfg:
    """Top-level cfg. Subclass in `tasks/` to set env-specific values."""

    num_envs: int = 1024
    algorithm: RslrlAlgorithmCfg = field(default_factory=RslrlAlgorithmCfg)
    policy: RslrlPolicyCfg = field(default_factory=RslrlPolicyCfg)
    runner: RslrlRunnerCfg = field(default_factory=RslrlRunnerCfg)
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"policy": ["policy"], "critic": ["policy"]}
    )

    def to_runner_dict(self) -> dict:
        """Flatten into the dict shape rsl_rl's OnPolicyRunner expects."""
        import dataclasses

        def _d(obj):
            return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj

        return {
            "algorithm": _d(self.algorithm),
            "policy": _d(self.policy),
            "obs_groups": dict(self.obs_groups),
            **_d(self.runner),
        }
