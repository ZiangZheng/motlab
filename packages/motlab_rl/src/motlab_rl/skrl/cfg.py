"""skrl PPO config schema.

A subset of ``skrl.agents.torch.ppo.PPO_CFG`` exposed as plain dataclasses,
so motlab tasks can register hyperparameters without depending on skrl
being importable at registry-decoration time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkrlAgentCfg:
    rollouts: int = 24
    learning_epochs: int = 5
    mini_batches: int = 4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    grad_norm_clip: float = 1.0
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.01
    value_loss_scale: float = 1.0
    kl_threshold: float = 0.0


@dataclass
class SkrlPolicyCfg:
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class SkrlRunnerCfg:
    timesteps: int = 100_000
    seed: Optional[int] = 42
    experiment_name: str = "motlab-skrl"


@dataclass
class SkrlCfg:
    """Top-level skrl PPO cfg. Subclass in `tasks/` to set env-specific values."""

    num_envs: int = 1024
    agent: SkrlAgentCfg = field(default_factory=SkrlAgentCfg)
    policy: SkrlPolicyCfg = field(default_factory=SkrlPolicyCfg)
    runner: SkrlRunnerCfg = field(default_factory=SkrlRunnerCfg)
