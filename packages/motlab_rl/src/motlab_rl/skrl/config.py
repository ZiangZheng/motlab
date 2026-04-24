"""Shared SKRL config schema (framework-level base class)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkrlRunnerCfg:
    seed: Optional[int] = 42
    timesteps: int = 100_000
    rollouts: int = 24
    learning_epochs: int = 5
    mini_batches: int = 4
    discount_factor: float = 0.99
    lambda_: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_loss_scale: float = 0.01


@dataclass
class SkrlCfg:
    """Base class for SKRL env-specific cfgs. Subclass in `tasks/`."""

    num_envs: int = 1024
    rollouts: int = 24
    runner: SkrlRunnerCfg = field(default_factory=SkrlRunnerCfg)
    experiment_name: str = "motlab-skrl"
