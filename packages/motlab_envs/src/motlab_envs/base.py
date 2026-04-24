"""Abstract env + structured config dataclasses for MotLab."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym


# ---------------------------------------------------------------------------
# Structured sub-configs. Tasks compose these into a top-level `EnvCfg`
# subclass. Each sub-config is optional — omit what you don't need.
# ---------------------------------------------------------------------------
@dataclass
class AssetCfg:
    """Robot / scene asset descriptors."""

    model_file: str = ""
    body_name: str = ""
    foot_name: str = ""
    ground_name: str = ""
    terminate_after_contacts_on: tuple[str, ...] = ()


@dataclass
class ControlCfg:
    """Low-level controller parameters (PD by default)."""

    stiffness: float = 0.0
    damping: float = 0.0
    action_scale: float = 1.0


@dataclass
class InitStateCfg:
    """Default pose / joint angles at reset."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_joint_angles: dict[str, float] = field(default_factory=dict)
    reset_noise_scale: float = 0.0


@dataclass
class CommandsCfg:
    """External command (velocity tracking, pose tracking, ...) ranges."""

    resample_seconds: float = 10.0
    # xy lin-vel + yaw rate by convention; override `shape` per task.
    vel_limit: tuple[tuple[float, float], tuple[float, float]] = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    )


@dataclass
class NormalizationCfg:
    """Observation scaling factors."""

    lin_vel: float = 1.0
    ang_vel: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 1.0


@dataclass
class RewardCfg:
    """Per-term reward weights (+ misc shaping parameters)."""

    scales: dict[str, float] = field(default_factory=dict)
    tracking_sigma: float = 0.25
    clip_reward: Optional[tuple[float, float]] = None


@dataclass
class SensorCfg:
    """Names of sim sensors used for observations."""

    local_linvel: str = ""
    gyro: str = ""


@dataclass
class DomainRandCfg:
    """Domain randomization ranges. Managers / tasks read this on reset."""

    push_interval_s: Optional[float] = None
    push_vel: tuple[float, float] = (0.0, 0.0)
    obs_noise_scale: float = 0.0
    action_noise_scale: float = 0.0
    action_latency_steps: int = 0
    mass_range: tuple[float, float] = (1.0, 1.0)
    friction_range: tuple[float, float] = (1.0, 1.0)


# ---------------------------------------------------------------------------
# Top-level env cfg. Tasks extend this and add their own sub-cfgs or fields.
# ---------------------------------------------------------------------------
@dataclass
class EnvCfg:
    """Base environment configuration."""

    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    max_episode_seconds: Optional[float] = None
    render_spacing: float = 1.0

    # Torch device for all tensor state. MotrixSim itself runs on CPU, so
    # ``"cpu"`` is the zero-copy default; set to ``"cuda"`` to keep the
    # policy's obs/reward tensors on GPU (at the cost of a per-step copy
    # across the engine boundary).
    device: str = "cpu"

    # Convenience flag — tasks may ignore.
    asset: AssetCfg = field(default_factory=AssetCfg)

    @property
    def max_episode_steps(self) -> Optional[int]:
        if self.max_episode_seconds is None:
            return None
        return int(self.max_episode_seconds / self.ctrl_dt)

    @property
    def sim_substeps(self) -> int:
        return int(round(self.ctrl_dt / self.sim_dt))

    @property
    def model_file(self) -> str:
        """Shortcut used by the env to load the scene model."""
        return self.asset.model_file

    def validate(self) -> None:
        if self.sim_dt <= 0 or self.ctrl_dt <= 0:
            raise ValueError("sim_dt and ctrl_dt must be positive")
        if self.sim_dt > self.ctrl_dt:
            raise ValueError("sim_dt must be <= ctrl_dt")


# ---------------------------------------------------------------------------
# Abstract env interface.
# ---------------------------------------------------------------------------
class ABEnv(abc.ABC):
    """Batched RL env interface. `num_envs` copies run in lockstep."""

    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def cfg(self) -> EnvCfg:
        ...

    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space:
        ...

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        ...
