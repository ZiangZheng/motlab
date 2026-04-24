"""Dataclass term-configs for the manager-based workflow.

These are *configuration* objects: the user declares reward / obs /
termination / command terms as named entries whose ``func`` is an MDP
function taking ``(env, **params) -> torch.Tensor``.

The ``ManagerBasedTorchEnv`` base class reads these configs at env init
and hydrates the lite managers in :mod:`motlab_envs.managers` with the
resulting callable+weight pairs.

Design is the same as isaaclab / mjlab / genesislab — ``func`` is any
importable callable, ``params`` is a ``dict`` forwarded as kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

TermFunc = Callable[..., Any]  # signature (env, **params) -> torch.Tensor


@dataclass
class RewardTermCfg:
    """A reward term — scalar weight × dt × ``func(env, **params)``.

    The ``scale_by_dt`` flag mirrors isaaclab: enabling it converts the
    weight from "per-second" to "per-step". Default is off to match
    MotrixLab's bare weighted-sum convention.
    """

    func: TermFunc
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    scale_by_dt: bool = False


@dataclass
class ObservationTermCfg:
    """An observation term — concatenated into the final vector."""

    func: TermFunc
    scale: float = 1.0
    noise_scale: float = 0.0
    clip: Optional[tuple[float, float]] = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminationTermCfg:
    """A termination predicate — combined via logical OR."""

    func: TermFunc
    params: dict[str, Any] = field(default_factory=dict)
    # If True, the term represents a real failure (terminated). Keep
    # time-based truncation out of here; TorchEnv handles it via
    # ``max_episode_seconds``.
    terminal: bool = True


@dataclass
class CommandTermCfg:
    """A command generator (e.g. velocity target)."""

    func: TermFunc  # should accept (env, **params) -> torch.Tensor (num_envs, D)
    params: dict[str, Any] = field(default_factory=dict)
    resample_seconds: float = 10.0


# ---------------------------------------------------------------------------
# Grouping dataclasses — tasks assemble these into their EnvCfg.
# Fields are plain ``dict[str, TermCfg]`` so tasks can use any names and
# override specific entries with ``replace()``.
# ---------------------------------------------------------------------------
@dataclass
class RewardsCfg:
    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    clip: Optional[tuple[float, float]] = None
    zero_on_termination: bool = True


@dataclass
class ObservationsCfg:
    """A single observation group (e.g. ``policy`` or ``critic``).

    Tasks that need privileged critic obs can instantiate two separate
    ``ObservationsCfg`` instances and expose them as different fields.
    """

    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)


@dataclass
class TerminationsCfg:
    terms: dict[str, TerminationTermCfg] = field(default_factory=dict)


@dataclass
class CommandsCfgMB:
    """Named commands (e.g. ``twist`` → (vx, vy, wz))."""

    terms: dict[str, CommandTermCfg] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action-space configuration — tells the base env how to map raw policy
# outputs to actuator torques. For now we only support PD on a single joint
# group; extend with more action classes as new robots are added.
# ---------------------------------------------------------------------------
@dataclass
class PDActionCfg:
    """Map raw policy outputs → PD target angles → torques."""

    joint_names: tuple[str, ...] = ()
    default_angles: Optional[Any] = None  # torch.Tensor / sequence of shape (num_joints,)
    stiffness: Optional[Any] = None       # scalar or (num_joints,)
    damping: Optional[Any] = None
    action_scale: float = 1.0
    torque_limit: Optional[float] = None
    latency_steps: int = 0
    action_noise_scale: float = 0.0


@dataclass
class ActionsCfg:
    joint_pd: PDActionCfg = field(default_factory=PDActionCfg)
