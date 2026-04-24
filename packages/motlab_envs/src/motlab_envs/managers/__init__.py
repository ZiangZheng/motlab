"""Optional manager-lite helpers.

Managers are **opt-in**: simple tasks like CartPole can ignore them and
write obs/reward directly. Locomotion tasks with a dozen reward terms
benefit from the modular pattern.

Each manager owns a list of named "terms" (callables that take ``state``
and return a per-env ndarray). The manager aggregates them via a weighted
sum (rewards) or concatenation (observations).
"""

from motlab_envs.managers.cfg import (  # noqa: F401
    ActionsCfg,
    CommandTermCfg,
    CommandsCfgMB,
    ObservationTermCfg,
    ObservationsCfg,
    PDActionCfg,
    RewardTermCfg,
    RewardsCfg,
    TerminationTermCfg,
    TerminationsCfg,
)
from motlab_envs.managers.commands import VelocityCommandManager  # noqa: F401
from motlab_envs.managers.observation import ObservationManager, ObsTerm  # noqa: F401
from motlab_envs.managers.reward import RewardManager, RewardTerm  # noqa: F401
from motlab_envs.managers.termination import TerminationManager, TerminationTerm  # noqa: F401
