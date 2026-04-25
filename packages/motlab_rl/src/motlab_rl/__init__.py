"""MotLab RL framework integrations (rsl_rl + skrl)."""

# Pulling in the env library + bundled tasks ensures every @envcfg /
# @rlcfg / @skrlcfg decorator has fired before users call default_*_cfg.
import motlab  # noqa: F401
import motlab_tasks  # noqa: F401
from motlab_rl import tasks  # noqa: F401  (registers RL hyperparameters)
from motlab_rl.registry import (  # noqa: F401
    default_rl_cfg,
    default_skrl_cfg,
    list_registered,
    rlcfg,
    skrlcfg,
)

__version__ = "0.2.0"
__all__ = [
    "rlcfg",
    "skrlcfg",
    "default_rl_cfg",
    "default_skrl_cfg",
    "list_registered",
]
