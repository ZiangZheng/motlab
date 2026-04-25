"""MotLab RL framework integrations (rsl_rl)."""

# Pulling in the env library + bundled tasks ensures every @envcfg /
# @rlcfg decorator has fired before users call default_rl_cfg / make_cfg.
import motlab  # noqa: F401
import motlab_tasks  # noqa: F401
from motlab_rl import tasks  # noqa: F401  (registers RL hyperparameters)
from motlab_rl.registry import default_rl_cfg, list_registered, rlcfg  # noqa: F401

__version__ = "0.2.0"
__all__ = ["rlcfg", "default_rl_cfg", "list_registered"]
