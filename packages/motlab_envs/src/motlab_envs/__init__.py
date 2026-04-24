"""MotLab environment library."""

from motlab_envs import registry  # noqa: F401
from motlab_envs.base import ABEnv, EnvCfg  # noqa: F401
from motlab_envs.env import TensorEnvState, TorchEnv  # noqa: F401
from motlab_envs.manager_env import ManagerBasedEnvCfg, ManagerBasedTorchEnv  # noqa: F401

# Import built-in task packages so their decorators register on import.
# Wrapped in try/except so the package remains importable (e.g. for the
# pure-Python tests) even when the simulation engine isn't installed yet.
try:
    from motlab_envs.tasks import basic  # noqa: F401
except ImportError:
    import logging

    logging.getLogger(__name__).warning(
        "Could not auto-import built-in tasks (engine probably missing). "
        "Install motrixsim to enable tasks."
    )

__version__ = "0.1.0"
