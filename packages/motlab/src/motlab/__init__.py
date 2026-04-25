"""motlab — torch-native RL environments on top of MotrixSim.

Importing this package gives you the framework (managers, envs, MDP
function library, registry). Robot configs live in ``motlab_assets`` and
ready-made tasks in ``motlab_tasks`` — import those packages to fire their
``@envcfg`` registrations before calling :func:`make_cfg` / :func:`make`.
"""

from motlab import mdp  # noqa: F401
from motlab.envs.manager_based_env import ManagerBasedEnv
from motlab.envs.manager_based_env_cfg import ManagerBasedEnvCfg
from motlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from motlab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from motlab.registry import (
    envcfg,
    list_envs,
    make,
    make_cfg,
    register,
)

__all__ = [
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "ManagerBasedRLEnv",
    "ManagerBasedRLEnvCfg",
    "envcfg",
    "list_envs",
    "make",
    "make_cfg",
    "register",
]
