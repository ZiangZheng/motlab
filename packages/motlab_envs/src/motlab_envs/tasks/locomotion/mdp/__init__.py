"""Pure-function MDP library for locomotion tasks.

Each function takes ``(env, **params)`` where ``env`` is a
:class:`motlab_envs.manager_env.ManagerBasedTorchEnv` (or an env exposing
the same accessor surface) and returns a per-env ``torch.Tensor``.

Modelled after isaaclab's / mjlab's ``tasks/locomotion/velocity/mdp/``.
"""

from . import commands  # noqa: F401
from . import observations  # noqa: F401
from . import rewards  # noqa: F401
from . import terminations  # noqa: F401
