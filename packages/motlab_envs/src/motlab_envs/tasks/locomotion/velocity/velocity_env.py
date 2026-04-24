"""Concrete manager-based locomotion env.

Thin subclass of :class:`ManagerBasedTorchEnv` that hard-codes the
floating-base DOF layout expected by the declarative velocity-tracking
task configs.
"""

from __future__ import annotations

from motlab_envs.manager_env import ManagerBasedTorchEnv


class LocomotionVelocityEnv(ManagerBasedTorchEnv):
    """Locomotion env with a free-joint base (7 pos / 6 vel)."""

    _freejoint_pos_dim = 7  # xyz + quat_xyzw
    _freejoint_vel_dim = 6  # lin + ang
