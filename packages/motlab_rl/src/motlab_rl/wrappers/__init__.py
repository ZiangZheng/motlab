"""Adapters turning a MotLab manager-based env into RL-library shapes."""

from motlab_rl.wrappers.rslrl import RslrlVecEnv  # noqa: F401

__all__ = ["RslrlVecEnv"]


def _import_skrl_wrapper():  # pragma: no cover
    """Lazy import — only when skrl is installed."""
    from motlab_rl.wrappers.skrl import SkrlVecEnv  # noqa: F401

    return SkrlVecEnv
