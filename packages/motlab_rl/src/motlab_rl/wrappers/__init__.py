"""Adapters turning a MotLab manager-based env into RL-library shapes."""

from motlab_rl.wrappers.rslrl import RslrlVecEnv  # noqa: F401

__all__ = ["RslrlVecEnv"]
