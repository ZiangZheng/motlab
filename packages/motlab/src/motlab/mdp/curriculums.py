"""Curriculum functions — return scalar progress for logging or None."""

from __future__ import annotations


def constant(env, env_ids=None, value: float = 0.0) -> float:
    return float(value)
