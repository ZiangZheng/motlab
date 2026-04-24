"""Reward-shaping helpers (tolerance / sigmoid primitives).

Mirrors the DeepMind Control Suite ``tolerance`` family so reward terms
are interchangeable with dm_control-style specs. All inputs/outputs are
torch tensors.
"""

from __future__ import annotations

import math

import torch

_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x: torch.Tensor, value_at_1: float, sigmoid: str) -> torch.Tensor:
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(f"value_at_1 must be in [0, 1) for '{sigmoid}', got {value_at_1}.")
    elif not 0 < value_at_1 < 1:
        raise ValueError(f"value_at_1 must be in (0, 1) for '{sigmoid}', got {value_at_1}.")

    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)
    if sigmoid == "hyperbolic":
        scale = math.acosh(1.0 / value_at_1)
        return 1.0 / torch.cosh(x * scale)
    if sigmoid == "long_tail":
        scale = math.sqrt(1.0 / value_at_1 - 1.0)
        return 1.0 / ((x * scale) ** 2 + 1.0)
    if sigmoid == "reciprocal":
        scale = 1.0 / value_at_1 - 1.0
        return 1.0 / (torch.abs(x) * scale + 1.0)
    if sigmoid == "linear":
        scale = 1.0 - value_at_1
        scaled = x * scale
        return torch.where(torch.abs(scaled) < 1, 1.0 - scaled, torch.zeros_like(scaled))
    if sigmoid == "quadratic":
        scale = math.sqrt(1.0 - value_at_1)
        scaled = x * scale
        return torch.where(torch.abs(scaled) < 1, 1.0 - scaled**2, torch.zeros_like(scaled))
    if sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1.0 - value_at_1))
        return 1.0 - torch.tanh(x * scale) ** 2
    raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> torch.Tensor:
    """Return 1.0 where ``x`` is within ``bounds``, decaying over ``margin``."""
    lower, upper = bounds
    if lower > upper:
        raise ValueError("tolerance: lower bound must be <= upper bound")
    if margin < 0:
        raise ValueError("tolerance: margin must be non-negative")

    in_bounds = (lower <= x) & (x <= upper)
    if margin == 0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

    d = torch.where(x < lower, lower - x, x - upper) / margin
    return torch.where(in_bounds, torch.ones_like(x), _sigmoids(d, value_at_margin, sigmoid))


def squared_error(value: torch.Tensor) -> torch.Tensor:
    """Sum-of-squares along the last axis."""
    return torch.sum(value * value, dim=-1)


def exp_neg(value: torch.Tensor, sigma: float) -> torch.Tensor:
    """``exp(-value / sigma)`` — common tracking-reward primitive."""
    return torch.exp(-value / sigma)
