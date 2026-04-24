"""Reward manager: compose per-env rewards from named weighted terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

RewardFn = Callable[[Any], torch.Tensor]


@dataclass
class RewardTerm:
    name: str
    fn: RewardFn
    weight: float = 1.0
    # Optional curriculum multiplier applied on top of ``weight``. Tasks can
    # mutate this over training (e.g. ramp up tracking accuracy).
    curriculum_scale: float = 1.0


@dataclass
class RewardManager:
    """Sum of weighted reward terms, optionally clipped and zeroed on termination.

    Every term receives the same object (usually the env) and must return
    shape ``(num_envs,)``.
    """

    terms: list[RewardTerm] = field(default_factory=list)
    clip: Optional[tuple[float, float]] = None
    zero_on_termination: bool = True

    def add(self, name: str, fn: RewardFn, weight: float = 1.0) -> "RewardManager":
        self.terms.append(RewardTerm(name=name, fn=fn, weight=weight))
        return self

    def compute(
        self,
        ctx: Any,
        terminated: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(reward, per_term_dict)`` with post-weight contributions."""
        total: torch.Tensor | None = None
        per_term: dict[str, torch.Tensor] = {}
        for term in self.terms:
            raw = term.fn(ctx)
            scaled = raw * (term.weight * term.curriculum_scale)
            per_term[term.name] = scaled
            total = scaled if total is None else total + scaled

        assert total is not None, "RewardManager has no terms"

        if self.clip is not None:
            total = torch.clamp(total, self.clip[0], self.clip[1])

        if self.zero_on_termination and terminated is not None:
            total = torch.where(terminated, torch.zeros_like(total), total)

        return total, per_term

    def set_curriculum(self, name: str, scale: float) -> None:
        for term in self.terms:
            if term.name == name:
                term.curriculum_scale = scale
                return
        raise KeyError(f"No reward term named {name!r}")
