"""Termination manager: logical-OR named predicates; distinguishes from timeout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

TermFn = Callable[[Any], torch.Tensor]


@dataclass
class TerminationTerm:
    name: str
    fn: TermFn
    # True when the term represents real failure (not a timeout). Timeouts
    # are handled via ``EnvCfg.max_episode_seconds`` / ``truncated``.
    terminal: bool = True


@dataclass
class TerminationManager:
    terms: list[TerminationTerm] = field(default_factory=list)

    def add(self, name: str, fn: TermFn, terminal: bool = True) -> "TerminationManager":
        self.terms.append(TerminationTerm(name=name, fn=fn, terminal=terminal))
        return self

    def compute(self, ctx: Any) -> tuple[torch.Tensor, dict]:
        """Return ``(done, per_term_dict)`` where ``done`` is the logical OR."""
        per_term: dict[str, torch.Tensor] = {}
        combined: torch.Tensor | None = None
        for term in self.terms:
            value = term.fn(ctx).to(torch.bool)
            per_term[term.name] = value
            combined = value if combined is None else combined | value
        if combined is None:
            raise RuntimeError("TerminationManager has no terms")
        return combined, per_term
