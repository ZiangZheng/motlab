"""Observation manager: concatenate named obs terms with optional noise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

ObsFn = Callable[[Any], torch.Tensor]


@dataclass
class ObsTerm:
    name: str
    fn: ObsFn
    scale: float = 1.0
    noise_scale: float = 0.0
    clip: Optional[tuple[float, float]] = None


@dataclass
class ObservationManager:
    """Concat named obs terms along dim=-1.

    Each term receives the same ``ctx`` (the env) and must return a
    ``(num_envs, dim)`` tensor. The manager applies per-term scaling,
    additive Gaussian noise, and clipping, then concatenates.
    """

    terms: list[ObsTerm] = field(default_factory=list)
    generator: Optional[torch.Generator] = None

    def add(
        self,
        name: str,
        fn: ObsFn,
        scale: float = 1.0,
        noise_scale: float = 0.0,
        clip: Optional[tuple[float, float]] = None,
    ) -> "ObservationManager":
        self.terms.append(ObsTerm(name=name, fn=fn, scale=scale, noise_scale=noise_scale, clip=clip))
        return self

    @property
    def num_terms(self) -> int:
        return len(self.terms)

    def compute(self, ctx: Any) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for term in self.terms:
            value = term.fn(ctx).to(torch.float32) * term.scale
            if term.noise_scale > 0:
                noise = torch.randn(
                    value.shape, device=value.device, dtype=value.dtype, generator=self.generator
                ) * term.noise_scale
                value = value + noise
            if term.clip is not None:
                value = torch.clamp(value, term.clip[0], term.clip[1])
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def dim(self, probe_ctx: Any) -> int:
        """Run each term once on ``probe_ctx`` to infer concatenated dim."""
        return int(sum(term.fn(probe_ctx).shape[-1] for term in self.terms))
