"""Regex name-resolution helpers — matches IsaacLab's utils.string API."""

from __future__ import annotations

import re
from typing import Sequence


def resolve_matching_names(
    keys: str | Sequence[str],
    names: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Return ``(indices, matched_names)`` for ``names`` matching ``keys`` regexes.

    ``keys`` may be a single pattern or a sequence; each is treated as a
    full-string regex. When ``preserve_order=True`` the output order follows
    the order of ``keys`` (falling back to order-of-appearance in ``names``
    within a key). Otherwise the output is in ``names`` order.
    """
    patterns: list[str] = [keys] if isinstance(keys, str) else list(keys)
    compiled = [re.compile(f"^{p}$") for p in patterns]

    if preserve_order:
        indices: list[int] = []
        matched: list[str] = []
        seen: set[int] = set()
        for pat in compiled:
            for i, n in enumerate(names):
                if pat.match(n) and i not in seen:
                    indices.append(i)
                    matched.append(n)
                    seen.add(i)
        return indices, matched

    hits: list[tuple[int, str]] = []
    for i, n in enumerate(names):
        if any(p.match(n) for p in compiled):
            hits.append((i, n))
    return [i for i, _ in hits], [n for _, n in hits]


def resolve_matching_names_values(
    data: dict[str, float | int] | float | int,
    names: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str], list[float]]:
    """Resolve a ``{regex: value}`` dict (or scalar) against ``names``.

    Returns ``(indices, matched_names, values)`` where ``values[i]`` is the
    value associated with each matched name. When ``data`` is a scalar it's
    broadcast to every name.
    """
    if isinstance(data, (int, float)):
        indices = list(range(len(names)))
        return indices, list(names), [float(data)] * len(names)

    indices: list[int] = []
    matched: list[str] = []
    values: list[float] = []
    seen: set[int] = set()
    for pattern, value in data.items():
        pat = re.compile(f"^{pattern}$")
        for i, n in enumerate(names):
            if pat.match(n) and i not in seen:
                indices.append(i)
                matched.append(n)
                values.append(float(value))
                seen.add(i)
    if not preserve_order:
        order = sorted(range(len(indices)), key=lambda k: indices[k])
        indices = [indices[k] for k in order]
        matched = [matched[k] for k in order]
        values = [values[k] for k in order]
    return indices, matched, values
