"""Lightweight ``@configclass`` decorator compatible with IsaacLab usage.

Wraps :func:`dataclasses.dataclass` with two ergonomic features lab users
rely on:

1. Mutable defaults (``list``, ``dict``, ``set``, other dataclass instances)
   declared directly as class-level assignments are auto-wrapped with
   ``field(default_factory=...)`` so they don't trip the "mutable default"
   check.
2. Every generated class gets ``to_dict()``, ``replace(**kwargs)`` and
   ``copy()`` helpers.

This is a stripped-down version of isaaclab's ``configclass`` — enough for
motlab's task / scene / manager term configs.
"""

from __future__ import annotations

import copy as _copy
import dataclasses as _dc
from typing import Any, Type, TypeVar

_T = TypeVar("_T")


def _is_mutable_default(value: Any) -> bool:
    """True if ``value`` is a mutable default that ``@dataclass`` rejects."""
    if isinstance(value, (list, dict, set)):
        return True
    if _dc.is_dataclass(value) and not isinstance(value, type):
        return True
    return False


def _make_factory(value: Any):
    # Bind ``value`` via default arg so the closure captures the snapshot.
    def factory(_v=value):
        return _copy.deepcopy(_v)

    return factory


def configclass(cls: Type[_T] | None = None, /, **dataclass_kwargs) -> Type[_T]:
    """Decorator: like ``@dataclass`` but tolerates mutable defaults."""

    def _wrap(c: Type[_T]) -> Type[_T]:
        annotations = getattr(c, "__annotations__", {})
        for name in annotations:
            if name in c.__dict__:
                val = c.__dict__[name]
                if isinstance(val, _dc.Field):
                    continue
                if _is_mutable_default(val):
                    setattr(c, name, _dc.field(default_factory=_make_factory(val)))

        built = _dc.dataclass(**dataclass_kwargs)(c)

        def _to_dict(self) -> dict:
            return _dc.asdict(self)

        def _replace(self, **kwargs):
            return _dc.replace(self, **kwargs)

        def _copy_self(self):
            return _copy.deepcopy(self)

        built.to_dict = _to_dict
        built.replace = _replace
        built.copy = _copy_self
        return built

    if cls is None:
        return _wrap  # type: ignore[return-value]
    return _wrap(cls)
