"""Per-env default RL cfg registry (rsl_rl only)."""

from __future__ import annotations

from typing import Callable, Type, TypeVar

from motlab_rl.rslrl.cfg import RslrlCfg

T = TypeVar("T", bound=RslrlCfg)

_REGISTRY: dict[str, Type[RslrlCfg]] = {}


def rlcfg(env_name: str) -> Callable[[Type[T]], Type[T]]:
    """Register a cfg class as the default RL config for ``env_name``."""

    def _wrap(cls: Type[T]) -> Type[T]:
        _REGISTRY[env_name] = cls
        return cls

    return _wrap


def list_registered() -> list[str]:
    return sorted(_REGISTRY)


def default_rl_cfg(env_name: str) -> RslrlCfg:
    if env_name not in _REGISTRY:
        raise KeyError(f"No RL cfg registered for env {env_name!r}")
    return _REGISTRY[env_name]()
