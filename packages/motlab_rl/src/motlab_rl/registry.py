"""Per-env default RL cfg registries.

Two parallel registries keyed by env name:

- :func:`rlcfg` / :func:`default_rl_cfg` for **rsl_rl** PPO cfgs.
- :func:`skrlcfg` / :func:`default_skrl_cfg` for **skrl** PPO cfgs.

A given env may register one, the other, or both.
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

from motlab_rl.rslrl.cfg import RslrlCfg
from motlab_rl.skrl.cfg import SkrlCfg

T_RSL = TypeVar("T_RSL", bound=RslrlCfg)
T_SKRL = TypeVar("T_SKRL", bound=SkrlCfg)

_REGISTRY: dict[str, Type[RslrlCfg]] = {}
_SKRL_REGISTRY: dict[str, Type[SkrlCfg]] = {}


def rlcfg(env_name: str) -> Callable[[Type[T_RSL]], Type[T_RSL]]:
    """Register a cfg class as the default rsl_rl config for ``env_name``."""

    def _wrap(cls: Type[T_RSL]) -> Type[T_RSL]:
        _REGISTRY[env_name] = cls
        return cls

    return _wrap


def skrlcfg(env_name: str) -> Callable[[Type[T_SKRL]], Type[T_SKRL]]:
    """Register a cfg class as the default skrl config for ``env_name``."""

    def _wrap(cls: Type[T_SKRL]) -> Type[T_SKRL]:
        _SKRL_REGISTRY[env_name] = cls
        return cls

    return _wrap


def list_registered(framework: str = "rslrl") -> list[str]:
    if framework == "rslrl":
        return sorted(_REGISTRY)
    if framework == "skrl":
        return sorted(_SKRL_REGISTRY)
    raise ValueError(f"Unknown framework {framework!r}; expected 'rslrl' or 'skrl'")


def default_rl_cfg(env_name: str) -> RslrlCfg:
    if env_name not in _REGISTRY:
        raise KeyError(f"No rsl_rl cfg registered for env {env_name!r}")
    return _REGISTRY[env_name]()


def default_skrl_cfg(env_name: str) -> SkrlCfg:
    if env_name not in _SKRL_REGISTRY:
        raise KeyError(f"No skrl cfg registered for env {env_name!r}")
    return _SKRL_REGISTRY[env_name]()
