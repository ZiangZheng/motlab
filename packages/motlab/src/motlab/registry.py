"""Decorator-based registry for env cfgs and env classes.

Tasks call :func:`envcfg("name")` on their cfg class to register the cfg
factory; the env class is implicit (always :class:`ManagerBasedRLEnv`
unless overridden).
"""

from __future__ import annotations

from typing import Callable, Type

from motlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from motlab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


_REGISTRY: dict[str, dict] = {}


def envcfg(name: str, *, env_class: Type = ManagerBasedRLEnv) -> Callable:
    """Decorator: register a cfg class under ``name``."""

    def _wrap(cls: Type[ManagerBasedRLEnvCfg]) -> Type[ManagerBasedRLEnvCfg]:
        _REGISTRY[name] = {"cfg_class": cls, "env_class": env_class}
        return cls

    return _wrap


def register(name: str, cfg_class: Type, env_class: Type = ManagerBasedRLEnv) -> None:
    _REGISTRY[name] = {"cfg_class": cfg_class, "env_class": env_class}


def list_envs() -> list[str]:
    return sorted(_REGISTRY.keys())


def make_cfg(name: str, **overrides) -> ManagerBasedRLEnvCfg:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown env {name!r}. Registered: {list_envs()}")
    cfg = _REGISTRY[name]["cfg_class"]()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make(name: str, *, device: str | None = None, **overrides) -> ManagerBasedRLEnv:
    cfg = make_cfg(name, **overrides)
    env_class = _REGISTRY[name]["env_class"]
    return env_class(cfg, device=device)
