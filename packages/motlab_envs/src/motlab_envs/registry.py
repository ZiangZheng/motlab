"""Decorator-based registry for env configs and env classes.

Two-step registration:
    1. `@envcfg("my-task")` on a dataclass subclass of :class:`EnvCfg`.
    2. `@env("my-task", sim_backend="torch")` on the env implementation class.

Instantiate with :func:`make`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from motlab_envs.base import ABEnv, EnvCfg

TEnvCfg = TypeVar("TEnvCfg", bound=EnvCfg)
TEnv = TypeVar("TEnv", bound=ABEnv)

# Supported simulation backends. motlab is torch-tensor end-to-end; the
# backend slot is kept only to leave room for future non-MotrixSim engines.
_SUPPORTED_BACKENDS = frozenset({"torch"})


@dataclass
class EnvMeta:
    env_cfg_cls: Type[EnvCfg]
    env_cls_dict: dict[str, Type[ABEnv]] = field(default_factory=dict)

    def available_sim_backend(self) -> Optional[str]:
        return next(iter(self.env_cls_dict), None)

    def support_sim_backend(self, sim_backend: str) -> bool:
        return sim_backend in self.env_cls_dict


_envs: dict[str, EnvMeta] = {}


def contains(name: str) -> bool:
    return name in _envs


def register_env_config(name: str, env_cfg_cls: Type[EnvCfg]) -> None:
    if name in _envs:
        raise ValueError(f"Env '{name}' is already registered.")
    _envs[name] = EnvMeta(env_cfg_cls=env_cfg_cls)


def envcfg(name: str) -> Callable[[Type[TEnvCfg]], Type[TEnvCfg]]:
    def decorator(cls: Type[TEnvCfg]) -> Type[TEnvCfg]:
        register_env_config(name, cls)
        return cls

    return decorator


def register_env(name: str, env_cls: Type[ABEnv], sim_backend: str) -> None:
    if sim_backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported sim backend '{sim_backend}'.")
    if name not in _envs:
        raise ValueError(f"Env '{name}' has no cfg registered; call envcfg first.")
    if sim_backend in _envs[name].env_cls_dict:
        raise ValueError(f"Env '{name}' already has backend '{sim_backend}' registered.")
    _envs[name].env_cls_dict[sim_backend] = env_cls


def env(name: str, sim_backend: str = "torch") -> Callable[[Type[TEnv]], Type[TEnv]]:
    def decorator(cls: Type[TEnv]) -> Type[TEnv]:
        register_env(name, cls, sim_backend)
        return cls

    return decorator


def find_available_sim_backend(env_name: str) -> str:
    if env_name not in _envs:
        raise ValueError(f"Env '{env_name}' is not registered.")
    backend = _envs[env_name].available_sim_backend()
    if backend is None:
        raise ValueError(f"Env '{env_name}' has no implementation registered.")
    return backend


def make(
    name: str,
    sim_backend: Optional[str] = None,
    env_cfg_override: Optional[dict[str, Any]] = None,
    num_envs: int = 1,
) -> ABEnv:
    """Instantiate a registered env.

    `env_cfg_override` supports dotted keys like ``"control.stiffness"`` to
    reach nested sub-configs.
    """
    if name not in _envs:
        raise ValueError(f"Env '{name}' is not registered.")

    meta = _envs[name]
    env_cfg = meta.env_cfg_cls()

    if env_cfg_override:
        for key, value in env_cfg_override.items():
            _set_nested(env_cfg, key, value)

    env_cfg.validate()

    backend = sim_backend or meta.available_sim_backend()
    if backend is None:
        raise ValueError(f"Env '{name}' has no backends registered.")
    if not meta.support_sim_backend(backend):
        raise ValueError(f"Env '{name}' does not support backend '{backend}'.")

    return meta.env_cls_dict[backend](env_cfg, num_envs=num_envs)


def _set_nested(obj: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = obj
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"{type(obj).__name__} has no attribute path '{dotted_key}'")
        target = getattr(target, part)
    last = parts[-1]
    if not hasattr(target, last):
        raise ValueError(f"{type(target).__name__} has no attribute '{last}' (from '{dotted_key}')")
    setattr(target, last, value)


def list_registered_envs() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "config_class": meta.env_cfg_cls.__name__,
            "available_backends": list(meta.env_cls_dict.keys()),
        }
        for name, meta in _envs.items()
    }
