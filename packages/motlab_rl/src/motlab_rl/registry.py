"""Per-env default RL configurations.

Tasks register one cfg class per ``(rl_framework, backend)`` pair. The RL
framework is inferred from the cfg class's parent class so decorator calls
stay short.

    @rlcfg("cartpole", backend="torch")
    @dataclass
    class CartPoleSkrlTorchCfg(SkrlCfg):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from motlab_envs import registry as env_registry

logger = logging.getLogger(__name__)

TRLCfg = TypeVar("TRLCfg")


@dataclass
class EnvRlCfgs:
    """{rl_framework: {backend_or_None: cfg_class}}."""

    cfgs: dict[str, dict[Optional[str], Type]] = field(default_factory=dict)


_rlcfgs: dict[str, EnvRlCfgs] = {}


def _infer_framework(cls: Type) -> str:
    # Lazy imports keep the envs-only import graph slim.
    from motlab_rl.rslrl.cfg import RslrlCfg
    from motlab_rl.skrl.config import SkrlCfg

    for base in cls.__mro__:
        if base is SkrlCfg:
            return "skrl"
        if base is RslrlCfg:
            return "rslrl"
    raise ValueError(
        f"Cannot infer RL framework from {cls.__name__}. Inherit from SkrlCfg or RslrlCfg."
    )


def _register(env_name: str, rllib: str, backend: Optional[str], cls: Type) -> None:
    if not env_registry.contains(env_name):
        raise ValueError(f"Env '{env_name}' is not registered in motlab_envs.")
    logger.info("Registering RL cfg for env=%s rllib=%s backend=%s", env_name, rllib, backend)
    _rlcfgs.setdefault(env_name, EnvRlCfgs()).cfgs.setdefault(rllib, {})[backend] = cls


def rlcfg(env_name: str, backend: Optional[str] = None) -> Callable[[Type[TRLCfg]], Type[TRLCfg]]:
    """Register ``cls`` as the default RL cfg for ``env_name`` (+ backend).

    If ``backend`` is None, registers for all backends of the inferred framework.
    """

    def decorator(cls: Type[TRLCfg]) -> Type[TRLCfg]:
        rllib = _infer_framework(cls)
        backends: list[Optional[str]] = [None] if backend is None else [backend]
        for b in backends:
            _register(env_name, rllib, b, cls)
        return cls

    return decorator


def default_rl_cfg(env_name: str, rllib: str, backend: str) -> Any:
    """Instantiate the default RL cfg for ``(env_name, rllib, backend)``.

    Falls back to the universal (``backend=None``) cfg when no backend-specific
    cfg is registered.
    """
    if env_name not in _rlcfgs:
        raise ValueError(f"No RL cfg registered for env '{env_name}'.")
    framework_cfgs = _rlcfgs[env_name].cfgs.get(rllib)
    if not framework_cfgs:
        raise ValueError(f"RL framework '{rllib}' not registered for env '{env_name}'.")
    if backend in framework_cfgs:
        return framework_cfgs[backend]()
    if None in framework_cfgs:
        return framework_cfgs[None]()
    raise ValueError(
        f"No cfg for env '{env_name}' rllib '{rllib}' backend '{backend}'."
    )
