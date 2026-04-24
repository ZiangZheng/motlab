"""Shared helpers for RL entry points (device detection, cfg overrides, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceSupports:
    torch: bool = False
    torch_gpu: bool = False
    jax: bool = False
    jax_gpu: bool = False


def get_device_supports() -> DeviceSupports:
    supports = DeviceSupports()
    try:
        import torch  # noqa: F401

        supports.torch = True
        try:
            supports.torch_gpu = torch.cuda.is_available()
        except Exception:  # pragma: no cover
            supports.torch_gpu = False
    except ImportError:
        pass
    try:
        import jax  # noqa: F401

        supports.jax = True
        try:
            supports.jax_gpu = any(d.platform == "gpu" for d in jax.devices())
        except Exception:  # pragma: no cover
            supports.jax_gpu = False
    except ImportError:
        pass
    return supports


def apply_overrides(cfg_obj: Any, overrides: dict[str, Any]) -> Any:
    """Apply dotted-key overrides in-place. Silently ignored if key missing."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = cfg_obj
        for part in parts[:-1]:
            if not hasattr(target, part):
                return cfg_obj
            target = getattr(target, part)
        if hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)
    return cfg_obj
