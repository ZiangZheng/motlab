"""MotrixSim adapter — the only file allowed to import ``motrixsim`` or ``numpy``.

MotrixSim returns and accepts numpy arrays; motlab's internal API is
torch-tensor end-to-end. Everything in this module converts between
those two worlds at zero or one copy, and nothing else in motlab
should do so.

Tensor-view accessors (``dof_pos_view`` / ``dof_vel_view`` /
``actuator_ctrls_view``) return a torch tensor that *shares memory*
with the underlying numpy buffer: reads see MotrixSim's latest step
output and in-place writes are picked up by the next physics step.
This zero-copy path only works when the tensor is on CPU; callers that
need the data on CUDA should ``.to(device)`` themselves (that forces
a copy).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    import motrixsim as _mtx
except ImportError as exc:  # pragma: no cover - install-time helper
    raise ImportError(
        "motrixsim is not installed. Install it from PyPI:\n"
        "  pip install motrixsim\n"
        "or run the bundled installer:  bash scripts/install.sh"
    ) from exc


SceneModel = _mtx.SceneModel
SceneData = _mtx.SceneData


# ---------------------------------------------------------------------------
# Model / data construction
# ---------------------------------------------------------------------------
def load_model(model_file: str, sim_dt: float | None = None) -> SceneModel:
    """Load an MJCF scene and optionally override the timestep."""
    model = _mtx.load_model(model_file)
    if sim_dt is not None:
        model.options.timestep = sim_dt
    return model


def make_batched_data(model: SceneModel, num_envs: int) -> SceneData:
    """Create batched scene data for ``num_envs`` parallel environments."""
    return _mtx.SceneData(model, batch=[num_envs])


def raw() -> Any:
    """Return the underlying ``motrixsim`` module (escape hatch)."""
    return _mtx


# ---------------------------------------------------------------------------
# np ↔ torch bridges
# ---------------------------------------------------------------------------
def dof_pos_view(data: SceneData) -> torch.Tensor:
    """Zero-copy CPU tensor view over ``data.dof_pos`` (layout depends on model)."""
    return torch.from_numpy(data.dof_pos)


def dof_vel_view(data: SceneData) -> torch.Tensor:
    """Zero-copy CPU tensor view over ``data.dof_vel``."""
    return torch.from_numpy(data.dof_vel)


def actuator_ctrls_view(data: SceneData) -> torch.Tensor:
    """Zero-copy CPU tensor view over ``data.actuator_ctrls``."""
    return torch.from_numpy(data.actuator_ctrls)


def set_dof_pos(data: SceneData, model: SceneModel, dof_pos: torch.Tensor) -> None:
    """Write ``dof_pos`` (torch) into ``data``. Copies through numpy."""
    data.set_dof_pos(_to_numpy_f32(dof_pos), model)


def set_dof_vel(data: SceneData, dof_vel: torch.Tensor) -> None:
    """Write ``dof_vel`` (torch) into ``data``."""
    data.set_dof_vel(_to_numpy_f32(dof_vel))


def set_actuator_ctrls(data: SceneData, ctrls: torch.Tensor) -> None:
    """Write actuator controls (torch) into ``data``."""
    data.actuator_ctrls = _to_numpy_f32(ctrls)


def init_dof_pos_tensor(model: SceneModel, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return ``model.compute_init_dof_pos()`` as a 1D torch tensor on ``device``."""
    arr = np.asarray(model.compute_init_dof_pos(), dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def physics_step(model: SceneModel, data: SceneData) -> None:
    """Advance the batched sim by one substep."""
    model.step(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_numpy_f32(tensor: torch.Tensor) -> np.ndarray:
    """Materialize a torch tensor as a contiguous float32 numpy array."""
    cpu = tensor.detach()
    if cpu.device.type != "cpu":
        cpu = cpu.cpu()
    if cpu.dtype != torch.float32:
        cpu = cpu.to(torch.float32)
    return np.ascontiguousarray(cpu.numpy())
