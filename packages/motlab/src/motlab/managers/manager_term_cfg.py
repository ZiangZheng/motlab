"""Term-level config dataclasses (one per manager kind)."""

from __future__ import annotations

from typing import Any, Callable, Type

from motlab.utils.configclass import configclass


# ---------------------------------------------------------------------------
# SceneEntityCfg — points a term at a specific asset + name slice
# ---------------------------------------------------------------------------
@configclass
class SceneEntityCfg:
    """Reference to an asset (joints / bodies) addressed by name regex."""

    name: str = "robot"
    joint_names: list[str] | str | None = None
    joint_ids: list[int] | None = None
    body_names: list[str] | str | None = None
    body_ids: list[int] | None = None
    preserve_order: bool = False

    def resolve(self, scene) -> None:
        """Populate ``joint_ids`` / ``body_ids`` from name regex against the
        asset in ``scene[self.name]``."""
        from motlab.scene.interactive_scene import InteractiveScene

        assert isinstance(scene, InteractiveScene)
        asset = scene[self.name]
        if self.joint_names is not None and self.joint_ids is None:
            ids, _ = asset.find_joints(self.joint_names, preserve_order=self.preserve_order)
            self.joint_ids = ids
        if self.body_names is not None and self.body_ids is None:
            ids, _ = asset.find_bodies(self.body_names, preserve_order=self.preserve_order)
            self.body_ids = ids


# ---------------------------------------------------------------------------
# Term cfgs (shared params kwarg-only)
# ---------------------------------------------------------------------------
@configclass
class ActionTermCfg:
    class_type: Type | None = None
    asset_name: str = "robot"
    joint_names: list[str] | str = ".*"
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    use_default_offset: bool = True
    preserve_order: bool = False


@configclass
class ObservationTermCfg:
    func: Callable[..., Any] | None = None
    params: dict = {}
    noise: Any = None  # callable that takes (env, tensor) and returns tensor
    clip: tuple[float, float] | None = None
    scale: float | None = None


@configclass
class ObservationGroupCfg:
    """Container for observation terms.  Field names = term names."""

    enable_corruption: bool = True
    concatenate_terms: bool = True
    history_length: int = 0


@configclass
class RewardTermCfg:
    func: Callable[..., Any] | None = None
    weight: float = 1.0
    params: dict = {}


@configclass
class TerminationTermCfg:
    func: Callable[..., Any] | None = None
    params: dict = {}
    time_out: bool = False  # True → goes into ``truncated``


@configclass
class EventTermCfg:
    func: Callable[..., Any] | None = None
    mode: str = "reset"  # one of {"startup", "reset", "interval"}
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    params: dict = {}


@configclass
class CommandTermCfg:
    class_type: Type | None = None
    resampling_time_range: tuple[float, float] = (5.0, 5.0)
    debug_vis: bool = False
    asset_name: str = "robot"


@configclass
class CurriculumTermCfg:
    func: Callable[..., Any] | None = None
    params: dict = {}
