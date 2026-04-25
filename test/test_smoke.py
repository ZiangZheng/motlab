"""Smoke tests that do not require motrixsim.

Covers pure-Python pieces (configclass plumbing, registry, term cfgs, math
helpers). End-to-end env tests require MotrixSim and live elsewhere.
"""

from __future__ import annotations

import pytest
import torch


def test_configclass_mutable_defaults():
    """@configclass auto-wraps mutable defaults so each instance is independent."""
    from motlab.utils.configclass import configclass

    @configclass
    class Inner:
        x: float = 1.0

    @configclass
    class Outer:
        inner: Inner = Inner()
        items: list[int] = [1, 2, 3]

    a = Outer()
    b = Outer()
    a.inner.x = 99.0
    a.items.append(4)
    assert b.inner.x == 1.0
    assert b.items == [1, 2, 3]


def test_registry_lists_builtin_envs():
    """Importing motlab_tasks should register cartpole + go1-velocity in motlab."""
    import motlab
    import motlab_tasks  # noqa: F401

    envs = motlab.list_envs()
    assert "cartpole" in envs
    assert "go1-velocity" in envs


def test_make_cfg_returns_rl_cfg():
    import motlab
    import motlab_tasks  # noqa: F401
    from motlab import ManagerBasedRLEnvCfg

    cfg = motlab.make_cfg("cartpole")
    assert isinstance(cfg, ManagerBasedRLEnvCfg)
    assert cfg.scene is not None
    assert cfg.observations is not None
    assert cfg.actions is not None
    assert cfg.rewards is not None
    assert cfg.terminations is not None


def test_observation_term_cfg_fields():
    """Term cfgs must accept field-style overrides without dataclass errors."""
    from motlab.managers.manager_term_cfg import (
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )

    obs = ObservationTermCfg(func=lambda env: env, scale=2.0)
    assert obs.scale == 2.0
    rew = RewardTermCfg(func=lambda env: env, weight=-0.5)
    assert rew.weight == -0.5
    term = TerminationTermCfg(func=lambda env: env, time_out=True)
    assert term.time_out is True


def test_quat_round_trip():
    from motlab.utils.math import quat_apply, quat_conjugate, quat_mul

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    out = quat_apply(q, v)
    torch.testing.assert_close(out, v)

    q2 = torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]])  # 90° about x
    inv = quat_conjugate(q2)
    eye = quat_mul(q2, inv)
    torch.testing.assert_close(eye[..., 0].abs(), torch.ones(1), atol=1e-5, rtol=1e-5)


def test_rl_default_cfg_lookup():
    import motlab_rl
    from motlab_rl.rslrl.cfg import RslrlCfg

    cfg = motlab_rl.default_rl_cfg("cartpole")
    assert isinstance(cfg, RslrlCfg)
    assert cfg.num_envs > 0
    assert cfg.runner.experiment_name


def test_skrl_default_cfg_lookup():
    import motlab_rl
    from motlab_rl.skrl.cfg import SkrlCfg

    cfg = motlab_rl.default_skrl_cfg("cartpole")
    assert isinstance(cfg, SkrlCfg)
    assert cfg.num_envs > 0
    assert cfg.agent.rollouts > 0
    assert cfg.runner.experiment_name

    assert "cartpole" in motlab_rl.list_registered("rslrl")
    assert "cartpole" in motlab_rl.list_registered("skrl")
    assert "go1-velocity" in motlab_rl.list_registered("skrl")


def test_unknown_env_raises():
    import motlab

    with pytest.raises(KeyError):
        motlab.make_cfg("does-not-exist")
