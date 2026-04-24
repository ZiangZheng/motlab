"""Smoke tests that do not require motrixsim.

Covers pure-Python pieces (base cfg, registry plumbing, managers, rewards).
End-to-end env tests live in the individual packages and require MotrixSim.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch


def test_env_cfg_defaults():
    from motlab_envs.base import EnvCfg

    cfg = EnvCfg()
    cfg.sim_dt = 0.005
    cfg.ctrl_dt = 0.02
    cfg.max_episode_seconds = 10.0
    cfg.validate()
    assert cfg.max_episode_steps == 500
    assert cfg.sim_substeps == 4


def test_env_cfg_invalid():
    from motlab_envs.base import EnvCfg

    cfg = EnvCfg()
    cfg.sim_dt = 0.02
    cfg.ctrl_dt = 0.01
    with pytest.raises(ValueError):
        cfg.validate()


def test_registry_roundtrip():
    from motlab_envs import registry
    from motlab_envs.base import ABEnv, EnvCfg

    @dataclass
    class DummyCfg(EnvCfg):
        pass

    class DummyEnv(ABEnv):
        def __init__(self, cfg, num_envs=1):
            self._cfg = cfg
            self._n = num_envs

        @property
        def num_envs(self):
            return self._n

        @property
        def cfg(self):
            return self._cfg

        @property
        def observation_space(self):
            raise NotImplementedError

        @property
        def action_space(self):
            raise NotImplementedError

    registry.register_env_config("dummy", DummyCfg)
    registry.register_env("dummy", DummyEnv, "torch")
    assert registry.contains("dummy")
    env = registry.make("dummy", num_envs=3)
    assert env.num_envs == 3
    assert isinstance(env.cfg, DummyCfg)


def test_reward_manager_weighted_sum():
    from motlab_envs.managers import RewardManager

    mgr = RewardManager()
    mgr.add("a", lambda _: torch.ones(4), weight=2.0)
    mgr.add("b", lambda _: torch.full((4,), 0.5), weight=-1.0)
    total, per_term = mgr.compute(
        ctx=None, terminated=torch.tensor([False, True, False, False])
    )
    torch.testing.assert_close(per_term["a"], torch.full((4,), 2.0))
    torch.testing.assert_close(per_term["b"], torch.full((4,), -0.5))
    # Total should be 1.5 everywhere except the terminated env (zeroed).
    torch.testing.assert_close(total, torch.tensor([1.5, 0.0, 1.5, 1.5]))


def test_tolerance_reward():
    from motlab_envs.utils.rewards import tolerance

    x = torch.tensor([0.0, 0.5, 1.0, 2.0])
    out = tolerance(x, bounds=(0.0, 0.0), margin=1.0, sigmoid="linear", value_at_margin=0.0)
    assert float(out[0]) == pytest.approx(1.0)
    assert float(out[-1]) == 0.0
