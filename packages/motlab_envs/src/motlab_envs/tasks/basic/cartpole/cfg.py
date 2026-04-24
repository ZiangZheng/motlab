"""CartPole env config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from motlab_envs import registry
from motlab_envs.base import AssetCfg, EnvCfg

_MODEL_FILE = os.path.join(os.path.dirname(__file__), "cartpole.xml")


@registry.envcfg("cartpole")
@dataclass
class CartPoleEnvCfg(EnvCfg):
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    max_episode_seconds: float = 10.0
    render_spacing: float = 2.0
    reset_noise_scale: float = 0.01
    asset: AssetCfg = field(default_factory=lambda: AssetCfg(model_file=_MODEL_FILE))
