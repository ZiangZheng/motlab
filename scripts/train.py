#!/usr/bin/env python
"""Train a MotLab environment with PPO (rsl_rl or skrl).

    python scripts/train.py --env cartpole
    python scripts/train.py --env cartpole --framework skrl
    python scripts/train.py --env go1-velocity --num-envs 128 --seed 123
"""

from __future__ import annotations

import logging

from absl import app, flags

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "Environment name (as registered).")
_FRAMEWORK = flags.DEFINE_enum(
    "framework", "rslrl", ["rslrl", "skrl"], "RL framework backend."
)
_NUM_ENVS = flags.DEFINE_integer("num-envs", None, "Override number of parallel envs.")
_RENDER = flags.DEFINE_bool("render", False, "Open a viewer during training.")
_SEED = flags.DEFINE_integer("seed", None, "Random seed override.")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Use a random seed.")
_TIMESTEPS = flags.DEFINE_integer(
    "timesteps", None, "(skrl only) Total env timesteps to train for."
)


def main(_argv) -> None:
    overrides: dict = {}
    if _NUM_ENVS.present:
        overrides["num_envs"] = _NUM_ENVS.value

    if _FRAMEWORK.value == "rslrl":
        from motlab_rl.rslrl.torch.train import RslrlTrainer

        if _RAND_SEED.value:
            overrides["runner.seed"] = None
        elif _SEED.present:
            overrides["runner.seed"] = _SEED.value
        trainer = RslrlTrainer(
            env_name=_ENV.value,
            cfg_override=overrides,
            enable_render=_RENDER.value,
        )
    else:
        from motlab_rl.skrl.torch.train import SkrlTrainer

        if _RAND_SEED.value:
            overrides["runner.seed"] = None
        elif _SEED.present:
            overrides["runner.seed"] = _SEED.value
        if _TIMESTEPS.present:
            overrides["runner.timesteps"] = _TIMESTEPS.value
        trainer = SkrlTrainer(env_name=_ENV.value, cfg_override=overrides)

    trainer.train()


if __name__ == "__main__":
    app.run(main)
