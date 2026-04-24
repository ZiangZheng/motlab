#!/usr/bin/env python
"""Train a MotLab environment with SKRL or rsl_rl.

    python scripts/train.py --env cartpole
    python scripts/train.py --env cartpole --rllib rslrl
    python scripts/train.py --env cartpole --num-envs 4096 --seed 123
"""

from __future__ import annotations

import logging

from absl import app, flags

from motlab_rl import utils

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "Environment name (as registered).")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Sim backend (auto if unset).")
_NUM_ENVS = flags.DEFINE_integer("num-envs", None, "Override number of parallel envs.")
_RENDER = flags.DEFINE_bool("render", False, "Open a viewer during training.")
_TRAIN_BACKEND = flags.DEFINE_string("train-backend", None, "'torch' or 'jax'.")
_SEED = flags.DEFINE_integer("seed", None, "Random seed override.")
_RAND_SEED = flags.DEFINE_bool("rand-seed", False, "Use a random seed.")
_RLLIB = flags.DEFINE_string("rllib", "rslrl", "RL framework: 'skrl' or 'rslrl'.")


def pick_train_backend(supports: utils.DeviceSupports, user_choice: str | None, rllib: str) -> str:
    if rllib == "rslrl":
        if user_choice and user_choice != "torch":
            raise SystemExit("rsl_rl only supports the torch backend.")
        if not supports.torch:
            raise SystemExit("rsl_rl requires torch; it is not importable on this environment.")
        return "torch"

    if user_choice:
        if user_choice == "jax" and not supports.jax:
            raise SystemExit("jax backend requested but jax is not installed.")
        if user_choice == "torch" and not supports.torch:
            raise SystemExit("torch backend requested but torch is not installed.")
        return user_choice

    if supports.torch and supports.torch_gpu:
        return "torch"
    if supports.jax and supports.jax_gpu:
        return "jax"
    if supports.torch:
        return "torch"
    if supports.jax:
        return "jax"
    raise SystemExit("Neither torch nor jax is installed; cannot train.")


def main(_argv) -> None:
    supports = utils.get_device_supports()
    logger.info("Device supports: %s", supports)

    overrides: dict = {}
    if _NUM_ENVS.present:
        overrides["num_envs"] = _NUM_ENVS.value
    if _RAND_SEED.value:
        overrides["runner.seed"] = None
    elif _SEED.present:
        overrides["runner.seed"] = _SEED.value

    rllib = _RLLIB.value
    backend = pick_train_backend(supports, _TRAIN_BACKEND.value, rllib)

    if rllib == "rslrl":
        from motlab_rl.rslrl.torch.train import RslrlTrainer

        trainer = RslrlTrainer(
            env_name=_ENV.value,
            sim_backend=_SIM_BACKEND.value,
            cfg_override=overrides,
            enable_render=_RENDER.value,
        )
    elif rllib == "skrl":
        if backend == "torch":
            from motlab_rl.skrl.torch.train import SkrlTorchTrainer

            trainer = SkrlTorchTrainer(
                env_name=_ENV.value,
                sim_backend=_SIM_BACKEND.value,
                cfg_override=overrides,
                enable_render=_RENDER.value,
            )
        else:
            raise SystemExit(
                f"SKRL {backend} backend is not wired up yet — contributions welcome. Use --train-backend torch."
            )
    else:
        raise SystemExit(f"Unknown --rllib '{rllib}'. Supported: skrl, rslrl.")

    trainer.train()


if __name__ == "__main__":
    app.run(main)
