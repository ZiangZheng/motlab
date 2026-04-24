#!/usr/bin/env python
"""Measure physics throughput for a registered env."""

from __future__ import annotations

import time

import torch
from absl import app, flags

_ENV = flags.DEFINE_string("env", "cartpole", "Env name.")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Sim backend.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 4096, "Parallel envs.")
_STEPS = flags.DEFINE_integer("steps", 500, "Number of control steps.")


def main(_argv) -> None:
    from motlab_envs import registry

    env = registry.make(_ENV.value, sim_backend=_SIM_BACKEND.value, num_envs=_NUM_ENVS.value)
    action_shape = (env.num_envs, env.action_space.shape[0])
    actions = torch.zeros(action_shape, dtype=torch.float32, device=env.device)

    # warmup
    for _ in range(10):
        env.step(actions)

    start = time.perf_counter()
    for _ in range(_STEPS.value):
        env.step(actions)
    dt = time.perf_counter() - start

    total = _STEPS.value * env.num_envs
    print(f"envs={env.num_envs} steps={_STEPS.value}  wall={dt:.2f}s  steps/s={total / dt:,.0f}")


if __name__ == "__main__":
    app.run(main)
