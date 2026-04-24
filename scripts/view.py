#!/usr/bin/env python
"""Quick-inspect an env with random actions (no training)."""

from __future__ import annotations

import torch
from absl import app, flags

_ENV = flags.DEFINE_string("env", "cartpole", "Env name.")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Sim backend.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 4, "Parallel envs.")
_STEPS = flags.DEFINE_integer("steps", 500, "Number of control steps.")


def main(_argv) -> None:
    from motlab_envs import registry

    env = registry.make(_ENV.value, sim_backend=_SIM_BACKEND.value, num_envs=_NUM_ENVS.value)
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=env.device)
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=env.device)
    shape = (env.num_envs, env.action_space.shape[0])
    for step in range(_STEPS.value):
        actions = low + torch.rand(shape, device=env.device) * (high - low)
        state = env.step(actions)
        if step % 50 == 0:
            print(
                f"step={step} mean_reward={state.reward.mean().item():.3f} "
                f"dones={int(state.done.sum().item())}"
            )


if __name__ == "__main__":
    app.run(main)
