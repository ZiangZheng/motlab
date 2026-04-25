#!/usr/bin/env python
"""Quick-inspect an env with random actions (no training)."""

from __future__ import annotations

import torch
from absl import app, flags

_ENV = flags.DEFINE_string("env", "cartpole", "Env name.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 4, "Parallel envs.")
_STEPS = flags.DEFINE_integer("steps", 500, "Number of control steps.")


def main(_argv) -> None:
    import motlab
    import motlab_tasks  # noqa: F401  (registers built-in envs)

    cfg = motlab.make_cfg(_ENV.value)
    cfg.scene.num_envs = _NUM_ENVS.value
    env = motlab.ManagerBasedRLEnv(cfg, device="cpu")
    env.reset()
    for step in range(_STEPS.value):
        actions = 0.1 * torch.randn(env.num_envs, env.action_dim, device=env.device)
        _, reward, term, trunc, _ = env.step(actions)
        if step % 50 == 0:
            done = (term | trunc).sum().item()
            print(f"step={step} mean_reward={reward.mean().item():.3f} dones={done}")


if __name__ == "__main__":
    app.run(main)
