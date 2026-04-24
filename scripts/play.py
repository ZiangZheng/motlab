#!/usr/bin/env python
"""Roll out a trained policy in a MotLab env.

    python scripts/play.py --env cartpole --policy runs/cartpole/.../model.pt
"""

from __future__ import annotations

import logging

from absl import app, flags

logger = logging.getLogger(__name__)

_ENV = flags.DEFINE_string("env", "cartpole", "Environment name.")
_SIM_BACKEND = flags.DEFINE_string("sim-backend", None, "Sim backend.")
_NUM_ENVS = flags.DEFINE_integer("num-envs", 16, "Number of envs to roll out in parallel.")
_POLICY = flags.DEFINE_string("policy", None, "Path to saved policy (torch .pt / skrl .pickle).")
_RLLIB = flags.DEFINE_string("rllib", "rslrl", "RL framework to load the policy with.")
_STEPS = flags.DEFINE_integer("steps", 1000, "Number of control steps to roll out.")


def main(_argv) -> None:
    from motlab_envs import registry as env_registry

    env = env_registry.make(_ENV.value, sim_backend=_SIM_BACKEND.value, num_envs=_NUM_ENVS.value)

    if _POLICY.value is None:
        logger.warning("--policy not given; running zero-action rollout.")
        import torch

        actions = torch.zeros(
            (env.num_envs, env.action_space.shape[0]), dtype=torch.float32, device=env.device,
        )
        for _ in range(_STEPS.value):
            env.step(actions)
        return

    if _RLLIB.value == "rslrl":
        import torch

        from motlab_rl.wrappers.rslrl import RslrlVecEnv

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapped = RslrlVecEnv(env, device=device)
        state_dict = torch.load(_POLICY.value, map_location=device)
        # rsl_rl stores an ActorCritic state_dict; build a placeholder and load.
        from rsl_rl.modules import ActorCritic

        actor_critic = ActorCritic(wrapped.num_obs, wrapped.num_obs, wrapped.num_actions)
        actor_critic.load_state_dict(state_dict["model_state_dict"])
        actor_critic.to(device).eval()
        obs, _ = wrapped.get_observations()
        with torch.no_grad():
            for _ in range(_STEPS.value):
                actions = actor_critic.act_inference(obs)
                obs, _, _, _ = wrapped.step(actions)
    else:
        raise SystemExit("SKRL play path is not implemented yet — contribute!")


if __name__ == "__main__":
    app.run(main)
