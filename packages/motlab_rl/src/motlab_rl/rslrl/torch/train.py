"""rsl_rl OnPolicyRunner trainer wired to a MotLab env."""

from __future__ import annotations

import os
from typing import Optional

from motlab_envs import registry as env_registry

from motlab_rl import registry as rl_registry
from motlab_rl.rslrl.cfg import RslrlCfg
from motlab_rl.utils import apply_overrides
from motlab_rl.wrappers.rslrl import RslrlVecEnv


class RslrlTrainer:
    def __init__(
        self,
        env_name: str,
        sim_backend: Optional[str] = None,
        cfg_override: Optional[dict] = None,
        enable_render: bool = False,
        log_root: str = "runs",
    ):
        import torch
        from rsl_rl.runners import OnPolicyRunner

        self._torch = torch

        cfg: RslrlCfg = rl_registry.default_rl_cfg(env_name, "rslrl", "torch")
        if cfg_override:
            apply_overrides(cfg, cfg_override)

        env = env_registry.make(env_name, sim_backend=sim_backend, num_envs=cfg.num_envs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = RslrlVecEnv(env, device=device)

        log_dir = os.path.join(log_root, env_name, cfg.runner.experiment_name)
        os.makedirs(log_dir, exist_ok=True)

        self.runner = OnPolicyRunner(
            env=self.env,
            train_cfg=cfg.to_runner_dict(),
            log_dir=log_dir,
            device=device,
        )
        self.cfg = cfg
        self.enable_render = enable_render

    def train(self):
        self.runner.learn(num_learning_iterations=self.cfg.runner.max_iterations)

    def eval(self, num_steps: int = 1000):
        policy = self.runner.get_inference_policy(device=self.runner.device)
        obs, _ = self.env.get_observations()
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _, _, _ = self.env.step(actions)
