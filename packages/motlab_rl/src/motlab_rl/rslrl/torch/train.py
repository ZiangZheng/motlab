"""rsl_rl OnPolicyRunner trainer wired to a MotLab manager-based env."""

from __future__ import annotations

import os
from typing import Optional

import motlab
from motlab_rl import registry as rl_registry
from motlab_rl.rslrl.cfg import RslrlCfg
from motlab_rl.utils import apply_overrides
from motlab_rl.wrappers.rslrl import RslrlVecEnv


class RslrlTrainer:
    def __init__(
        self,
        env_name: str,
        cfg_override: Optional[dict] = None,
        enable_render: bool = False,
        log_root: str = "runs",
    ):
        import torch
        from rsl_rl.runners import OnPolicyRunner

        self._torch = torch

        cfg: RslrlCfg = rl_registry.default_rl_cfg(env_name)
        if cfg_override:
            apply_overrides(cfg, cfg_override)

        env_cfg = motlab.make_cfg(env_name)
        env_cfg.scene.num_envs = cfg.num_envs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env_cfg.sim.device = device
        env = motlab.ManagerBasedRLEnv(env_cfg, device=device)
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
        obs = self.env.get_observations()
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _, _, _ = self.env.step(actions)
