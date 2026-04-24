"""SKRL PPO runner for MotLab envs (torch backend)."""

from __future__ import annotations

from typing import Optional

from motlab_envs import registry as env_registry

from motlab_rl import registry as rl_registry
from motlab_rl.skrl.config import SkrlCfg
from motlab_rl.utils import apply_overrides
from motlab_rl.wrappers.skrl import SkrlVecEnv


class SkrlTorchTrainer:
    """Minimal SKRL PPO trainer — spins up GaussianMixin actor + DeterministicMixin critic."""

    def __init__(
        self,
        env_name: str,
        sim_backend: Optional[str] = None,
        cfg_override: Optional[dict] = None,
        enable_render: bool = False,
    ):
        import torch
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.trainers.torch import SequentialTrainer

        self._torch = torch

        cfg: SkrlCfg = rl_registry.default_rl_cfg(env_name, "skrl", "torch")
        if cfg_override:
            apply_overrides(cfg, cfg_override)

        env = env_registry.make(env_name, sim_backend=sim_backend, num_envs=cfg.num_envs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = SkrlVecEnv(env, device=device)

        obs_dim = int(self.env.observation_space.shape[0])
        act_dim = int(self.env.action_space.shape[0])

        class Actor(GaussianMixin, Model):
            def __init__(self, observation_space, action_space, device):
                Model.__init__(self, observation_space, action_space, device)
                GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 256), torch.nn.ELU(),
                    torch.nn.Linear(256, 128), torch.nn.ELU(),
                )
                self.mean = torch.nn.Linear(128, act_dim)
                self.log_std = torch.nn.Parameter(torch.zeros(act_dim))

            def compute(self, inputs, role):
                h = self.net(inputs["states"])
                return self.mean(h), self.log_std, {}

        class Critic(DeterministicMixin, Model):
            def __init__(self, observation_space, action_space, device):
                Model.__init__(self, observation_space, action_space, device)
                DeterministicMixin.__init__(self, clip_actions=False)
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(obs_dim, 256), torch.nn.ELU(),
                    torch.nn.Linear(256, 128), torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                )

            def compute(self, inputs, role):
                return self.net(inputs["states"]), {}

        models = {
            "policy": Actor(self.env.observation_space, self.env.action_space, device),
            "value": Critic(self.env.observation_space, self.env.action_space, device),
        }

        memory = RandomMemory(memory_size=cfg.rollouts, num_envs=cfg.num_envs, device=device)
        agent_cfg = PPO_DEFAULT_CONFIG.copy()
        agent_cfg.update({
            "rollouts": cfg.rollouts,
            "learning_epochs": cfg.runner.learning_epochs,
            "mini_batches": cfg.runner.mini_batches,
            "discount_factor": cfg.runner.discount_factor,
            "lambda": cfg.runner.lambda_,
            "learning_rate": cfg.runner.learning_rate,
            "ratio_clip": cfg.runner.clip_ratio,
            "entropy_loss_scale": cfg.runner.entropy_loss_scale,
        })
        self.agent = PPO(
            models=models,
            memory=memory,
            cfg=agent_cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=device,
        )
        self.trainer = SequentialTrainer(
            cfg={"timesteps": cfg.runner.timesteps, "headless": not enable_render},
            env=self.env,
            agents=self.agent,
        )

    def train(self):
        self.trainer.train()

    def eval(self):
        self.trainer.eval()
