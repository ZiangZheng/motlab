"""skrl PPO trainer for MotLab manager-based envs (PyTorch backend)."""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import motlab
from motlab_rl import registry as rl_registry
from motlab_rl.skrl.cfg import SkrlCfg
from motlab_rl.utils import apply_overrides
from motlab_rl.wrappers.skrl import SkrlVecEnv

_ACTIVATIONS = {
    "elu": "torch.nn.ELU",
    "relu": "torch.nn.ReLU",
    "tanh": "torch.nn.Tanh",
}


class SkrlTrainer:
    """Wires :class:`SkrlVecEnv` to skrl's PPO + RandomMemory + SequentialTrainer."""

    def __init__(
        self,
        env_name: str,
        cfg_override: Optional[dict] = None,
        log_root: str = "runs",
    ) -> None:
        import torch
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.memories.torch import RandomMemory
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
        from skrl.trainers.torch import SequentialTrainer
        from skrl.utils import set_seed

        cfg: SkrlCfg = rl_registry.default_skrl_cfg(env_name)
        if cfg_override:
            apply_overrides(cfg, cfg_override)
        self.cfg = cfg

        if cfg.runner.seed is not None:
            set_seed(cfg.runner.seed)

        env_cfg = motlab.make_cfg(env_name)
        env_cfg.scene.num_envs = cfg.num_envs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env_cfg.sim.device = device
        env = motlab.ManagerBasedRLEnv(env_cfg, device=device)
        self.env = SkrlVecEnv(env)

        obs_dim = int(self.env.observation_space.shape[0])
        act_dim = int(self.env.action_space.shape[0])
        activation_cls = _resolve_activation(cfg.policy.activation)

        class Actor(GaussianMixin, Model):
            def __init__(self, observation_space, action_space, device):
                Model.__init__(
                    self,
                    observation_space=observation_space,
                    action_space=action_space,
                    device=device,
                )
                GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True)
                layers, in_dim = [], obs_dim
                for h in cfg.policy.hidden_dims:
                    layers += [torch.nn.Linear(in_dim, h), activation_cls()]
                    in_dim = h
                self.net = torch.nn.Sequential(*layers)
                self.mean = torch.nn.Linear(in_dim, act_dim)
                init_log = torch.log(torch.tensor(cfg.policy.init_noise_std))
                self.log_std = torch.nn.Parameter(init_log * torch.ones(act_dim))

            def compute(self, inputs, role):
                h = self.net(inputs["observations"])
                return self.mean(h), {"log_std": self.log_std}

        class Critic(DeterministicMixin, Model):
            def __init__(self, observation_space, action_space, device):
                Model.__init__(
                    self,
                    observation_space=observation_space,
                    action_space=action_space,
                    device=device,
                )
                DeterministicMixin.__init__(self, clip_actions=False)
                layers, in_dim = [], obs_dim
                for h in cfg.policy.hidden_dims:
                    layers += [torch.nn.Linear(in_dim, h), activation_cls()]
                    in_dim = h
                layers.append(torch.nn.Linear(in_dim, 1))
                self.net = torch.nn.Sequential(*layers)

            def compute(self, inputs, role):
                return self.net(inputs["observations"]), {}

        models = {
            "policy": Actor(self.env.observation_space, self.env.action_space, device),
            "value": Critic(self.env.observation_space, self.env.action_space, device),
        }

        memory = RandomMemory(memory_size=cfg.agent.rollouts, num_envs=cfg.num_envs, device=device)

        agent_cfg = dataclasses.asdict(PPO_CFG())
        for k, v in dataclasses.asdict(cfg.agent).items():
            agent_cfg[k] = v
        agent_cfg["experiment"] = {
            "directory": os.path.join(log_root, env_name),
            "experiment_name": cfg.runner.experiment_name,
            "write_interval": "auto",
            "checkpoint_interval": "auto",
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {},
        }

        self.agent = PPO(
            models=models,
            memory=memory,
            cfg=agent_cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=device,
        )
        self.trainer = SequentialTrainer(
            cfg={"timesteps": cfg.runner.timesteps, "headless": True},
            env=self.env,
            agents=self.agent,
        )

    def train(self) -> None:
        self.trainer.train()

    def eval(self) -> None:
        self.trainer.eval()


def _resolve_activation(name: str):
    import torch.nn as nn

    table = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    if name not in table:
        raise ValueError(f"Unsupported activation {name!r}; pick one of {sorted(table)}")
    return table[name]
