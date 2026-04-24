# CLAUDE.md — motlab

Guidance for AI assistants working in this repo.

## Project summary

**MotLab** is a reinforcement learning framework built on the **MotrixSim**
physics engine. It mirrors the structure of MotrixLab (Motphys's upstream
reference) but:

- **torch-native end-to-end** — all env state, managers, MDP functions and
  RL wrappers speak `torch.Tensor`. Numpy is only used inside
  `motlab_envs/engine/motrix.py` to bridge MotrixSim's numpy buffers.
- isolates all engine calls in `motlab_envs/engine/motrix.py` (single adapter
  point),
- offers opt-in **Manager-lite** helpers (`motlab_envs/managers/`),
- ships **sim2real** primitives (actuator delay/noise) from day one,
- installs cleanly via **uv**, **conda** or plain **pip**.

## Workspace layout

```
motlab/
├── pyproject.toml                   # workspace root (uv + dev tools)
├── uv.toml                          # uv-specific indexes
├── environment.yml                  # conda env
├── scripts/{train,play,view,bench}.py
├── test/test_smoke.py               # pure-Python smoke tests (no engine)
└── packages/
    ├── motlab_envs/                 # env library
    │   └── src/motlab_envs/
    │       ├── base.py              # ABEnv, EnvCfg + sub-cfgs
    │       ├── registry.py          # decorator registry
    │       ├── env.py               # TorchEnv / TensorEnvState (direct-style)
    │       ├── manager_env.py       # ManagerBasedTorchEnv (declarative)
    │       ├── wrappers.py          # obs/action noise + latency wrappers
    │       ├── engine/motrix.py     # MotrixSim adapter (only place that imports motrixsim AND the only np↔torch boundary)
    │       ├── managers/            # opt-in reward/obs/termination/commands (torch)
    │       ├── math/quaternion.py   # torch quaternion utils
    │       ├── utils/rewards.py     # tolerance(), sigmoids (torch)
    │       ├── sim2real/actuator.py # PD + latency + noise (torch)
    │       ├── asset_zoo/robots/    # MJCF + constants per robot
    │       │   └── unitree_go1/
    │       └── tasks/
    │           ├── basic/cartpole/
    │           └── locomotion/velocity/go1/
    └── motlab_rl/                   # RL integrations
        └── src/motlab_rl/
            ├── registry.py          # per-env default RL cfgs
            ├── wrappers/            # TorchEnv → rsl_rl / skrl adapters
            ├── skrl/ rslrl/         # framework-specific trainers
            └── tasks/cartpole.py    # default PPO cfgs for cartpole
```

## Install

MotrixSim (the physics engine) is on public PyPI — no private index needed.
Python 3.10 / 3.11 / 3.12 are supported.

The fastest path is the bundled installer, which auto-detects `uv` /
`conda` / `pip` and sets everything up:

```bash
bash scripts/install.sh                     # auto-detect + rsl_rl extra
bash scripts/install.sh --method uv --rllib skrl-torch
bash scripts/install.sh --method conda
bash scripts/install.sh --method pip --no-extras   # minimal (envs only)
```

Or step-by-step:

### uv
```bash
uv sync --all-packages --extra rslrl        # or --extra skrl-torch
```

### conda + pip
```bash
conda env create -f environment.yml
conda activate motlab
pip install -e packages/motlab_envs -e "packages/motlab_rl[rslrl]"
```

### plain pip
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install motrixsim torch
pip install -e packages/motlab_envs -e "packages/motlab_rl[rslrl]"
```

## Common commands

```bash
# Smoke test (no MotrixSim required)
pytest test/ -q

# Quick random-action rollout
python scripts/view.py --env cartpole

# Train
python scripts/train.py --env cartpole                # default: rsl_rl + torch
python scripts/train.py --env cartpole --rllib skrl
python scripts/train.py --env cartpole --num-envs 4096 --seed 123

# Benchmark throughput
python scripts/bench.py --env cartpole --num-envs 4096

# Evaluate a checkpoint
python scripts/play.py --env cartpole --policy runs/cartpole/.../model.pt
```

## Adding a new env

1. (If it's a new robot) drop MJCF + meshes into
   `packages/motlab_envs/src/motlab_envs/asset_zoo/robots/<robot>/xmls/`, and
   write a `<robot>_constants.py` that exports joint names, default angles,
   PD gains, and init pose as plain Python tuples (no numpy).
2. Create a folder under `packages/motlab_envs/src/motlab_envs/tasks/...`
   with `cfg.py`, `<name>.py`, and (if needed) a task-specific `scene.xml`
   that `<include>`s the robot xml and adds actuators + terrain.
3. Decorate the cfg with `@registry.envcfg("<name>")` and the env class with
   `@registry.env("<name>", sim_backend="torch")`.
4. Import the new package from the enclosing `__init__.py` so the decorators
   fire on import.
5. For default training hyperparameters, add a file under
   `packages/motlab_rl/src/motlab_rl/tasks/` and use `@rlcfg(<name>, ...)`.

## Conventions

- **Torch-only.** Every tensor that flows through an env — obs, reward,
  terminated, truncated, actions, commands, managers' intermediates — is a
  `torch.Tensor` on `cfg.device`. The sole exception is
  `motlab_envs.engine.motrix`, which wraps MotrixSim's numpy buffers via
  `torch.from_numpy(...)` (zero-copy on CPU) and copies torch tensors back
  on set-paths. Task code must **not** import numpy.
- Tasks **never** import `motrixsim` directly. Go through
  `motlab_envs.engine`.
- Keep `TensorEnvState` the only mutable container passed between
  `apply_action` / `update_state` / `reset`.
- Distinguish `terminated` from `truncated` — rsl_rl needs the distinction
  for correct bootstrap.
- Reward weights live in `cfg.reward.scales`; shaping constants in
  `cfg.reward`. Never hard-code them inside `_reward_*` methods.
- When a task grows more than ~5 reward terms, migrate to
  `motlab_envs.managers.RewardManager` or a full
  `ManagerBasedTorchEnv` rather than expanding `_reward_*` inline.

## Known issues

- **MotrixSim solver panic at large batch sizes with floating-base robots.**
  Symptom: `pyo3_runtime.PanicException: LTL factorization failed:
  NotPositiveDefinite` during `physics_step`. Reproduces with
  ``go1-velocity`` at ``num_envs >= 256`` rolling out ~100 steps of random
  actions — lots of quadrupeds collapse into degenerate contact piles before
  the termination handler resets them, and the batched solver can't recover
  the mass matrix. Workaround: start training with ``num_envs = 128`` and
  raise once a policy is learning; also ensure sane joint ``armature`` +
  ``damping`` in the robot MJCF.
