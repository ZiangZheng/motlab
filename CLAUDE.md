# CLAUDE.md — motlab

Guidance for AI assistants working in this repo.

## Project summary

**MotLab** is a reinforcement learning framework built on the **MotrixSim**
physics engine. The architecture mirrors **IsaacLab** at the package level
(four cooperating packages: `motlab` core, `motlab_assets`, `motlab_tasks`,
`motlab_rl`) and at the API level (manager-based env, articulation/asset
abstractions, configclass system, MDP function library) — but stays
torch-native end-to-end:

- **torch-only** — every tensor that flows through an env (obs, reward,
  terminated, truncated, actions, commands, manager intermediates) is a
  `torch.Tensor` on the env's device. Numpy is confined to
  `motlab/engine/motrix.py`, the only place that bridges MotrixSim's numpy
  buffers via `torch.from_numpy`.
- isolates all engine calls in a single adapter (`engine/motrix.py`),
- declarative env construction via `@configclass` cfgs + manager terms,
- ships `IdealPDActuator` for sim2real-style PD control,
- installs cleanly via **uv**, **conda** or plain **pip**.

## Workspace layout

```
motlab/
├── pyproject.toml                    # workspace root (uv + dev tools)
├── uv.toml                           # uv-specific indexes
├── environment.yml                   # conda env
├── scripts/{train,play,view,bench}.py
├── test/test_smoke.py                # pure-Python smoke tests
└── packages/
    ├── motlab/                       # CORE: framework, no robot/task content
    │   └── src/motlab/
    │       ├── __init__.py           # exports ManagerBasedRLEnv*, registry helpers
    │       ├── registry.py           # @envcfg / make / make_cfg / list_envs
    │       ├── utils/
    │       │   ├── configclass.py    # IsaacLab-style @configclass decorator
    │       │   ├── math.py           # torch quaternion / frame transforms (wxyz)
    │       │   └── string_utils.py
    │       ├── sim/simulation_cfg.py # SimulationCfg
    │       ├── engine/motrix.py      # MotrixEngine — only numpy↔torch boundary
    │       ├── actuators/            # ActuatorBase + IdealPDActuator
    │       ├── assets/               # Articulation + ArticulationData (the *class*, not robot cfgs)
    │       ├── scene/                # InteractiveScene + InteractiveSceneCfg
    │       ├── managers/             # action / obs / reward / term / event / cmd / curriculum
    │       ├── mdp/                  # MDP function library (obs/rewards/term/events/...)
    │       │   ├── observations.py
    │       │   ├── rewards.py
    │       │   ├── terminations.py
    │       │   ├── events.py
    │       │   ├── curriculums.py
    │       │   ├── actions/joint_actions.py
    │       │   └── commands/velocity_command.py
    │       └── envs/
    │           ├── manager_based_env.py / *_cfg.py
    │           └── manager_based_rl_env.py / *_cfg.py
    │
    ├── motlab_assets/                # ROBOT CFGS + bundled MJCFs
    │   └── src/motlab_assets/
    │       ├── __init__.py           # re-exports CARTPOLE_CFG, GO1_CFG, ...
    │       ├── cartpole/
    │       │   ├── cartpole.py       # CARTPOLE_CFG (ArticulationCfg)
    │       │   └── xmls/cartpole.xml
    │       └── unitree_go1/
    │           ├── unitree_go1.py    # GO1_CFG
    │           └── xmls/{go1.xml, scene.xml, assets/}
    │
    ├── motlab_tasks/                 # READY-MADE ENVS (fires @envcfg on import)
    │   └── src/motlab_tasks/
    │       ├── __init__.py
    │       ├── classic/
    │       │   └── cartpole/cartpole_env_cfg.py
    │       └── locomotion/
    │           └── velocity/go1/go1_velocity_env_cfg.py
    │
    └── motlab_rl/                    # RL INTEGRATION (rsl_rl + skrl PPO)
        └── src/motlab_rl/
            ├── __init__.py           # imports motlab + motlab_tasks to register
            ├── registry.py           # @rlcfg / @skrlcfg + default_{rl,skrl}_cfg
            ├── utils.py              # apply_overrides, DeviceSupports
            ├── wrappers/
            │   ├── rslrl.py          # RslrlVecEnv (TensorDict obs)
            │   └── skrl.py           # SkrlVecEnv (skrl Wrapper, flat obs)
            ├── rslrl/
            │   ├── cfg.py            # RslrlCfg + obs_groups
            │   └── torch/train.py    # RslrlTrainer (OnPolicyRunner)
            ├── skrl/
            │   ├── cfg.py            # SkrlCfg (PPO_CFG-shaped subset)
            │   └── torch/train.py    # SkrlTrainer (PPO + RandomMemory + SeqTrainer)
            └── tasks/{cartpole,go1_velocity}.py   # both @rlcfg + @skrlcfg per env
```

## Install

MotrixSim ships on public PyPI; Python 3.10–3.12 are supported.

```bash
bash scripts/install.sh                     # auto-detect installer + rsl_rl
bash scripts/install.sh --method uv
bash scripts/install.sh --method conda
bash scripts/install.sh --method pip --no-extras   # minimal (envs only)
```

Step-by-step alternatives:

```bash
# uv
uv sync --all-packages --extra rslrl              # or --extra skrl, or both
uv sync --all-packages --extra rslrl --extra skrl

# conda + pip
conda env create -f environment.yml
conda activate motlab
pip install -e packages/motlab -e packages/motlab_assets \
            -e packages/motlab_tasks -e "packages/motlab_rl[rslrl,skrl]"

# plain pip
python3.10 -m venv .venv && source .venv/bin/activate
pip install motrixsim torch
pip install -e packages/motlab -e packages/motlab_assets \
            -e packages/motlab_tasks -e "packages/motlab_rl[rslrl,skrl]"
```

## Common commands

```bash
# Smoke test (pure-Python)
pytest test/ -q

# Quick random-action rollout
python scripts/view.py --env cartpole

# Train (PPO; --framework {rslrl,skrl}, default rslrl)
python scripts/train.py --env cartpole
python scripts/train.py --env cartpole --framework skrl
python scripts/train.py --env go1-velocity --num-envs 128 --seed 123

# Benchmark throughput
python scripts/bench.py --env cartpole --num-envs 4096

# Evaluate a checkpoint
python scripts/play.py --env cartpole --policy runs/cartpole/.../model.pt
```

## Adding a new env

1. **Robot asset** (if new) — drop MJCF + meshes into
   `packages/motlab_assets/src/motlab_assets/<robot>/xmls/`, and build an
   `ArticulationCfg` in `<robot>.py` exporting a module-level `<ROBOT>_CFG`.
   Specify joint regex → actuator group mapping inside the cfg
   (kp/kd/effort_limit per group). Re-export it from
   `motlab_assets/__init__.py`.
2. **Task cfg** — under `packages/motlab_tasks/src/motlab_tasks/<group>/...`
   create `<task>_env_cfg.py`. Subclass `ManagerBasedRLEnvCfg` and define
   nested cfgs (`SceneCfg`, `ActionsCfg`, `ObservationsCfg`,
   `RewardsCfg`, `TerminationsCfg`, optional `EventCfg`/`CommandsCfg`).
   - **CRITICAL:** every nested cfg field must have a type annotation,
     e.g. `joint_pos: ObservationTermCfg = ObservationTermCfg(...)`. Without
     the annotation, `@configclass` won't see it as a dataclass field and
     the manager will silently treat the group as empty.
3. Decorate the top-level cfg with `@envcfg("<name>")` (imported from
   `motlab.registry`).
4. Wire the new task into the `motlab_tasks/__init__.py` chain so
   `import motlab_tasks` triggers its `@envcfg` decorator.
5. **RL hyperparameters** — add a file under
   `packages/motlab_rl/src/motlab_rl/tasks/<name>.py` and decorate the cfg
   with `@rlcfg("<name>")` (rsl_rl) and/or `@skrlcfg("<name>")` (skrl).
   Each env may register one, the other, or both.

## Conventions

- **Torch-only.** Tasks must NOT import numpy or motrixsim directly. The
  only legitimate np↔torch boundary is `motlab/engine/motrix.py`.
- **Quaternion layout.** Internally everything is `(w, x, y, z)`. MotrixSim
  uses `(x, y, z, w)` — convert at the engine boundary via
  `motlab.utils.math.convert_quat`.
- **Actuator vs joint ordering.** MotrixSim's `actuator_names` order is
  generally NOT the same as `joint_names` order. The `Articulation` class
  computes `_actuator_joint_ids` once and uses it whenever it writes
  torques back to the engine — always go through Articulation, never poke
  `engine.actuator_ctrls` by index.
- **`@configclass` mutable defaults.** The decorator auto-wraps mutable
  defaults via `default_factory` so each instance gets a fresh copy.
- **Manager init order.** `ManagerBasedEnv.__init__` defers
  `ObservationManager` creation to `_finalize_managers()`. RL subclasses
  build command/reward/termination/curriculum managers first, *then* call
  `self._finalize_managers()`.
- **Registration via import side-effects.** `motlab` core has no built-in
  envs. Tasks register themselves only when `motlab_tasks` is imported (or
  the user imports a specific task module). Scripts that want a registered
  env always do `import motlab_tasks  # noqa: F401`. The `motlab_rl`
  package does this automatically in its `__init__.py`.
- **Step return.** `ManagerBasedRLEnv.step` returns
  `(obs_dict, reward, terminated, truncated, info)`. The rsl_rl wrapper
  flattens obs into a `TensorDict` and surfaces `truncated` as
  `extras["time_outs"]`.

## Known issues

- **MotrixSim solver panic at large batch sizes with floating-base robots.**
  Symptom: `pyo3_runtime.PanicException: LTL factorization failed:
  NotPositiveDefinite` during `physics_step`. Reproduces with
  ``go1-velocity`` at ``num_envs >= 256`` rolling out ~100 steps of random
  actions. Workaround: start training with ``num_envs = 128`` and raise
  once a policy is learning; ensure sane joint ``armature`` + ``damping``
  in the robot MJCF.
