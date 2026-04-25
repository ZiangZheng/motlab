# MotLab

A torch-native reinforcement learning framework for robot training, built on
[MotrixSim](https://motphys.com). The architecture follows **IsaacLab**
(manager-based env, articulation/asset abstractions, configclass system,
MDP function library), keeping the Python surface familiar while running on
MotrixSim's batched solver:

- **Torch-only API** — obs, reward, actions, commands, manager outputs are
  all `torch.Tensor` on the env's device. Numpy lives only inside
  `engine/motrix.py`, the single np↔torch boundary.
- **Manager-based envs** — `ManagerBasedRLEnv` composes Action /
  Observation / Reward / Termination / Event / Command / Curriculum
  managers from declarative `@configclass` cfgs.
- **Sim2real-ready actuators** — `IdealPDActuator` (kp, kd, effort limit)
  bridging joint targets and engine torques.
- **Multi-install** — uv workspace, conda env, or plain pip.

## Layout

```
scripts/                   # train / play / view / bench entry points
packages/motlab/           # core: managers, envs, MDP library, registry
packages/motlab_assets/    # robot configs + bundled MJCFs (cartpole, go1)
packages/motlab_tasks/     # ready-made envs (fires @envcfg on import)
packages/motlab_rl/        # rsl_rl integration (PPO)
test/                      # pure-Python smoke tests
```

See [CLAUDE.md](CLAUDE.md) for an architecture & contribution guide.

## Quick start

```bash
# auto-detect uv / conda / pip and install everything
bash scripts/install.sh
# or pick a specific path:
bash scripts/install.sh --method uv  --rllib rslrl
bash scripts/install.sh --method conda
bash scripts/install.sh --method pip

# Smoke-test and train
python scripts/view.py --env cartpole
python scripts/train.py --env cartpole

# Quadruped locomotion
python scripts/view.py  --env go1-velocity --num-envs 16
python scripts/bench.py --env go1-velocity --num-envs 32 --steps 200
python scripts/train.py --env go1-velocity --num-envs 128
```

MotrixSim is on public PyPI (`pip install motrixsim`); Python 3.10 / 3.11 / 3.12.

## Status

- **Cartpole** reference task — 4-d obs, 1-d action, IsaacLab-style
  manager pipeline; rolls out and trains end-to-end.
- **`go1-velocity`** quadruped locomotion — 48-d obs (`base_lin_vel`,
  `base_ang_vel`, `projected_gravity`, velocity command, joint
  pos/vel/last_action), 12-d actions, `IdealPDActuator`, 8 reward terms
  (track_lin/ang, lin_vel_z, ang_vel_xy, orientation, torques,
  action_rate, alive). Stable up to ~128 envs; see `CLAUDE.md` *Known
  issues* for the MotrixSim solver bound at larger batches.
