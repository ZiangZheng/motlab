# MotLab

A torch-native reinforcement learning framework for robot training, built on
[MotrixSim](https://motphys.com). MotLab stays close to MotrixLab's simple,
direct-env style while adding a few upgrades:

- **Torch-only API** — obs, reward, actions, commands, manager outputs are
  all `torch.Tensor` on `cfg.device`. Numpy lives only inside the engine
  adapter.
- **Engine adapter** — all MotrixSim calls funnel through
  `motlab_envs/engine/motrix.py`; tasks stay engine-agnostic.
- **Manager-lite helpers** — opt-in `RewardManager`, `ObservationManager`,
  `TerminationManager`, `VelocityCommandManager` for tasks with many terms.
- **Sim2Real primitives** — PD actuator with latency/noise and obs/action
  wrappers shipped from day one.
- **Multi-install** — uv workspace, conda env, or plain pip.

## Layout

```
scripts/                   # train / play / view / bench entry points
packages/motlab_envs/      # env base + registry + tasks
packages/motlab_rl/        # SKRL + rsl_rl integrations
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
bash scripts/install.sh --method pip  --rllib skrl-torch

# Smoke-test and train
python scripts/view.py --env cartpole
python scripts/train.py --env cartpole --num-envs 4096

# Quadruped locomotion
python scripts/view.py  --env go1-velocity --num-envs 16
python scripts/bench.py --env go1-velocity --num-envs 32 --steps 200
python scripts/train.py --env go1-velocity --num-envs 128
```

MotrixSim is on public PyPI (`pip install motrixsim`); Python 3.10 / 3.11 / 3.12.

## Status

- **Cartpole** reference task works end-to-end (~900k steps/s at N=512).
- **`go1-velocity`** quadruped locomotion — 48-d obs, 12-d actions,
  PDActuator + Manager-lite rewards (track_lin/ang, lin_vel_z, orientation,
  torques, action_rate, alive). Rolls out OK up to ~128 envs; see
  `CLAUDE.md` *Known issues* for the MotrixSim solver bound at larger
  batches.
- Asset zoo ports **Unitree Go1** from mjlab (meshes + MJCF + motlab-style
  constants). `locomotion/g1` and a torch-tensor pathway are next.
