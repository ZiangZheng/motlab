# motlab-rl

Unified RL training wrapper for MotLab environments.

- `registry.py` — per-env default RL configs (keyed by `(rllib, backend)`).
- `wrappers/` — convert `NpEnv` into the vectorised env interface expected by SKRL and rsl_rl.
- `skrl/`, `rslrl/` — framework-specific trainer entry points.
- `tasks/` — concrete default configs, e.g. `cartpole` PPO.

Pick an extra depending on your target framework:

```
pip install -e packages/motlab_rl[skrl-torch]
pip install -e packages/motlab_rl[rslrl]
```
