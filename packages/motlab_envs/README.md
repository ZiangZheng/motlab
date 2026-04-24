# motlab-envs

Task library and engine adapters for MotLab. Built on top of [MotrixSim](https://motphys.com).

- `base.py` — `ABEnv`, `EnvCfg`, and structured sub-configs (asset / control / commands / rewards / ...).
- `registry.py` — decorator-based env & cfg registry.
- `engine/motrix.py` — single source of truth for MotrixSim interop.
- `np/env.py` — `NpEnv`, `NpEnvState` — batched env loop using `mtx.SceneData(batch=[N])`.
- `managers/` — opt-in helpers for reward / observation / termination / commands.
- `sim2real/` — actuator delay, action/obs noise.
- `tasks/` — concrete environments (cartpole, ...).
