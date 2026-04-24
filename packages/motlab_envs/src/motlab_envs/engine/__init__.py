"""Engine adapter layer.

All MotrixSim imports are funneled through this package. Task code should
never `import motrixsim` directly — go through :mod:`motlab_envs.engine.motrix`.
This keeps the option of swapping engines (or adding an alternative backend)
without rewriting the tasks.
"""

from motlab_envs.engine.motrix import (  # noqa: F401
    SceneData,
    SceneModel,
    load_model,
)
