"""MotLab RL framework integrations."""

from motlab_rl import registry  # noqa: F401

# Importing the tasks subpackage triggers per-env RL config registration.
# Tasks registered here can silently fail to register if their underlying
# env is missing (because, e.g., motrixsim isn't installed) — in that case
# we warn and continue so pure-Python inspection still works.
try:
    from motlab_rl import tasks  # noqa: F401
except ValueError:
    import logging

    logging.getLogger(__name__).warning(
        "Could not auto-register RL task configs (likely because engine is "
        "missing and envs didn't auto-register). Install motrixsim to enable."
    )

__version__ = "0.1.0"
