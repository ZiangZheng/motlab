"""motlab_tasks — ready-made manager-based environments.

Importing this package fires every ``@envcfg`` decorator inside the task
modules, registering the environments with :mod:`motlab`.
"""

from motlab_tasks import classic, locomotion  # noqa: F401

__all__ = ["classic", "locomotion"]
