"""motlab_assets — robot configurations + bundled MJCFs.

Importing this package re-exports the cfg constants of every robot under
the ``motlab_assets`` namespace for convenience::

    from motlab_assets import CARTPOLE_CFG, GO1_CFG
"""

from motlab_assets.cartpole.cartpole import CARTPOLE_CFG
from motlab_assets.unitree_go1.unitree_go1 import GO1_CFG

__all__ = ["CARTPOLE_CFG", "GO1_CFG"]
