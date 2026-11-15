"""thermo_cycles: ideal thermodynamic cycles as an importable library."""

from .fluids import AIR, IdealGas
from .state import State
from .model.cycle_def import get_cycle, list_cycles, register_cycle
from . import catalogue  # noqa: F401  (populates the cycle registry)

__version__ = "0.1.0"
__all__ = ["AIR", "IdealGas", "State", "get_cycle", "list_cycles", "register_cycle", "__version__"]
