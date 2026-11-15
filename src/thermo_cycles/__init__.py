"""thermo_cycles: ideal thermodynamic cycles as an importable library."""

from . import errors
from .fluids import AIR, IdealGas
from .state import State
from .model.cycle_def import get_cycle, list_cycles, register_cycle
from . import catalogue  # noqa: F401  (populates the cycle registry)
from .results import CycleResults, StepResult
from .cycle_state import CycleState
from .cycle import Cycle, requirements
from .optimise import OptimisationResult, optimise

__version__ = "0.1.0"
__all__ = ["Cycle", "IdealGas", "AIR", "requirements", "list_cycles", "register_cycle", "errors",
           "State", "StepResult", "CycleResults", "CycleState", "get_cycle", "__version__",
           "optimise", "OptimisationResult"]
