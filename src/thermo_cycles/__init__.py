"""thermo_cycles: ideal thermodynamic cycles as an importable library."""

from . import errors, units
from .fluids import AIR, IdealGas
from .state import State
from .model.cycle_def import CycleDef, ParamDef, StepDef, get_cycle, list_cycles, register_cycle
from . import catalogue  # noqa: F401  (populates the cycle registry)
from .results import CycleResults, StepResult
from .cycle_state import CycleState
from .cycle import Cycle, describe, requirements
from .margins import Margin, MarginReport, SecondLaw
from .optimise import OptimisationResult, optimise
from .plotting import plot_cycle, plot_cycles   # matplotlib is imported lazily inside these functions
from . import custom   # noqa: E402  (custom cycles: CycleDef, StepDef, ParamDef, relation helpers)

__version__ = "0.1.0"
# The plan's final __all__ (first twelve names), then the other public names that earlier tasks exported.
__all__ = ["Cycle", "IdealGas", "AIR", "requirements", "describe", "list_cycles", "register_cycle", "optimise",
           "plot_cycles", "errors", "units", "__version__",
           "State", "StepResult", "CycleResults", "CycleState", "get_cycle", "plot_cycle", "OptimisationResult",
           "Margin", "MarginReport", "SecondLaw", "CycleDef", "StepDef", "ParamDef", "custom"]
