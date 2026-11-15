"""The solution container (spec section 5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model.cycle_def import CycleDef
from .results import CycleResults, StepResult
from .solver.numeric import SolveReport
from .state import State

__all__ = ["CycleState"]


@dataclass(frozen=True)
class CycleState:
    """One solved cycle: definition, fluid, the inputs as given (canonical names), every solved variable, the
    states, steps and results, and the solver report (blocks, method per block, max residual, redundant
    checks). Mappings are read-only views."""

    definition: CycleDef
    fluid: object
    inputs: Mapping[str, float]            # canonical names, as given
    values: Mapping[str, float]            # every solved variable of the equation system
    states: Mapping[int, State]
    hidden_states: Mapping[str, State]
    steps: "tuple[StepResult, ...]"
    results: CycleResults
    report: SolveReport                    # blocks, method per block (explicit/torn/newton), redundant checks
