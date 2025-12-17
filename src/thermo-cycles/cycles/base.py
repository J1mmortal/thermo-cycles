from typing import Any, Dict, Optional
from core.cycle_state import CycleState
from core.state import State
from core.input_interpreter import InputInterpreter
from cycles.atkinson import AtkinsonSolver


class BaseCycle:
    def __init__(self, gamma: float = 1.4, R: float = 287.0):
        self.gamma = gamma
        self.R = R
        self._last_cycle_state: Optional[CycleState] = None

    @property
    def states(self) -> Dict[int, State]:
        if self._last_cycle_state is None:
            raise RuntimeError("No solution yet. Call solve() first.")
        return self._last_cycle_state.states

    @property
    def results(self) -> Dict[str, float]:
        if self._last_cycle_state is None:
            raise RuntimeError("No solution yet. Call solve() first.")
        return self._last_cycle_state.results

class AtkinsonCycle(BaseCycle):
    def __init__(self, gamma: float = 1.4, R: float = 287.0):
        super().__init__(gamma=gamma, R=R)
        self._interpreter = InputInterpreter()
        self._solver = AtkinsonSolver(gamma=gamma, R=R)

    def solve(self, mode: Optional[str] = None, **kwargs: Any) -> Dict[str, float]:
        cs = self._interpreter.interpret(cycle_type="atkinson", mode=mode, **kwargs,)
        
        self._solver.solve(cs)
        self._last_cycle_state = cs

        return cs.results
