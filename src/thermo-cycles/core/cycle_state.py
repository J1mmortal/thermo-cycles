from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from core.state import State

@dataclass
class CycleState:
    """Container for all data related to a single cycle solution."""
    cycle_type: str
    mode: Optional[str] = None
    known: Dict[str, Any] = field(default_factory=dict)
    internal: Dict[str, Any] = field(default_factory=dict)
    states: Dict[int, State] = field(default_factory=dict)
    results: Dict[str, float] = field(default_factory=dict)

    def add_state(self, sid: int, **props: float) -> State:
        st = State(state_id=sid)
        if props:
            st.set(**props)
        self.states[sid] = st
        return st

    def validate(self) -> None:
        if not self.cycle_type:
            raise ValueError("cycle_type must be set in CycleState")
        if not self.mode:
            raise ValueError("mode must be set in CycleState")

    def reset_states(self) -> None:
        self.states.clear()
        self.results.clear()
