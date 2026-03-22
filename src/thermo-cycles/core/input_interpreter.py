from typing import Optional
from core.cycle_state import CycleState
from typing import Dict, Any, Optional
from core.state import State

class InputInterpreter:
    def __init__(self):
        self._patterns = {
            "atkinson": [
                #each pattern: (mode_name, required_keys, optional_keys)
                ("design", {"T1", "P1", "R_C"}, {"T3", "Q_IN"}),
                ("efficiency", {"T1", "T3_MAX", "R_C_MIN", "R_C_MAX"}, {"P_RATIO", "P2_MAX"}),
            ]
        }

    def interpret(self, cycle_type: str, mode: Optional[str] = None, **kwargs) -> CycleState:
        known = {k.upper(): v for k, v in kwargs.items() if v is not None}
        cs = CycleState(cycle_type=cycle_type, known=known)
        if mode is not None:
            cs.mode = mode
            cs.internal = self._build_internal_for_mode(cycle_type, mode, known)
            cs.validate()
            return cs

        patterns = self._patterns.get(cycle_type, [])
        matched_mode = None
        for m_name, required, optional in patterns:
            if required.issubset(known.keys()):
                matched_mode = m_name
                break

        if matched_mode is None:
            raise ValueError(f"Could not infer mode for cycle_type={cycle_type} from known keys={list(known.keys())}")

        cs.mode = matched_mode
        cs.internal = self._build_internal_for_mode(cycle_type, matched_mode, known)
        cs.validate()
        return cs

    def _build_internal_for_mode(self, cycle_type: str, mode: str, known: Dict[str, Any]) -> Dict[str, Any]:
        internal: Dict[str, Any] = {}

        if cycle_type == "atkinson":
            if mode == "DESIGN":
                internal["T1"] = known["T1"]
                internal["P1"] = known["P1"]
                internal["R_C"] = known["R_C"]
                internal["R_E"] = known["R_E"]
                if "T3" in known:
                    internal["T3"] = known["T3"]
                elif "Q_IN" in known:
                    internal["Q_IN"] = known["Q_IN"]
                else:
                    raise ValueError("Atkinson DESIGN mode needs T3 or Q_IN")
            elif mode == "EFFICIENCY":
                internal["T1"] = known["T1"]
                internal["T3_MAX"] = known["T3_MAX"]
                internal["R_C_MIN"] = known["R_C_MIN"]
                internal["R_C_MAX"] = known["R_C_MAX"]
                if "P_RATIO" in known:
                    internal["P_RATIO"] = known["P_RATIO"]
                elif "P2_MAX" in known:
                    internal["P2_MAX"] = known["P2_MAX"]
                else:
                    raise ValueError("Atkinson EFFICIENCY mode needs P_RATIO or P2_MAX")
            else:
                raise NotImplementedError(f"Mode {mode} not implemented for Atkinson")

        #TODO: elif cycle_type == "otto": ...
        return internal
