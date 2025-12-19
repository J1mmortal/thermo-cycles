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
                ("efficiency", {"T1", "T3_MAX", "R_C_MIN", "R_C_MAX", "P_RATIO", "P3_MAX"}, set()),
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
            if mode == "design":
                internal["T1"] = known["T1"]
                internal["P1"] = known["P1"]
                internal["R_C"] = known["R_C"]
                internal["R_E"] = known["R_E"]
                if "T3" in known:
                    internal["T3"] = known["T3"]
                elif "Q_IN" in known:
                    internal["Q_IN"] = known["Q_IN"]
                else:
                    raise ValueError("Atkinson design mode needs T3 or Q_IN")
            elif mode == "efficiency":
                internal["T1"] = known["T1"]
                internal["T3_MAX"] = known["T3_MAX"]
                internal["ETA_TARGET"] = known["ETA_TARGET"]
            else:
                raise NotImplementedError(f"Mode {mode} not implemented for Atkinson")

        #TODO: elif cycle_type == "otto": ...
        return internal
