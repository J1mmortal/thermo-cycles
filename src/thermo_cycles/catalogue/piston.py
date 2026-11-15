"""Piston (closed-system) cycles of the catalogue (spec section 8.2). v0.1 so far: Otto."""

from __future__ import annotations

import math

from ..model.cycle_def import CycleDef, ParamDef, StepDef, register_cycle
from ..model.relations import ratio

OTTO = CycleDef(
    name="otto", title="Ideal Otto cycle (air-standard)", aliases=(), system="closed",
    states=(1, 2, 3, 4),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
           StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2 (> 1)", ratio("r_c", "v1", "v2"),
                     bounds=(1.0, math.inf)),),
    v_max_state=1, v_min_state=2,
    suggest=("r_c", "T3", "q_in", "P3", "T4"),
    typical_inputs=(("T1", "P1", "r_c", "T3"), ("T1", "P1", "r_c", "q_in")),
    description="Isentropic compression, constant-volume heat addition, isentropic expansion, "
                "constant-volume heat rejection.",
    references=("Waterloo ME354 Tutorial wk8", "Cengel & Boles 7e SM 9-39"),
)

register_cycle(OTTO, replace=True)
