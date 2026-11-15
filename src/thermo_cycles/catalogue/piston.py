"""Piston (closed-system) cycles of the catalogue (spec section 8.2). v0.1: Otto, Diesel, Dual, Atkinson."""

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

DIESEL = CycleDef(
    name="diesel", title="Ideal Diesel cycle (air-standard)", aliases=(), system="closed",
    states=(1, 2, 3, 4),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
           StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2 (> 1)", ratio("r_c", "v1", "v2"),
                     bounds=(1.0, math.inf)),
            ParamDef("cutoff", "cut-off ratio v3/v2 (>= 1)", ratio("cutoff", "v3", "v2"),
                     bounds=(1.0, math.inf))),
    v_max_state=1, v_min_state=2,
    suggest=("r_c", "cutoff", "T3", "q_in"),
    typical_inputs=(("T1", "P1", "r_c", "cutoff"), ("T1", "P1", "r_c", "T3"), ("T1", "P1", "r_c", "q_in")),
    description="Isentropic compression, constant-pressure heat addition, isentropic expansion, "
                "constant-volume heat rejection.",
    references=("Cengel & Boles 8e SM 9-57", "Cengel & Boles 7e SM 9-58 (states 1-3)"),
)

DUAL = CycleDef(
    name="dual", title="Ideal dual (limited-pressure) cycle (air-standard)", aliases=(), system="closed",
    states=(1, 2, 3, 4, 5),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
           StepDef("isobaric", 3, 4, "heat_in"), StepDef("isentropic", 4, 5, "expansion"),
           StepDef("isochoric", 5, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2 (> 1)", ratio("r_c", "v1", "v2"),
                     bounds=(1.0, math.inf)),
            ParamDef("alpha", "constant-volume pressure ratio P3/P2 (>= 1)", ratio("alpha", "P3", "P2"),
                     bounds=(1.0, math.inf)),
            ParamDef("cutoff", "cut-off ratio v4/v3 (>= 1)", ratio("cutoff", "v4", "v3"),
                     bounds=(1.0, math.inf))),
    v_max_state=1, v_min_state=2,
    suggest=("r_c", "alpha", "cutoff", "T4", "q_in", "P3"),
    typical_inputs=(("T1", "P1", "r_c", "alpha", "cutoff"), ("T1", "P1", "r_c", "P3", "T4"),
                    ("T1", "P1", "r_c", "alpha", "q_in")),
    description="Isentropic compression, constant-volume then constant-pressure heat addition, isentropic "
                "expansion, constant-volume heat rejection. Cengel's state x is our 3, his 3 our 4, his 4 our 5.",
    references=("Cengel & Boles 7e SM 9-54",),
)

ATKINSON = CycleDef(
    name="atkinson", title="Ideal Atkinson cycle (air-standard)", aliases=("atk",), system="closed",
    states=(1, 2, 3, 4),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
           StepDef("isentropic", 3, 4, "expansion"), StepDef("isobaric", 4, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2 (> 1)", ratio("r_c", "v1", "v2"),
                     bounds=(1.0, math.inf)),
            ParamDef("r_e", "expansion ratio v4/v3 (> r_c)", ratio("r_e", "v4", "v3"),
                     bounds=(1.0, math.inf))),
    v_max_state=4, v_min_state=2,
    suggest=("r_c", "T3", "P3", "q_in", "r_e"),
    typical_inputs=(("T1", "P1", "r_c", "T3"), ("T1", "P1", "r_c", "q_in"), ("T1", "P1", "r_c", "r_e")),
    description="Isentropic compression, constant-volume heat addition, isentropic expansion to P1, "
                "constant-pressure heat rejection.",
    references=("Rajput, Engineering Thermodynamics, sec. 13.8 and Example 13.31",),
)

register_cycle(OTTO, replace=True)
register_cycle(DIESEL, replace=True)
register_cycle(DUAL, replace=True)
register_cycle(ATKINSON, replace=True)
