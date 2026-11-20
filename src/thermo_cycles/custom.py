"""Everything needed to define your own cycle (spec 8.1), in one namespace (v0.1.1).

    from thermo_cycles.custom import CycleDef, StepDef, ParamDef, ratio, register_cycle

    register_cycle(CycleDef(
        name="three_step", title="isentropic, isobaric, isochoric", aliases=(), system="closed",
        states=(1, 2, 3),
        steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
               StepDef("isochoric", 3, 1, "heat_out")),
        params=(ParamDef("r_c", "compression ratio v1/v2", ratio("r_c", "v1", "v2")),),
        v_max_state=1, v_min_state=2))
    Cycle("three_step", T1=293, P1=1e5, r_c=5)

``CycleDef``, ``StepDef`` and ``ParamDef`` are also exported from ``thermo_cycles`` itself. Relation helpers:
``ratio(out, num, den)`` (out = num/den), ``linear(out, {var: coef})`` (out = sum coef*var), ``equal(a, b, name)``,
``product(out, a, b)``, ``fixed(var, value)`` and the general ``Relation``. A bad definition raises
``CycleDefinitionError`` (a ``ValueError``).
"""

from .model.cycle_def import CycleDef, ParamDef, StepDef, get_cycle, list_cycles, register_cycle
from .model.relations import Relation, equal, fixed, linear, product, ratio

__all__ = ["CycleDef", "StepDef", "ParamDef", "register_cycle", "get_cycle", "list_cycles",
           "Relation", "ratio", "linear", "equal", "product", "fixed"]
