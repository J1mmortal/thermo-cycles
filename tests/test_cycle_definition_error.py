"""5e (v0.1.1): definition errors from register_cycle / build_system are CycleDefinitionError, a
ThermoCyclesError that is still a ValueError (T0091 open issue 3)."""

import pytest

from thermo_cycles import AIR, errors
from thermo_cycles.custom import CycleDef, ParamDef, StepDef, equal, linear, ratio, register_cycle
from thermo_cycles.errors import CycleDefinitionError, ThermoCyclesError
from thermo_cycles.solver import build_system


def _defn(**over):
    kw = dict(name="bad_def", title="t", aliases=(), system="closed", states=(1, 2, 3),
              steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
                     StepDef("isochoric", 3, 1, "heat_out")),
              params=(ParamDef("r_c", "v1/v2", ratio("r_c", "v1", "v2")),), v_max_state=1, v_min_state=2)
    kw.update(over)
    return CycleDef(**kw)


def test_class_in_hierarchy():
    assert issubclass(CycleDefinitionError, ThermoCyclesError)
    assert issubclass(CycleDefinitionError, ValueError)
    assert errors.CycleDefinitionError is CycleDefinitionError


def test_no_heat_in_step():
    steps = (StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_out"),
             StepDef("isochoric", 3, 1, "heat_out"))
    with pytest.raises(CycleDefinitionError, match="needs at least one step with role 'heat_in'"):
        register_cycle(_defn(steps=steps))


def test_still_a_value_error_for_old_code():
    with pytest.raises(ValueError):
        register_cycle(_defn(name="1 bad name"))


def test_not_a_cycledef():
    with pytest.raises(CycleDefinitionError, match="expected a CycleDef"):
        register_cycle("otto")


def test_name_clash():
    with pytest.raises(CycleDefinitionError, match="already registered"):
        register_cycle(_defn(name="otto"))


def test_build_system_unknown_variable():
    bad = _defn(extra_relations=(linear("T9", {"T1": 1.0}),))
    with pytest.raises(CycleDefinitionError, match="unknown variables"):
        build_system(bad, AIR)


def test_relation_helpers():
    with pytest.raises(CycleDefinitionError):
        equal("T1", "T1", "same")
    with pytest.raises(CycleDefinitionError):
        linear("T1", {})
