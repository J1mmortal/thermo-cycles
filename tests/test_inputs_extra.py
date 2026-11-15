"""Extra Task 7 tests for resolve_inputs (spec 3 look-up order, 9.8 validation, review focus 1, 2, 4)."""

import math
import warnings

import numpy as np
import pytest

from thermo_cycles.errors import InputError, InputPlausibilityWarning, UnknownInputError
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import CycleDef, ParamDef, StepDef, get_cycle
from thermo_cycles.model.relations import ratio
from thermo_cycles.solver.system import build_system, resolve_inputs

OTTO = build_system(get_cycle("otto"), AIR)

# A Diesel-shaped test cycle (the catalogue Diesel arrives in Task 9): both r_c and cutoff exist.
_DIESELISH = CycleDef(
    name="dieselish", title="test", aliases=(), system="closed", states=(1, 2, 3, 4),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
           StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio", ratio("r_c", "v1", "v2"), bounds=(1.0, math.inf)),
            ParamDef("cutoff", "cutoff ratio", ratio("cutoff", "v3", "v2"), bounds=(1.0, math.inf))),
    v_max_state=1, v_min_state=2)
DIESELISH = build_system(_DIESELISH, AIR)


def test_values_are_plain_floats_in_input_order():
    got = resolve_inputs(OTTO, {"r_c": np.int64(8), "T1": np.float32(300.0), "P1": 100_000})
    assert list(got) == ["r_c", "T1", "P1"]
    assert all(type(v) is float for v in got.values())


@pytest.mark.parametrize("bad", [True, None, np.array([300.0]), np.complex128(300), [300.0]])
def test_non_real_values_name_the_input(bad):
    with pytest.raises(InputError, match="T1"):
        resolve_inputs(OTTO, {"T1": bad})


def test_zero_for_positive_kind_rejected_but_signed_may_be_negative():
    with pytest.raises(InputError, match=r"P1 .*> 0"):
        resolve_inputs(OTTO, {"P1": 0})
    assert resolve_inputs(OTTO, {"w_12": -3e5})["w_12"] == -3e5


@pytest.mark.parametrize("name,value", [("T4", 2e5), ("P1", 1e11), ("r_c", 0.5), ("eta_th", 1.2),
                                        ("v1", 1e5), ("q_in", 5e9)])
def test_values_outside_the_solvable_range_raise_naming_the_input(name, value):
    # T0078 open issue 2: T4 = 2e5 K used to reach the solver and blame r_c.
    with pytest.raises(InputError, match=name):
        resolve_inputs(OTTO, {name: value})


def test_unknown_input_error_carries_name_and_accepted():
    with pytest.raises(UnknownInputError) as ei:
        resolve_inputs(OTTO, {"T9": 300})
    assert ei.value.name == "T9" and "T1" in ei.value.accepted


def test_alias_is_exact_only_so_R_is_not_taken_for_r_c():
    # "R" is much more likely the gas constant than the old compression-ratio alias "r".
    with pytest.raises(UnknownInputError):
        resolve_inputs(OTTO, {"R": 287})


def test_rc_suggests_both_meanings_where_both_exist():
    # Review focus 4: Cengel & Boles call the cutoff ratio r_c.
    with pytest.raises(UnknownInputError) as ei:
        resolve_inputs(DIESELISH, {"rc": 2})
    assert {"r_c", "cutoff"} <= set(ei.value.suggestions)
    assert "cutoff" in str(ei.value)


def test_post_solve_quantities_are_not_inputs():
    with pytest.raises(UnknownInputError, match="computed after the solve"):
        resolve_inputs(OTTO, {"eta_carnot": 0.7})


def test_two_spellings_of_one_input_raise():
    with pytest.raises(InputError, match=r"T1.*t1|t1.*T1"):
        resolve_inputs(OTTO, {"T1": 300, "t1": 300})
    with pytest.raises(InputError, match="r_c"):
        resolve_inputs(OTTO, {"r": 8, "r_c": 8})


def test_non_string_name_rejected():
    with pytest.raises(InputError):
        resolve_inputs(OTTO, {1: 300})


def test_plausibility_messages():
    with pytest.warns(InputPlausibilityWarning) as rec:
        resolve_inputs(OTTO, {"P1": 100}, "warn")
    assert str(rec[0].message).startswith("P1 = 100 Pa: did you mean 100 kPa? thermo_cycles works in SI")
    with pytest.warns(InputPlausibilityWarning) as rec:
        resolve_inputs(OTTO, {"T1": 25}, "warn")
    assert str(rec[0].message).startswith(
        "T1 = 25 K: did you mean 25 deg C (298.15 K)? thermo_cycles works in SI")
    # boundaries: exactly 1000 Pa and 100 K do not warn; the check is for T and P only
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_inputs(OTTO, {"P3": 1000.0, "T3": 100.0, "v1": 0.5, "q_in": 50.0, "r_c": 8})


def test_plausibility_applies_to_every_state():
    with pytest.raises(InputError, match="P3 = 500 Pa"):
        resolve_inputs(OTTO, {"P3": 500.0})
    with pytest.warns(InputPlausibilityWarning, match="P3"):
        resolve_inputs(OTTO, {"P3": 500.0}, "warn")
