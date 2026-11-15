# tests/test_inputs.py  (Review Focus 1, 2, 4)
import math, warnings
import numpy as np, pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system, resolve_inputs
from thermo_cycles.errors import InputError, UnknownInputError, InputPlausibilityWarning

OTTO = build_system(get_cycle("otto"), AIR)

def test_names_case_and_alias():
    got = resolve_inputs(OTTO, {"t1": 300, "P1": np.float64(1e5), "R_C": 8, "Q_IN": 1e6})
    assert got == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6}
    assert resolve_inputs(OTTO, {"r": 8, "T1": 300})["r_c"] == 8.0

def test_unknown_name_suggests():
    with pytest.raises(UnknownInputError) as ei:
        resolve_inputs(OTTO, {"rc": 8})
    assert "r_c" in ei.value.suggestions

@pytest.mark.parametrize("bad", [math.nan, math.inf, -300.0, "300", 1j])
def test_bad_values(bad):
    with pytest.raises(InputError, match="T1"):
        resolve_inputs(OTTO, {"T1": bad})

def test_plausibility_warning_for_kpa_and_celsius():
    with pytest.warns(InputPlausibilityWarning, match="P1"):
        resolve_inputs(OTTO, {"P1": 100})
    with pytest.warns(InputPlausibilityWarning, match="T1"):
        resolve_inputs(OTTO, {"T1": 25})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_inputs(OTTO, {"T1": 300, "P1": 1e5})
