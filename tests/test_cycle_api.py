# tests/test_cycle_api.py  (incl. Review Focus 3)
import dataclasses, pytest
from thermo_cycles import Cycle, requirements, AIR
from thermo_cycles.errors import (InvalidCycleError, UnderdeterminedError, InconsistentInputError,
                                  OverdeterminedError, CycleValidityWarning)

def test_states_params_value():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    assert c.states[1].T == 300 and c.params["r_c"] == 8 and c.value("P1") == 1e5
    assert c.inputs == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6} and c.fluid is AIR

def test_with_inputs_does_not_mutate():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    c2 = c.with_inputs(r_c=10)
    assert c.params["r_c"] == 8 and c2.params["r_c"] == 10
    c3 = c.with_inputs(q_in=None, T3=2000)
    assert "q_in" not in c3.inputs and c3.states[3].T == pytest.approx(2000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.states[1].T = 1

def test_heat_addition_below_T2_names_step():          # Review Focus 3
    with pytest.raises(InvalidCycleError, match=r"2->3"):
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600)
    with pytest.warns(CycleValidityWarning):
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600, check="warn")

def test_underdetermined_from_front():
    with pytest.raises(UnderdeterminedError, match="Add one of: T3"):
        Cycle("otto", T1=300, P1=1e5, r_c=10)
    assert requirements("otto", T1=300, P1=1e5, r_c=10).missing == 1

def test_redundant_consistent_and_inconsistent():
    base = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    with pytest.raises(OverdeterminedError):
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=base.states[3].T)
    ok = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=base.states[3].T, allow_redundant=True)
    assert ok.results.eta_th == pytest.approx(base.results.eta_th)
    with pytest.raises(InconsistentInputError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=2000, allow_redundant=True)
    assert ei.value.name in {"T3", "q_in"} and ei.value.given != pytest.approx(ei.value.implied)
