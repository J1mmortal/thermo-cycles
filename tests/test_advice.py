import pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system
from thermo_cycles.solver.advice import analyse, raise_for
from thermo_cycles.errors import UnderdeterminedError, OverdeterminedError

OTTO = build_system(get_cycle("otto"), AIR)

def test_ok():
    assert analyse(OTTO, ["T1", "P1", "r_c", "q_in"]).status == "ok"

def test_one_missing_options_exclude_determined():
    req = analyse(OTTO, ["T1", "P1", "r_c"])
    assert req.status == "under" and req.missing == 1
    assert {"T3", "q_in", "P3", "T4"} <= set(req.add_one_of)
    assert "T2" not in req.add_one_of and "T2" in req.determined and "u3" not in req.add_one_of
    assert req.add_one_of[0] == "T3"            # suggest order after r_c (already given)

def test_two_missing_examples():
    req = analyse(OTTO, ["T1", "P1"])
    assert req.missing == 2 and ("r_c", "T3") in req.example_sets

def test_over_determined_names_conflict():
    req = analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"])
    assert req.status == "over" and {"q_in", "T3"} <= set(req.drop_one_of)
    req2 = analyse(OTTO, ["T1", "u1", "P1", "r_c", "q_in"])
    assert {"T1", "u1"} <= set(req2.drop_one_of)

def test_message_template():
    with pytest.raises(UnderdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "P1", "r_c"]), "otto")
    msg = str(ei.value)
    assert msg.startswith("otto needs 1 more input.")
    assert "Given: T1, P1, r_c" in msg and "Add one of: T3, " in msg
    with pytest.raises(OverdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"]), "otto")
    assert "is over-determined" in str(ei.value) and "allow_redundant=True" in str(ei.value)
