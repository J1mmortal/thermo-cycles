"""Extra Task 7 tests for analyse / raise_for (spec 9.3, 9.4, 15.3).

Physics notes used below (cold-air Otto, gamma = 1.4):
- eta_th = 1 - r_c**(1 - gamma) depends on r_c only, so eta_th is fixed by r_c and cannot replace q_in.
- an ideal-gas cycle given only temperatures and ratios has no pressure scale: T1, r_c, q_in, T3 fix every
  temperature but leave P1 free, so P1 is not one of the conflicting inputs of (T1, P1, r_c, q_in, T3).
"""

import time
from dataclasses import replace

import pytest

from thermo_cycles.errors import OverdeterminedError, UnderdeterminedError, UnknownInputError
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.advice import Requirements, analyse, raise_for
from thermo_cycles.solver.numeric import solve_numeric
from thermo_cycles.solver.system import build_system

OTTO = build_system(get_cycle("otto"), AIR)


def test_requirements_is_frozen_with_plan_fields_first():
    req = analyse(OTTO, ["T1", "P1", "r_c", "q_in"])
    assert isinstance(req, Requirements)
    assert tuple(Requirements.__dataclass_fields__)[:7] == (
        "status", "given", "missing", "add_one_of", "example_sets", "drop_one_of", "determined")
    with pytest.raises(Exception):
        req.status = "under"


def test_ok_lists_what_the_inputs_fix_and_suggests_nothing():
    req = analyse(OTTO, ["T1", "P1", "r_c", "q_in"])
    assert (req.status, req.missing, req.add_one_of, req.example_sets, req.drop_one_of) == ("ok", 0, (), (), ())
    assert {"T2", "T3", "P4", "eta_th", "mep"} <= set(req.determined)
    assert not {"T1", "P1", "r_c", "q_in"} & set(req.determined)
    assert raise_for(req, "otto") is None


def test_names_are_resolved_and_deduplicated():
    req = analyse(OTTO, ["t1", "P1", "R_C", "r", "T1"])
    assert req.given == ("T1", "P1", "r_c")
    with pytest.raises(UnknownInputError):
        analyse(OTTO, ["rc"])


def test_eta_th_is_fixed_by_r_c_and_never_suggested():
    req = analyse(OTTO, ["T1", "P1", "r_c"])
    assert "eta_th" in req.determined and "eta_th" not in req.add_one_of


def test_eta_th_with_r_c_is_over_determined():
    # Structurally square, but eta_th and r_c carry the same information: the solver would return a
    # zero-heat cycle (q_in = 0). Must be reported, not solved.
    req = analyse(OTTO, ["T1", "P1", "r_c", "eta_th"])
    assert req.status == "over" and req.missing == 1
    assert set(req.conflicting) == {"r_c", "eta_th"} == set(req.drop_one_of)
    with pytest.raises(OverdeterminedError) as ei:
        raise_for(req, "otto")
    assert "r_c, eta_th are not independent" in str(ei.value)
    assert ei.value.conflicting == ("r_c", "eta_th")


def test_pressure_scale_input_is_not_a_conflict():
    req = analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"])
    assert set(req.drop_one_of) == {"T1", "r_c", "q_in", "T3"}     # dropping P1 leaves P free
    # the purely structural analysis (spec 9.3 Dulmage-Mendelsohn) cannot see this and offers P1 too
    s = analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"], numeric=False)
    assert s.status == "over" and "P1" in s.drop_one_of


def test_structural_fallback_still_meets_the_plan_cases():
    assert analyse(OTTO, ["T1", "P1", "r_c", "q_in"], numeric=False).status == "ok"
    req = analyse(OTTO, ["T1", "P1", "r_c"], numeric=False)
    assert req.missing == 1 and req.add_one_of[0] == "T3" and "T2" in req.determined
    assert ("r_c", "T3") in analyse(OTTO, ["T1", "P1"], numeric=False).example_sets
    assert {"T1", "u1"} <= set(analyse(OTTO, ["T1", "u1", "P1", "r_c", "q_in"], numeric=False).drop_one_of)


def test_exact_under_message_one_missing():
    with pytest.raises(UnderdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "P1", "r_c"]), "otto")
    assert str(ei.value) == (
        "otto needs 1 more input.\n"
        "  Given: T1, P1, r_c  (these already fix: v1, T2, P2, v2, v3, v4, eta_th)\n"
        "  Add one of: T3, q_in, P3, T4, P4, q_out, w_net, mep\n"
        "  See thermo_cycles.requirements('otto', ...) for the full list.")
    e = ei.value
    assert e.missing == 1 and e.given == ("T1", "P1", "r_c") and e.add_one_of[0] == "T3"
    assert "T2" in e.determined and e.example_sets == ()


def test_under_message_two_missing_has_examples():
    req = analyse(OTTO, ["T1", "P1"])
    assert req.example_sets[0] == ("r_c", "T3") and len(req.example_sets) <= 6
    for pair in req.example_sets:
        assert analyse(OTTO, ["T1", "P1", *pair]).status == "ok"
    for c in req.add_one_of:
        assert any(analyse(OTTO, ["T1", "P1", c, d]).status == "ok" for d in OTTO.candidates if d != c)
    with pytest.raises(UnderdeterminedError) as ei:
        raise_for(req, "otto")
    msg = str(ei.value)
    assert msg.startswith("otto needs 2 more inputs.\n  Given: T1, P1  (these already fix: v1, v4)\n")
    assert "\n  For example: (r_c, T3) or (" in msg and msg.endswith(
        "\n  See thermo_cycles.requirements('otto', ...) for the full list.")


def test_three_and_four_missing_greedy_and_typical_sets():
    req = analyse(OTTO, ["T1"])
    assert req.missing == 3 and "r_c" in req.add_one_of
    assert all(analyse(OTTO, ["T1", *ex]).status == "ok" for ex in req.example_sets)
    assert ("P1", "r_c", "T3") in req.example_sets           # typical_inputs superset, given part removed
    req0 = analyse(OTTO, [])
    assert req0.missing == 4 and ("T1", "P1", "r_c", "T3") in req0.example_sets
    with pytest.raises(UnderdeterminedError, match=r"needs 4 more inputs\.\n  Given: \(none\)\n"):
        raise_for(req0, "otto")


def test_exact_over_message():
    with pytest.raises(OverdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "u1", "P1", "r_c", "q_in"]), "otto")
    assert str(ei.value) == (
        "otto is over-determined: T1, u1 are not independent\n"
        "  (2 inputs for 1 degrees of freedom in that part of the cycle).\n"
        "  Remove one of: T1, u1  -- or pass allow_redundant=True if the values are consistent.")
    assert ei.value.drop_one_of == ("T1", "u1")


def test_two_independent_redundancies_ask_to_remove_two():
    req = analyse(OTTO, ["T1", "u1", "P1", "r_c", "eta_th", "q_in"])
    assert req.status == "over"
    assert set(req.conflicting) == {"T1", "u1", "r_c", "eta_th"}
    with pytest.raises(OverdeterminedError) as ei:
        raise_for(req, "otto")
    assert "(4 inputs for 2 degrees of freedom" in str(ei.value)
    assert "Remove one of each group: (r_c, eta_th), (T1, u1)  -- or pass allow_redundant=True" in str(ei.value)


def test_every_one_input_suggestion_actually_solves():
    ref, _ = solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1.8e6})
    req = analyse(OTTO, ["T1", "P1", "r_c"])
    for c in req.add_one_of:
        vals, _ = solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "r_c": 8.0, c: ref[c]})
        assert vals["q_in"] == pytest.approx(1.8e6, rel=1e-6), c


def test_cycle_without_typical_inputs_still_gets_the_numeric_refinement():
    bare = build_system(replace(get_cycle("otto"), typical_inputs=()), AIR)
    req = analyse(bare, ["T1", "P1", "r_c"])
    assert "eta_th" in req.determined and analyse(bare, ["T1", "P1", "r_c", "eta_th"]).status == "over"


def test_pair_scan_is_fast():
    sys2 = build_system(get_cycle("otto"), AIR)       # fresh system: includes the reference solves
    t = time.perf_counter()
    analyse(sys2, ["T1", "P1"])
    assert time.perf_counter() - t < 2.0
