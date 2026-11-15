"""Task 8 extra tests: validity rules (spec 9.7), results (spec 10), solve_cycle order (spec 9.4-9.6), the Cycle
front (spec 4) and the package exports. Numbers are hand calculations with AIR (R 287, cv 717.5, cp 1004.5)."""
import dataclasses
import json
import math
import time
import warnings

import pytest

import thermo_cycles as tc
from thermo_cycles import AIR, Cycle, get_cycle, requirements
from thermo_cycles.cycle_state import CycleState
from thermo_cycles.errors import (AmbiguousSolutionError, CycleValidityWarning, InconsistentInputError, InputError,
                                  InputPlausibilityWarning, InvalidCycleError, OverdeterminedError,
                                  UnknownInputError)
from thermo_cycles.results import CycleResults, StepResult, compute_results, validity_problems
from thermo_cycles.solver import solve_cycle
from thermo_cycles.units import kPa

OTTO = get_cycle("otto")


def _values(c):
    return dict(c.cycle_state.values)


def _problems(defn, vals):
    return validity_problems(defn, vals, compute_results(defn, AIR, vals)[2])


# --- spec 4 / 9.5 examples --------------------------------------------------------------------------------

def test_spec4_example_numbers():
    # hand: T2 = 288*10^0.4 = 723.42 K; T3 = T2 + 1.8e6/717.5 = 3232.1 K; v3 = 287*288/1e5/10 = 0.082656 m3/kg;
    # P3 = 287*3232.1/0.082656 = 1.1223e7 Pa; eta = 1 - 10^-0.4 = 0.60189; mep = eta*q_in/(0.9 v1) = 1.4564e6 Pa
    c = Cycle("otto", T1=288, P1=100 * kPa, r_c=10, q_in=1800e3)
    assert c.states[3].T == pytest.approx(3232.1, rel=1e-4) and c.states[3].P == pytest.approx(1.1223e7, rel=1e-4)
    s = c.steps[1]
    assert (s.a, s.b, s.kind, s.role) == (2, 3, "isochoric", "heat_in")
    assert s.q == pytest.approx(1.8e6, rel=1e-12) and s.w == pytest.approx(0.0, abs=1e-6) and s.hidden is None
    assert c.results.eta_th == pytest.approx(1 - 10 ** -0.4, rel=1e-12)
    assert c.results.mep == pytest.approx(1.4564e6, rel=1e-4)


def test_O3_loop_from_front():
    # spec 15.1 O3: P1 solved through an algebraic loop; 99 943 Pa by hand (1e5 * 11216.6/11223.0)
    c = Cycle("otto", T1=288, r_c=10, q_in=1800e3, P3=11216.6e3)
    assert c.value("P1") == pytest.approx(1e5, rel=1e-3)
    assert set(c.cycle_state.report.methods) & {"torn", "newton"}


def test_two_valid_roots_stay_ambiguous_from_front():
    # spec 9.5 example "r_c = 2.736 or 20.43" (T1 300, T3 1500, w_net 250 kJ/kg); hand: 717.5 (1500 - 300 x)(1 - 1/x)
    # = 250e3 at x = r_c^0.4 = 1.4964 and 3.343
    with pytest.raises(AmbiguousSolutionError) as ei:
        Cycle("otto", T1=300, P1=1e5, T3=1500, w_net=250e3)
    assert ei.value.variable == "r_c"
    assert sorted(ei.value.candidates) == [pytest.approx(2.736, rel=3e-3), pytest.approx(20.43, rel=3e-3)]


# --- results (spec 5.3, 10) -----------------------------------------------------------------------------

def test_compute_results_fields():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    states, hidden, steps, r = compute_results(OTTO, AIR, c.cycle_state.values)
    assert set(states) == {1, 2, 3, 4} and hidden == {} and len(steps) == 4
    assert all(isinstance(s, StepResult) for s in steps) and isinstance(r, CycleResults)
    assert r.w_compression == pytest.approx(steps[0].w) and r.w_expansion == pytest.approx(steps[2].w)
    assert r.T_min == 300 and r.T_max == pytest.approx(1800) and r.P_min == 1e5 and r.P_max == states[3].P
    assert r.v_min == pytest.approx(states[2].v) and r.v_max == pytest.approx(states[1].v)
    assert r.eta_carnot == pytest.approx(1 - 300 / 1800) and 0 < r.eta_ii < 1
    assert r.mep == pytest.approx(r.w_net / (states[1].v - states[2].v)) and r.bwr is None
    assert set(r.as_dict()) == {f.name for f in dataclasses.fields(CycleResults)}
    # sign conventions (spec 3): q_in, q_out positive magnitudes
    assert r.q_in == pytest.approx(steps[1].q) and r.q_out == pytest.approx(-steps[3].q)


def test_cycle_state_is_frozen_with_readonly_mappings():
    cs = solve_cycle(OTTO, AIR, {"T1": 300, "P1": 1e5, "r_c": 8, "q_in": 1e6})
    assert isinstance(cs, CycleState) and cs.report.redundant == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.results = None
    with pytest.raises(TypeError):
        cs.values["T1"] = 1.0
    with pytest.raises(TypeError):
        cs.states[1] = None


# --- validity (spec 9.7) --------------------------------------------------------------------------------

def test_review_focus_3_message_lists_every_problem():
    # hand: T2 = 300*10^0.4 = 753.6 K > T3 = 600 K -> q_23 = 717.5 (600 - 753.6) = -110.2 kJ/kg;
    # T4 = 600/10^0.4 = 238.9 K < T1 -> q_41 = 717.5 (300 - 238.9) = +43.9 kJ/kg
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600)
    p = ei.value.problems
    assert p[0] == ("step 2->3 (isochoric, heat_in) has q = -110.2 kJ/kg < 0: T3 = 600 K is below T2 = 753.57 K. "
                    "Raise T3 / q_in or lower r_c.")
    assert any(m.startswith("step 4->1 (isochoric, heat_out) has q = +43.9 kJ/kg > 0") for m in p)
    assert any(m.startswith("q_in = -110.2 kJ/kg is not > 0") for m in p)
    assert all(m in str(ei.value) for m in p)


def test_check_warn_and_off():
    with pytest.warns(CycleValidityWarning, match="step 2->3"):
        c = Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600, check="warn")
    assert c.steps[1].q < 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600, check="off")
    assert c.results.q_in < 0
    with pytest.raises(InputError, match="check"):
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=1600, check="loud")


def test_heat_in_zero_is_valid_within_tolerance():
    # spec 9.7: q >= -1e-9 q_in on heat_in, q <= 1e-9 q_in on heat_out (zero allowed: Dual limit cases)
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    vals = _values(c)
    q_in = vals["q_in"]
    for q, bad in ((0.0, False), (-0.5e-9 * q_in, False), (-2e-9 * q_in, True)):
        v = dict(vals, q_23=q)
        assert any(m.startswith("step 2->3") for m in _problems(OTTO, v)) is bad, q
    for q, bad in ((0.0, False), (0.5e-9 * q_in, False), (2e-9 * q_in, True)):
        v = dict(vals, q_41=q)
        assert any(m.startswith("step 4->1") for m in _problems(OTTO, v)) is bad, q


def test_compression_expansion_state_and_bound_rules():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    vals = _values(c)
    assert _problems(OTTO, vals) == []
    p = _problems(OTTO, dict(vals, P2=0.9e5, w_12=1.0))
    assert any("step 1->2 (isentropic, compression) has P2 = 90.0 kPa <= P1 = 100.0 kPa" in m for m in p)
    assert any("step 1->2 (isentropic, compression) has w = 0.0 kJ/kg >= 0" in m for m in p)
    p = _problems(OTTO, dict(vals, P4=vals["P3"] * 2, w_34=-5.0))
    assert any("step 3->4 (isentropic, expansion) has P4" in m and ">= P3" in m for m in p)
    assert any("step 3->4 (isentropic, expansion) has w = -0.0 kJ/kg <= 0" in m for m in p)
    p = _problems(OTTO, dict(vals, T2=-1.0, v4=math.nan))
    assert "state 2: T2 = -1 K is not > 0." in p and "state 4: v4 = nan m^3/kg is not > 0." in p
    p = _problems(OTTO, dict(vals, r_c=0.5))
    assert any(m.startswith("parameter r_c = 0.5 is outside its bounds [1, inf]") for m in p)


# --- solve_cycle order (spec 9.4-9.6) ---------------------------------------------------------------------

def test_degenerate_input_set_is_caught_before_the_solve():
    # Otto eta_th depends on r_c alone: (T1, P1, r_c, eta_th) is over-determined, not a zero-heat cycle (T0079)
    eta = 1 - 8 ** -0.4
    with pytest.raises(OverdeterminedError, match="r_c"):
        Cycle("otto", T1=300, P1=1e5, r_c=8, eta_th=eta)


def test_former_hanging_set_fails_fast_from_front():
    # T0079 open issue 7 set; analyse reports it before any numeric solve
    base = Cycle("otto", T1=310, P1=1.2e5, r_c=8.5, q_in=1.5e6)
    t0 = time.perf_counter()
    with pytest.raises(OverdeterminedError):
        Cycle("otto", **{k: base.value(k) for k in ("q_in", "T2", "q_out", "eta_th")})
    assert time.perf_counter() - t0 < 5.0


def test_redundant_drop_order_and_message():
    base = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    ok = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=base.states[3].T, allow_redundant=True)
    ((name, given, implied),) = ok.cycle_state.report.redundant
    assert name == "T3" and implied == pytest.approx(given, rel=1e-9)
    assert ok.inputs == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6, "T3": base.states[3].T}
    # the input given last is the one checked: here q_in, with T3 given before it
    with pytest.raises(InconsistentInputError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=8, T3=2000, q_in=1e6, allow_redundant=True)
    e = ei.value
    # hand: q_in = 717.5 (2000 - 300*8^0.4) = 717.5 * 1310.8 = 940.5 kJ/kg
    assert e.name == "q_in" and e.given == 1e6 and e.implied == pytest.approx(940.5e3, rel=1e-3)
    assert str(e).startswith("otto: q_in = 1000 kJ/kg given, but T1, P1, r_c, T3 imply q_in = 940 kJ/kg.")


def test_redundant_within_tolerance_accepted():
    base = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    T3 = base.states[3].T
    Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=T3 * (1 + 0.5e-6), allow_redundant=True)
    with pytest.raises(InconsistentInputError):
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=T3 * (1 + 2e-6), allow_redundant=True)


# --- the Cycle front (spec 4) -----------------------------------------------------------------------------

def test_cycle_is_immutable():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.fluid = None
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.extra = 1
    with pytest.raises(TypeError):
        c.inputs["T1"] = 1.0
    with pytest.raises(TypeError):
        c.params["r_c"] = 1.0


def test_names_resolved_like_inputs():
    c = Cycle("OTTO", t1=300, P1=1e5, R_C=8, Q_IN=1e6)
    assert c.name == "otto" and dict(c.inputs) == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6}
    assert c.value("p4") == c.states[4].P and c.value("ETA_CARNOT") == c.results.eta_carnot
    assert c.value("T_max") == c.results.T_max and c.value("eta_th") == c.results.eta_th
    with pytest.raises(UnknownInputError):
        c.value("T9")
    c2 = c.with_inputs(r=10)                      # alias table r -> r_c
    assert c2.params["r_c"] == 10 and list(c2.inputs) == ["T1", "P1", "r_c", "q_in"]
    with pytest.raises(InputError, match="not an input"):
        c.with_inputs(T3=None)


def test_with_inputs_keeps_options():
    c = Cycle("otto", T1=300, P1=1e5, r_c=10, T3=1600, check="warn")
    with pytest.warns(CycleValidityWarning):
        c2 = c.with_inputs(T3=600)
    assert c2.states[3].T == pytest.approx(600)


def test_plausibility_warning_from_front():
    # spec §3/§9.8: a sub-1000 Pa input is physically solvable, so it warns loudly but still solves
    with pytest.warns(InputPlausibilityWarning, match="P1 = 100 Pa"):
        c = Cycle("otto", T1=300, P1=100, r_c=8, q_in=1e6)
    assert c.states[1].P == 100 and c.results.eta_th == pytest.approx(1 - 8 ** -0.4, rel=1e-9)


def test_requirements_front():
    r = requirements("otto", T1=300, P1=1e5, r_c=10, T3=1800)
    assert r.status == "ok" and r.missing == 0
    r = requirements("otto", T1=300, P1=1e5)
    assert r.status == "under" and r.missing == 2
    with pytest.raises(UnknownInputError):
        requirements("otto", T1=300, Q=1)
    with pytest.raises(InputError):
        requirements("otto", T1=float("nan"))


def test_margins_plot_limits_placeholders():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    m = c.margins
    assert len(m) == 0 and m.ok and m.violated == () and m.worst is None
    assert c.plot("Ts") is not None            # plotting is implemented (Task 13); see tests/test_plotting.py
    # limits are real since Task 11 (details in tests/test_margins.py)
    lim = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, limits={"T_max": 5000})
    assert lim.margins.ok and lim.margins["T_max"].value == pytest.approx(lim.results.T_max)
    with pytest.raises(InputError, match="on_violation"):
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, on_violation="shout")


def test_summary_and_to_dict():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    text = c.summary()
    for part in ("otto", "2->3", "isochoric", "eta_th", "mep", "r_c = 8", "T1 = 300"):
        assert part in text, part
    d = c.to_dict()
    assert d["name"] == "otto" and d["inputs"]["r_c"] == 8.0 and d["params"] == {"r_c": 8.0}
    assert d["states"][3]["T"] == c.states[3].T and d["steps"][1]["role"] == "heat_in"
    assert d["results"]["eta_th"] == c.results.eta_th
    json.dumps(d)
    assert repr(c) == "Cycle('otto', T1=300.0, P1=100000.0, r_c=8.0, q_in=1000000.0)"


def test_package_exports():
    for name in ("Cycle", "IdealGas", "AIR", "requirements", "list_cycles", "register_cycle", "errors"):
        assert name in tc.__all__ and hasattr(tc, name), name
    assert tc.errors.InvalidCycleError is InvalidCycleError
