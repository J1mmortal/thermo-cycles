"""Extra Task 6 tests: closed forms through torn blocks, ambiguity mapping and validation, the no-solution
message, Newton, closure blocks, explicit failures, pole rejection and the report (spec 9.5, 15.2, 15.3)."""

import math
from types import MappingProxyType

import pytest

from thermo_cycles.errors import AmbiguousSolutionError, ConvergenceError, NoSolutionError
from thermo_cycles.fluids import AIR, FluidBase, IdealGas
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.model.relations import Relation, product
from thermo_cycles.solver.numeric import SolveReport, solve_numeric
from thermo_cycles.solver.system import EquationSystem, build_system

OTTO = build_system(get_cycle("otto"), AIR)
GAMMA = 1.4
CV = 287.0 / 0.4


def _toy(relations, kinds, ranges=None):
    """A hand-built EquationSystem (no cycle) for solver-only tests."""
    variables = tuple(kinds)
    rng = {v: (ranges or {}).get(v, (1e-3, 1e3, "log") if k == "positive" else (-1e9, 1e9, "asinh"))
           for v, k in kinds.items()}
    return EquationSystem(variables=variables, relations=tuple(relations),
                          var_kind=MappingProxyType(dict(kinds)), scan_range=MappingProxyType(rng),
                          candidates=variables, defaults=MappingProxyType({}), aliases=MappingProxyType({}),
                          cycle=None, equivalent=MappingProxyType({}))


def _rel(name, vars_, f, solvers=None, scale=1.0):
    return Relation(name, tuple(vars_), f, solvers or {}, scale)


# --- Otto: closed forms through the torn (s, v) blocks (spec 15.2, rtol 1e-9) -------------------------


@pytest.mark.parametrize("r_c", [4.0, 8.0, 12.0, 20.0])
@pytest.mark.parametrize("q_in", [0.5e6, 1.8e6])
def test_otto_closed_forms(r_c, q_in):
    x, rep = solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "r_c": r_c, "q_in": q_in})
    assert rep.max_residual < 1e-9
    assert x["T2"] == pytest.approx(300.0 * r_c ** (GAMMA - 1), rel=1e-9)
    assert x["P2"] == pytest.approx(1e5 * r_c ** GAMMA, rel=1e-9)
    assert x["T4"] == pytest.approx(x["T3"] / r_c ** (GAMMA - 1), rel=1e-9)
    assert x["eta_th"] == pytest.approx(1 - r_c ** (1 - GAMMA), rel=1e-9)
    assert x["w_net"] == pytest.approx(x["q_in"] - x["q_out"], rel=1e-9)
    assert x["s2"] == pytest.approx(x["s1"], abs=1e-9 * 100) and x["s4"] == pytest.approx(x["s3"], abs=1e-9 * 100)


def test_O1_all_printed_values():
    """Spec 15.1 O1 printed values (P4 = 446.5 kPa: 445.6 on the source sheet is a typo, P4 = P1*T4/T1)."""
    x, _ = solve_numeric(OTTO, {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    exp = {"T2": 723.4, "P2": 2511.9e3, "T3": 3230.4, "P3": 11216.6e3, "T4": 1286, "P4": 446.5e3,
           "q_out": 716.6e3, "w_net": 1083.4e3, "eta_th": 0.602, "mep": 1456.4e3}
    for k, v in exp.items():
        assert x[k] == pytest.approx(v, rel=3e-3), k


def test_O2_all_printed_values():
    x, _ = solve_numeric(OTTO, {"T1": 308.0, "P1": 1e5, "r_c": 9.5, "T4": 800.0})
    exp = {"T2": 757.9, "P2": 2338e3, "T3": 1969, "P3": 6072e3, "eta_th": 0.594, "mep": 652e3}
    for k, v in exp.items():
        assert x[k] == pytest.approx(v, rel=3e-3), k


def test_O3_matches_cold_air_hand_value():
    """P1 = P3 / (r_c T3/T1) with T3 = T1 r^0.4 + q_in/cv, cv 717.5: 99 945.75 Pa (spec 15.1 O3: 99 946)."""
    x, rep = solve_numeric(OTTO, {"T1": 288.0, "r_c": 10.0, "q_in": 1.8e6, "P3": 11216.6e3})
    T3 = 288.0 * 10 ** 0.4 + 1.8e6 / CV
    assert x["P1"] == pytest.approx(11216.6e3 / (10 * T3 / 288.0), rel=1e-9)
    assert "torn" in rep.methods or "newton" in rep.methods


# --- report ---------------------------------------------------------------------------------------------


def test_report_covers_every_variable_once():
    x, rep = solve_numeric(OTTO, {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    assert isinstance(rep, SolveReport)
    assert len(rep.blocks) == len(rep.methods)
    flat = [v for b in rep.blocks for v in b]
    assert sorted(flat) == sorted(OTTO.variables) == sorted(x)
    assert all(isinstance(v, float) and math.isfinite(v) for v in x.values())
    assert set(rep.methods) <= {"explicit", "torn", "newton", "closure"}


def test_inputs_not_mutated_and_ints_accepted():
    inp = {"T1": 288, "P1": 100000, "r_c": 10, "q_in": 1800000}
    before = dict(inp)
    x, _ = solve_numeric(OTTO, inp)
    assert inp == before and x["T1"] == 288.0


@pytest.mark.parametrize("inputs", [
    {"T1": 288.0, "P1": 1e5, "r_c": 10.0},                                  # under-determined
    {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6, "T3": 3000.0},     # over-determined
    {"T1": 288.0, "u1": 2e5, "r_c": 10.0, "q_in": 1.8e6},                   # ideal-gas degeneracy
])
def test_structurally_not_ok_is_value_error(inputs):
    with pytest.raises(ValueError, match="structurally"):
        solve_numeric(OTTO, inputs)


# --- multiple roots: ambiguity, mapping to r_c, validation (spec 9.5 step 3, 15.3) ----------------------


AMB = {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 250e3}


def test_ambiguity_reports_r_c_per_spec_15_3():
    with pytest.raises(AmbiguousSolutionError) as ei:
        solve_numeric(OTTO, AMB)
    assert ei.value.variable == "r_c"
    assert sorted(ei.value.candidates) == pytest.approx([2.736178786, 20.43057263], rel=1e-7)
    msg = str(ei.value)
    assert "r_c = 2.736 or 20.43" in msg and "pass r_c explicitly" in msg


def test_validate_discards_invalid_candidate():
    x, rep = solve_numeric(OTTO, AMB, validate=lambda v: ["r_c too high"] if v["r_c"] > 10 else [])
    assert x["r_c"] == pytest.approx(2.736178786, rel=1e-7) and rep.max_residual < 1e-9
    x, _ = solve_numeric(OTTO, AMB, validate=lambda v: ["r_c too low"] if v["r_c"] < 10 else [])
    assert x["r_c"] == pytest.approx(20.43057263, rel=1e-7)
    # each candidate is a complete solution: the closed forms hold
    assert x["T2"] == pytest.approx(300.0 * x["r_c"] ** 0.4, rel=1e-9)
    assert x["w_net"] == pytest.approx(250e3, rel=1e-12)


def test_validate_rejecting_all_is_still_ambiguous():
    seen = []

    def validate(v):
        seen.append(v["r_c"])
        return ["nope"]

    with pytest.raises(AmbiguousSolutionError, match="None of them passes the validity checks") as ei:
        solve_numeric(OTTO, AMB, validate=validate)
    assert len(seen) == 2 and len(ei.value.candidates) == 2


def test_validate_both_valid_is_ambiguous():
    with pytest.raises(AmbiguousSolutionError):
        solve_numeric(OTTO, AMB, validate=lambda v: [])


def test_validate_not_called_for_unique_solution():
    calls = []
    solve_numeric(OTTO, {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6},
                  validate=lambda v: calls.append(1) or ["x"])
    assert calls == []


# --- no solution: message names the inputs and the reachable extreme (spec 9.5 step 3) ------------------


def test_no_solution_message_names_max_and_r_c():
    with pytest.raises(NoSolutionError) as ei:
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 400e3})
    msg = str(ei.value)
    assert "w_net = 400 kJ/kg exceeds what T1, P1, T3 allow" in msg
    assert "max ~ 329 kJ/kg near r_c = 7.48" in msg                      # 328.87 kJ/kg at r_c = 5^1.25
    assert "r_c" in ei.value.variables


def test_no_solution_efficiency_above_one():
    """Otto eta_th = 1 - r_c^(1-gamma) < 1 for every r_c: eta_th = 1.2 has no solution."""
    with pytest.raises(NoSolutionError) as ei:
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "q_in": 1e6, "eta_th": 1.2})
    # max over the r_c scan range 1..1e4 is 1 - 1e4^-0.4 = 0.97488, reached at the range edge
    assert "eta_th = 1.2 exceeds what T1, P1, q_in allow (max ~ 0.9749 near r_c = 1e+04, at the edge" in str(ei.value)


# --- explicit-inverse failures -------------------------------------------------------------------------


def test_explicit_division_by_zero_is_no_solution():
    s = _toy([product("o", "a", "b"), _rel("o = 1", ["o"], lambda o: o - 1, {"o": lambda: 1.0}),
              _rel("b = 0", ["b"], lambda b: b, {"b": lambda: 0.0})],
             {"o": "signed", "a": "signed", "b": "signed"})
    with pytest.raises(NoSolutionError, match="o = a\\*b") as ei:
        solve_numeric(s, {})
    assert ei.value.variables == ("a",)


def test_math_domain_error_is_no_solution():
    s = _toy([_rel("x = -1", ["x"], lambda x: x + 1, {"x": lambda: -1.0}),
              _rel("y = log x", ["y", "x"], lambda y, x: y - math.log(x), {"y": lambda x: math.log(x)})],
             {"x": "signed", "y": "signed"})
    with pytest.raises(NoSolutionError, match="y = log x"):
        solve_numeric(s, {})


# --- torn scan details -----------------------------------------------------------------------------------


def test_one_by_one_without_inverse_is_torn():
    s = _toy([_rel("x^3 = 8", ["x"], lambda x: x ** 3 - 8)], {"x": "positive"})
    x, rep = solve_numeric(s, {})
    assert rep.methods == ("torn",) and x["x"] == pytest.approx(2.0, rel=1e-12)


def test_pole_is_not_a_root():
    """1/(x-2) + 1 changes sign at the pole x = 2 and at the root x = 1: only the root counts."""
    s = _toy([_rel("pole", ["x"], lambda x: 1 / (x - 2) + 1)], {"x": "positive"}, {"x": (0.1, 10.0, "log")})
    x, rep = solve_numeric(s, {})
    assert x["x"] == pytest.approx(1.0, rel=1e-12) and rep.methods == ("torn",)


def test_close_roots_are_distinct_far_ones_ambiguous():
    s = _toy([_rel("quad", ["x"], lambda x: (x - 3.0) * (x - 30.0))], {"x": "positive"})
    with pytest.raises(AmbiguousSolutionError) as ei:
        solve_numeric(s, {})
    assert ei.value.variable == "x" and sorted(ei.value.candidates) == pytest.approx([3.0, 30.0], rel=1e-12)


def test_signed_scan_finds_negative_root():
    s = _toy([_rel("x + 5e4 = 0", ["x"], lambda x: math.sinh(x / 1e4) + math.sinh(5.0))], {"x": "signed"})
    x, _ = solve_numeric(s, {})
    assert x["x"] == pytest.approx(-5e4, rel=1e-10)


def test_no_sign_change_in_toy_is_no_solution():
    s = _toy([_rel("x^2 + 1", ["x"], lambda x: x * x + 1)], {"x": "signed"})
    with pytest.raises(NoSolutionError) as ei:
        solve_numeric(s, {})
    assert ei.value.variables == ("x",)


# --- Newton (blocks one tear cannot reduce) ---------------------------------------------------------------


def test_newton_unique_positive_solution():
    """x^3 + y = 10, x + y^3 = 10 without inverses: one tear cannot reduce it; the positive root is (2, 2)."""
    s = _toy([_rel("r1", ["x", "y"], lambda x, y: x ** 3 + y - 10), _rel("r2", ["x", "y"], lambda x, y: x + y ** 3 - 10)],
             {"x": "positive", "y": "positive"})
    x, rep = solve_numeric(s, {})
    assert rep.methods == ("newton",) and rep.max_residual < 1e-9
    assert x["x"] == pytest.approx(2.0, rel=1e-9) and x["y"] == pytest.approx(2.0, rel=1e-9)


def test_newton_two_solutions_is_ambiguous():
    """x^2 + y^2 = 5, x*y = 2, both positive: (1, 2) and (2, 1)."""
    s = _toy([_rel("circle", ["x", "y"], lambda x, y: x * x + y * y - 5),
              _rel("hyperbola", ["x", "y"], lambda x, y: x * y - 2)], {"x": "positive", "y": "positive"})
    with pytest.raises(AmbiguousSolutionError) as ei:
        solve_numeric(s, {})
    assert sorted(ei.value.candidates) == pytest.approx([1.0, 2.0], rel=1e-9)
    x, _ = solve_numeric(s, {}, validate=lambda v: [] if v["x"] < v["y"] else ["x >= y"])
    assert (x["x"], x["y"]) == pytest.approx((1.0, 2.0), rel=1e-9)


def test_newton_no_convergence_is_convergence_error():
    s = _toy([_rel("r1", ["x", "y"], lambda x, y: x * x + y * y + 1), _rel("r2", ["x", "y"], lambda x, y: x - y + x * y)],
             {"x": "signed", "y": "signed"})
    with pytest.raises(ConvergenceError) as ei:
        solve_numeric(s, {})
    assert set(ei.value.variables) == {"x", "y"} and ei.value.residual > 0
    assert "r1" in str(ei.value) and "r2" in str(ei.value)


# --- generic closure: the "closure" block pattern (spec 9.5, last paragraph) -------------------------------


class _OpaqueAir(FluidBase):
    """AIR without sparse relations: forces the generic per-state closure."""
    name = "opaque air"

    def state(self, **two):
        return AIR.state(**two)


def test_closure_blocks_match_ideal_gas():
    inp = {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6}
    ref, _ = solve_numeric(OTTO, inp)
    x, rep = solve_numeric(build_system(get_cycle("otto"), _OpaqueAir()), inp)
    assert "closure" in rep.methods and rep.max_residual < 1e-9
    for k in ref:
        assert x[k] == pytest.approx(ref[k], rel=1e-9, abs=1e-6), k


def test_other_gas_closed_form():
    gas = IdealGas(cp=920.0, gamma=1.22)
    s = build_system(get_cycle("otto"), gas)
    x, rep = solve_numeric(s, {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "T3": 1800.0})
    assert x["eta_th"] == pytest.approx(1 - 8.0 ** (1 - 1.22), rel=1e-9) and rep.max_residual < 1e-9
