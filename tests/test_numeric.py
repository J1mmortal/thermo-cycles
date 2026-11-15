# tests/test_numeric.py
import pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system
from thermo_cycles.solver.numeric import solve_numeric
from thermo_cycles.errors import AmbiguousSolutionError, NoSolutionError

OTTO = build_system(get_cycle("otto"), AIR)

def test_forward_otto_is_all_explicit():          # Waterloo ME354 wk8 (O1)
    # Deviation from the plan: the plan asserts set(rep.methods) == {"explicit"}, following spec 9.5 step 2.
    # States 2 and 4 are fixed by (s, v); with the ideal-gas relations (eos, s(T,P)) each is a 2x2 loop
    # that one tear solves (T0077 open issue 1). Everything else is explicit.
    vals, rep = solve_numeric(OTTO, {"T1": 288.0, "P1": 100e3, "r_c": 10.0, "q_in": 1800e3})
    assert set(rep.methods) == {"explicit", "torn"}
    torn = sorted(tuple(sorted(b)) for b, m in zip(rep.blocks, rep.methods) if m == "torn")
    assert torn == [("P2", "T2"), ("P4", "T4")]
    assert all(len(b) == 1 for b, m in zip(rep.blocks, rep.methods) if m == "explicit")
    assert vals["T2"] == pytest.approx(723.4, rel=3e-3) and vals["P3"] == pytest.approx(11216.6e3, rel=3e-3)
    assert vals["w_net"] == pytest.approx(1083.4e3, rel=3e-3) and rep.max_residual < 1e-9

def test_backward_otto_T4_given():                # Cengel 7e SM 9-39 (O2)
    vals, _ = solve_numeric(OTTO, {"T1": 308.0, "P1": 100e3, "r_c": 9.5, "T4": 800.0})
    assert vals["T3"] == pytest.approx(1969, rel=3e-3) and vals["P3"] == pytest.approx(6072e3, rel=3e-3)

def test_algebraic_loop_P1_solved():               # O3: P1 unknown, P3 given
    vals, rep = solve_numeric(OTTO, {"T1": 288.0, "r_c": 10.0, "q_in": 1800e3, "P3": 11216.6e3})
    assert vals["P1"] == pytest.approx(1e5, rel=1e-3)
    assert set(rep.methods) <= {"explicit", "torn", "newton"} and rep.max_residual < 1e-9

def test_two_roots_is_ambiguous():
    with pytest.raises(AmbiguousSolutionError) as ei:
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 250e3})
    roots = sorted(ei.value.candidates)
    assert ei.value.variable in {"r_c", "v2", "T2", "P2"}
    if ei.value.variable == "r_c":
        assert roots == pytest.approx([2.7362, 20.4306], rel=1e-4)

def test_no_root_is_no_solution():                 # max w_net at T3 = 1500 is 328.87 kJ/kg
    with pytest.raises(NoSolutionError):
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 400e3})
