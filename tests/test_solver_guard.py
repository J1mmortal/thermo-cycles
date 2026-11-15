"""Solver guard: degenerate input sets fail loudly and fast (brief T0080; T0079 open issue 7).

Both sets are structurally well-determined but numerically singular for Otto (q_in, q_out and eta_th are
tied by eta_th = 1 - q_out/q_in; T1, T2 fix r_c), so nothing fixes the pressure scale. ``analyse`` reports
them as over-determined; these tests call ``solve_numeric`` directly, which has no such check.
"""
import time

import pytest

from thermo_cycles import AIR, get_cycle
from thermo_cycles.errors import ConvergenceError
from thermo_cycles.solver.numeric import solve_numeric
from thermo_cycles.solver.system import build_system

OTTO = build_system(get_cycle("otto"), AIR)
# the T0079 survey truth point
TRUTH, _ = solve_numeric(OTTO, {"T1": 310.0, "P1": 1.2e5, "r_c": 8.5, "q_in": 1.5e6})


def _inputs(names):
    return {n: TRUTH[n] for n in names}


def test_hanging_set_fails_fast_with_degenerate_message():
    # T0079 survey2.py sample after ('T3', 'v3', 'P4', 'eta_th'): ran > 10 min before the guard
    t0 = time.perf_counter()
    with pytest.raises(ConvergenceError, match=r"degenerate") as ei:
        solve_numeric(OTTO, _inputs(("q_in", "T2", "q_out", "eta_th")))
    assert time.perf_counter() - t0 < 10.0
    assert "analyse" in str(ei.value) and "requirements" in str(ei.value)


def test_forty_solutions_case_is_degenerate_not_ambiguous():
    # T0079 survey3.out part C: was "AmbiguousSolutionError: the inputs allow 40 solutions: P3 = 1.489 or ..."
    t0 = time.perf_counter()
    with pytest.raises(ConvergenceError, match=r"degenerate"):
        solve_numeric(OTTO, _inputs(("q_in", "T1", "T2", "eta_th")))
    assert time.perf_counter() - t0 < 10.0


@pytest.mark.parametrize("names", [("r_c", "T2", "P2", "eta_th"), ("P3", "v3", "v4", "eta_th"),
                                   ("q_in", "T4", "T1", "T2")])
def test_other_many_root_degenerate_sets(names):
    # survey3.out part C: 24, 42 and 16 "solutions"
    with pytest.raises(ConvergenceError, match=r"degenerate"):
        solve_numeric(OTTO, _inputs(names))


def test_regular_two_root_case_still_ambiguous():
    # spec 9.5 example (w_net at fixed T_max): two genuine roots stay an AmbiguousSolutionError
    from thermo_cycles.errors import AmbiguousSolutionError
    with pytest.raises(AmbiguousSolutionError, match="r_c"):
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 250e3})


def test_work_budget_raises_convergence_error(monkeypatch):
    from thermo_cycles.solver import numeric
    monkeypatch.setattr(numeric, "MAX_WORK", 500)
    # O3 loop (torn block, 400 scan points) needs more than 500 evaluations
    with pytest.raises(ConvergenceError, match=r"gave up after \d+ relation evaluations.*degenerate"):
        solve_numeric(OTTO, {"T1": 288.0, "r_c": 10.0, "q_in": 1.8e6, "P3": 11216.6e3})


def test_reach_sweep_budget_falls_back_to_plain_detail(monkeypatch):
    from thermo_cycles.errors import NoSolutionError
    from thermo_cycles.solver import numeric
    monkeypatch.setattr(numeric, "REACH_WORK", 10)
    with pytest.raises(NoSolutionError, match="no root|cannot be evaluated") as ei:
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 400e3})
    assert "exceeds what" not in str(ei.value)


def test_regular_solve_reports_work_well_below_budget():
    from thermo_cycles.solver.numeric import MAX_WORK
    _, rep = solve_numeric(OTTO, {"T1": 288.0, "r_c": 10.0, "q_in": 1.8e6, "P3": 11216.6e3})
    assert 0 < rep.work < MAX_WORK / 10
