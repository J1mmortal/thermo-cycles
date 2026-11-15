# Spec 12 behaviour that the plan's four tests do not pin down.
import math
import warnings

import pytest

from thermo_cycles import Cycle, OptimisationResult, optimise
from thermo_cycles.errors import (InputError, InputPlausibilityWarning, NoFeasiblePointError,
                                  OverdeterminedError, UnknownInputError)

OTTO = dict(T1=300, P1=1e5, T3=1500)


def test_result_type_history_and_evaluations():
    res = optimise("otto", vary={"r_c": (1.5, 30)}, maximise="w_net", fixed=OTTO, grid=21)
    assert isinstance(res, OptimisationResult) and res.feasible
    assert res.evaluations == len(res.history) >= 21
    xs, values, feas = zip(*res.history)
    assert all(set(x) == {"r_c"} for x in xs)
    assert all(isinstance(v, float) for v in values) and all(feas)          # every point solves, no limits
    assert max(values) == pytest.approx(res.value, rel=1e-12)               # nothing better was seen
    assert "w_net" in res.message and "evaluations" in res.message


def test_cycle_belongs_to_the_optimum_not_the_last_point():
    # spec 12: "The optimum's states are the optimum's" (old bug: states of the last grid point)
    res = optimise("otto", vary={"r_c": (1.5, 30)}, maximise="w_net", fixed=OTTO)
    ref = Cycle("otto", r_c=res.x["r_c"], **OTTO)
    assert res.cycle.inputs["r_c"] == res.x["r_c"]
    assert res.cycle.states[2].T == pytest.approx(ref.states[2].T, rel=1e-12)
    assert res.cycle.states[2].T == pytest.approx(math.sqrt(300 * 1500), rel=1e-6)   # T2* = sqrt(T1 T3)
    assert res.cycle.results.w_net == pytest.approx(res.value, rel=1e-12)


def test_minimise_by_name_and_alias_keys():
    # minimise q_out for Otto with q_in fixed: q_out = q_in r_c^-0.4 falls monotonically -> upper bound
    res = optimise("otto", vary={"R_C": (2, 20)}, minimise="q_out", fixed=dict(T1=300, P1=1e5, q_in=1e6))
    assert set(res.x) == {"r_c"} and res.x["r_c"] == 20
    assert res.value == pytest.approx(1e6 * 20 ** -0.4, rel=1e-9)


def test_boundary_of_the_solvable_region_by_bisection():
    # Otto with T3 fixed: eta_th = 1 - r_c^-0.4 rises until T2 reaches T3 (q_in -> 0) at r_c = 5^2.5 = 55.9017.
    # Beyond it the solve fails (invalid / degenerate), so the optimum is the edge of the solvable region.
    res = optimise("otto", vary={"r_c": (2, 80)}, maximise="eta_th", fixed=OTTO)
    assert res.x["r_c"] == pytest.approx(5 ** 2.5, rel=1e-4)
    assert res.value == pytest.approx(0.8, rel=1e-5)
    assert not all(f for _, _, f in res.history)                           # infeasible points were seen
    assert any(v is None for _, v, _ in res.history)                       # ...and recorded with value None


def test_two_variables_otto_pressure_limit():
    # eta_th depends on r_c only; P3 = P1 r_c T3/T1 <= 5 MPa is cheapest at T3 = 1000 -> r_c = 5e6/1e5*300/1000 = 15
    res = optimise("otto", vary={"r_c": (2, 20), "T3": (1000, 3000)}, maximise="eta_th",
                   fixed=dict(T1=300, P1=1e5), limits={"P_max": 5e6})
    assert res.active == ("P_max",)
    assert res.x["r_c"] == pytest.approx(15, rel=1e-5) and res.x["T3"] == pytest.approx(1000, rel=1e-6)
    assert res.value == pytest.approx(1 - 15 ** -0.4, rel=1e-6)
    assert res.cycle.margins.ok                                            # strictly feasible


def test_two_variables_deterministic_for_a_seed():
    kw = dict(vary={"r_p": (2, 40), "T3": (800, 2000)}, maximise="w_net", fixed=dict(T1=300, P1=1e5),
              limits={"T_max": 1600})
    a, b = optimise("brayton", seed=3, **kw), optimise("brayton", seed=3, **kw)
    assert dict(a.x) == dict(b.x) and a.history == b.history


def test_no_warnings_leak_from_the_search():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th", fixed=dict(T1=298, P1=101325, T3=2700),
                 limits={"P2_max": 70 * 101325})


def test_plausibility_warning_still_shown_once_for_fixed_inputs():
    with pytest.warns(InputPlausibilityWarning):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=100, T3=1500), grid=5)


def test_no_feasible_point_reports_closest():
    with pytest.raises(NoFeasiblePointError) as ei:
        optimise("otto", vary={"r_c": (5, 10), "T3": (1000, 2000)}, maximise="eta_th",
                 fixed=dict(T1=300, P1=1e5), limits={"P_max": 1e5, "T_max": 5000})
    e = ei.value
    assert set(e.closest) == {"P_max", "T_max"}
    assert not e.closest["P_max"].ok and e.closest["T_max"].ok
    assert "P_max" in str(e) and "smallest violation" in str(e)


@pytest.mark.parametrize("kw, err, text", [
    (dict(maximise="eta_th", minimise="q_out"), InputError, "exactly one"),
    (dict(), InputError, "exactly one"),
    (dict(maximise="eta_thh"), UnknownInputError, "eta_thh"),
    (dict(maximise=3.0), InputError, "objective"),
    (dict(maximise="bwr"), InputError, "flow cycles only"),
    (dict(maximise="eta_th", vary={"r_c": (10, 5)}), InputError, "lo < hi"),
    (dict(maximise="eta_th", vary={"r_c": (5, float("nan"))}), InputError, "finite"),
    (dict(maximise="eta_th", vary={"r_c": 5}), InputError, "pair"),
    (dict(maximise="eta_th", vary={}), InputError, "non-empty"),
    (dict(maximise="eta_th", vary={"T3": (1000, 2000)}), InputError, "both fixed and varied"),
    (dict(maximise="eta_th", limits={"T_maximum": 1}), UnknownInputError, "T_maximum"),
    (dict(maximise="eta_th", grid=2), InputError, "grid"),
])
def test_bad_arguments_fail_before_the_search(kw, err, text):
    args = dict(vary={"r_c": (5, 10)}, fixed=OTTO)
    args.update(kw)
    with pytest.raises(err, match=text):
        optimise("otto", **args)


def test_overdetermined_raises_before_search():
    with pytest.raises(OverdeterminedError):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(OTTO, q_in=1e6))


def test_bound_outside_the_solvable_range_names_the_input():
    with pytest.raises(InputError, match="r_c"):
        optimise("otto", vary={"r_c": (0.5, 10)}, maximise="eta_th", fixed=OTTO)
