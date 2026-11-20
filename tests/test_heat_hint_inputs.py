"""5a (v0.1.1): heat-sign hints only suggest changes to the given inputs, in the direction that helps.

T0091 open issue 1: Cycle("dual", T1=330, P1=120e3, r_c=15, P3=130e5, T4=2300) said "Raise T4 / q_in or lower
r_c", but with P3 given T3 = T1 P3 / (P1 r_c), so lowering r_c raises T3 and makes it worse.
"""

import pytest

from thermo_cycles import Cycle
from thermo_cycles.errors import InvalidCycleError

DUAL = dict(T1=330, P1=120e3, r_c=15, P3=130e5, T4=2300)


def _heat_problem(exc):
    return next(p for p in exc.problems if "step 3->4 (isobaric, heat_in)" in p)


def test_dual_with_P3_hint_points_the_right_way():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("dual", **DUAL)
    p = _heat_problem(ei.value)
    assert "T4 = 2300 K is below T3 = 2383.3 K." in p
    assert "lower r_c" not in p and "q_in" not in p
    # closed forms: T3 = T1 P3/(P1 r_c) -> r_c >= 330*130e5/(120e3*2300) = 15.543; P3 <= 2300*120e3*15/330 = 12545 kPa
    assert "Raise T4 (to >= 2383.3 K), r_c (to >= 15.54)" in p
    assert "lower P3 (to <= 12545.5 kPa)" in p
    # only given inputs, each in a direction that helps: check the suggested values really cure the step
    for change in (dict(r_c=15.55), dict(P3=12540e3), dict(T4=2384)):
        c = Cycle("dual", **{**DUAL, **change}, check="off")
        q34 = next(s.q for s in c.steps if (s.a, s.b) == (3, 4))
        assert q34 >= 0, change


def test_hint_suggestions_are_given_inputs_only():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("dual", **DUAL)
    p = _heat_problem(ei.value)
    fix = p.split("Raise ", 1)[1]
    for name in ("q_in", "alpha", "cutoff", "T3", "P4"):
        assert f"{name} (" not in fix


def test_otto_hint_unchanged_when_generic_hint_is_right():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600)
    assert ei.value.problems[0].endswith("Raise T3 / q_in or lower r_c.")


def test_brayton_hint_unchanged_when_generic_hint_is_right():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("brayton", T1=288.15, P1=101300, r_p=8, T3=400)
    assert "T3 = 400 K is below T2 = 521.97 K. Raise T3 / q_in or lower r_p." in str(ei.value)
