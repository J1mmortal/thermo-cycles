# tests/test_zero_heat_input.py  (fix: zero heat input and other undefined results, spec 9.7 / 10)
#
# A cycle whose eta_th = w_net/q_in, mep = w_net/(v_max - v_min) or bwr = -w_compression/w_expansion is
# undefined (divisor zero within 1e-9 of the cycle's own scale) must fail with InvalidCycleError naming the
# step(s) and the fix, for every check mode, never with a divide-by-zero NoSolutionError or a silent number.
import math

import pytest

from thermo_cycles import AIR, Cycle, get_cycle
from thermo_cycles.errors import InvalidCycleError, NoSolutionError
from thermo_cycles.solver import solve_cycle

T2_OTTO8 = 300 * 8 ** 0.4                  # isentropic T2 of Otto r_c = 8, gamma = 1.4 (AIR)
T2_BRAYTON10 = 300 * 10 ** (0.4 / 1.4)     # isentropic T2 of Brayton r_p = 10

ZERO_HEAT = [
    # (cycle, inputs, substrings every message must contain)
    ("otto", dict(T1=300, P1=1e5, r_c=8, T3=T2_OTTO8),
     ["zero heat input", "step 2→3 (isochoric, heat_in)", "T3 = T2", "raise T3 / q_in"]),
    ("otto", dict(T1=300, P1=1e5, r_c=8, q_in=0.0),
     ["zero heat input", "step 2→3 (isochoric, heat_in)", "T3 = T2", "raise T3 / q_in"]),
    ("atkinson", dict(T1=300, P1=1e5, r_c=8, r_e=8),
     ["zero heat input", "step 2→3 (isochoric, heat_in)", "T3 = T2", "r_e must exceed r_c"]),
    ("diesel", dict(T1=300, P1=1e5, r_c=18, cutoff=1),
     ["zero heat input", "step 2→3 (isobaric, heat_in)", "T3 = T2", "cutoff = 1"]),
    ("dual", dict(T1=300, P1=1e5, r_c=18, alpha=1, cutoff=1),
     ["zero heat input", "step 2→3 (isochoric, heat_in)", "step 3→4 (isobaric, heat_in)", "alpha = 1",
      "cutoff = 1"]),
    ("brayton", dict(T1=300, P1=1e5, r_p=10, T3=T2_BRAYTON10),
     ["zero heat input", "step 2→3 (isobaric, heat_in)", "T3 = T2", "raise T3 / q_in"]),
    ("brayton", dict(T1=300, P1=1e5, r_p=10, q_in=0.0),
     ["zero heat input", "step 2→3 (isobaric, heat_in)", "T3 = T2", "raise T3 / q_in"]),
]

OTHER_UNDEFINED = [
    # r_c = 1: heat goes in (q_in > 0) but the piston sweeps no volume: mep = w_net/(v1 - v2) is 0/0.
    ("otto", dict(T1=300, P1=1e5, r_c=1, T3=1500), ["mep", "v1 = v2", "r_c = 1", "raise r_c above 1"]),
    ("otto", dict(T1=300, P1=1e5, r_c=1, q_in=5e5), ["mep", "v1 = v2", "r_c = 1", "raise r_c above 1"]),
    # r_p = 1: the turbine delivers no work: bwr = -w_compression/w_expansion is 0/0.
    ("brayton", dict(T1=300, P1=1e5, r_p=1, T3=1400), ["bwr", "w_expansion", "r_p = 1", "raise r_p above 1"]),
]


def _check_error(ei, name, parts):
    msg = str(ei.value)
    assert msg.startswith(f"{name}: ")
    assert ei.value.problems, "problems must list the reasons"
    for part in parts:
        assert part in msg, (part, msg)
    assert all(p in msg for p in ei.value.problems)
    assert "divisor is zero" not in msg


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
@pytest.mark.parametrize("name, inputs, parts", ZERO_HEAT + OTHER_UNDEFINED)
def test_undefined_result_raises_invalid_cycle_error_in_every_check_mode(name, inputs, parts, check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle(name, check=check, **inputs)
    _check_error(ei, name, parts)
    assert "every check mode" in str(ei.value)


@pytest.mark.parametrize("name, inputs, parts", ZERO_HEAT + OTHER_UNDEFINED)
def test_undefined_result_direct_solve(name, inputs, parts):
    with pytest.raises(InvalidCycleError) as ei:
        solve_cycle(get_cycle(name), AIR, inputs, check="raise")
    _check_error(ei, name, parts)


def test_zero_heat_error_is_not_a_no_solution_error():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("atkinson", T1=300, P1=1e5, r_c=8, r_e=8)
    assert not isinstance(ei.value, NoSolutionError)
    # names the heat-in step with its q and the coinciding temperatures
    zero = [p for p in ei.value.problems if p.startswith("zero heat input")]
    assert len(zero) == 1
    assert "q = 0.0 kJ/kg" in zero[0] and "T3 = T2 = 689.22 K" in zero[0]
    assert "r_e = r_c = 8" in zero[0]


def test_otto_t3_equal_t2_off_no_longer_returns_nonsense_eta():
    # Before the fix: q_in = 3.5e-10 J/kg of rounding, check='off' returned eta_th = 1.33.
    with pytest.raises(InvalidCycleError, match="zero heat input"):
        Cycle("otto", T1=300, P1=1e5, r_c=8, T3=T2_OTTO8, check="off")


def test_zero_heat_no_spurious_heat_out_sign_problem():
    # With q_in ~ 0 the heat-sign tolerance uses the cycle's step scale, so 4→1 (q = +rounding) is not reported.
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=8, T3=T2_OTTO8)
    assert not any("heat_out" in p for p in ei.value.problems)


# --- valid limiting cases must still solve (spec 9.7) ------------------------------------------------------

def test_dual_alpha_1_is_a_valid_limit():
    c = Cycle("dual", T1=300, P1=1e5, r_c=18, alpha=1, cutoff=2)
    q23 = next(s.q for s in c.steps if (s.a, s.b) == (2, 3))
    assert q23 == pytest.approx(0.0, abs=1e-9 * c.results.q_in)
    assert c.results.q_in > 0 and 0 < c.results.eta_th < 1 and math.isfinite(c.results.mep)


def test_dual_cutoff_1_is_a_valid_limit():
    c = Cycle("dual", T1=300, P1=1e5, r_c=18, alpha=2, cutoff=1)
    q34 = next(s.q for s in c.steps if (s.a, s.b) == (3, 4))
    assert q34 == pytest.approx(0.0, abs=1e-9 * c.results.q_in)
    assert c.results.q_in > 0 and 0 < c.results.eta_th < 1


def test_negative_heat_input_keeps_warn_behaviour():
    # q_in < 0 (not zero) leaves eta_th defined: check='warn' still returns the states with a warning.
    from thermo_cycles.errors import CycleValidityWarning
    with pytest.warns(CycleValidityWarning, match="q_in = -110.2 kJ/kg is not > 0"):
        c = Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600, check="warn")
    assert c.results.q_in < 0
