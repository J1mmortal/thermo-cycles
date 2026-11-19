# tests/test_ericsson.py  (v0.2, T0096): the ideal Ericsson cycle with a regenerator of effectiveness eps.
#
# Numbering (catalogue convention, state 1 = start of compression; regenerator pair as brayton_regen):
#   1->2 isothermal heat_out (compression at T_L)       2->3 isobaric internal (regenerator cold side, P_high)
#   3->4 isobaric heat_in (external top-up, 0 at eps 1)   4->5 isothermal heat_in (expansion at T_H)
#   5->6 isobaric internal (regenerator hot side, P_low) 6->1 isobaric heat_out (external, 0 at eps 1)
# Closed (piston) system with an MEP; per unit mass identical to the steady-flow cycle of the textbooks.
import math

import numpy as np
import pytest

from thermo_cycles import Cycle, IdealGas, describe, get_cycle, requirements
from thermo_cycles.errors import InputError, InvalidCycleError

R_AIR = 287.0                 # library AIR: cp 1004.5 - cv 717.5


def test_registered_with_regenerator_pair():
    d = get_cycle("ericsson")
    assert d.system == "closed" and d.states == (1, 2, 3, 4, 5, 6)
    assert [(s.a, s.b, s.kind, s.role) for s in d.steps] == [
        (1, 2, "isothermal", "heat_out"), (2, 3, "isobaric", "internal"), (3, 4, "isobaric", "heat_in"),
        (4, 5, "isothermal", "heat_in"), (5, 6, "isobaric", "internal"), (6, 1, "isobaric", "heat_out")]
    assert d.regenerators == (((2, 3), (5, 6)),)
    p = {x.name: x for x in d.params}
    assert p["eps"].default == 1.0 and p["eps"].bounds == (0.0, 1.0)
    assert p["r_p"].bounds[0] == 1.0
    assert (d.v_max_state, d.v_min_state) == (5, 2)


# --- textbook reference -------------------------------------------------------------------------------------------

def test_cengel_boles_5e_ch9_p65():
    """Cengel & Boles 5e, Chapter 9, Problem 65 (statement in CSUN ME470 homework set): ideal Ericsson, air, 27 C and
    120 kPa at the start of isothermal compression, 150 kJ/kg rejected during it, heat added at 1200 K.
    Asked: (a) P_max, (b) w_net, (c) eta_th. No printed answers were found; the expectations are the hand
    calculation with R = 0.287 kJ/kg K: P_max = 120 exp(150/(0.287*300)) = 685.16 kPa,
    q_in = 0.287*1200*ln(P_max/120) = 600 kJ/kg, w_net = 450 kJ/kg, eta_th = 0.75 (= 1 - 300/1200)."""
    c = Cycle("ericsson", T1=300.0, P1=120e3, q_12=-150e3, T4=1200.0)
    r = c.results
    assert r.P_max == pytest.approx(685.16e3, rel=1e-4)
    assert c.params["r_p"] == pytest.approx(math.exp(150e3 / (R_AIR * 300.0)), rel=1e-9)
    assert r.q_in == pytest.approx(600e3, rel=1e-9)
    assert r.q_out == pytest.approx(150e3, rel=1e-9)
    assert r.w_net == pytest.approx(450e3, rel=1e-9)
    assert r.eta_th == pytest.approx(0.75, rel=1e-9)
    assert c.value("q_34") == pytest.approx(0.0, abs=1e-6) and c.value("q_61") == pytest.approx(0.0, abs=1e-6)


# --- closed forms (rtol 1e-9) --------------------------------------------------------------------------------------

GAMMAS = (1.3, 1.4, 1.67)
RATIOS = (2.0, 5.0, 12.0)
T_HOT = (600.0, 1200.0, 1800.0)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("r_p", RATIOS)
@pytest.mark.parametrize("T_H", T_HOT)
def test_ideal_regeneration_is_carnot(gamma, r_p, T_H):
    fl = IdealGas(R=R_AIR, gamma=gamma)
    T_L, P1 = 300.0, 1e5
    c = Cycle("ericsson", fluid=fl, T1=T_L, P1=P1, r_p=r_p, T4=T_H)
    r = c.results
    ln = math.log(r_p)
    assert r.eta_th == pytest.approx(1 - T_L / T_H, rel=1e-9)
    assert r.q_in == pytest.approx(R_AIR * T_H * ln, rel=1e-9)               # isothermal expansion 4->5
    assert c.value("q_45") == pytest.approx(R_AIR * T_H * ln, rel=1e-9)
    assert r.q_out == pytest.approx(R_AIR * T_L * ln, rel=1e-9)
    assert r.w_net == pytest.approx(R_AIR * (T_H - T_L) * ln, rel=1e-9)
    assert r.P_max == pytest.approx(r_p * P1, rel=1e-9)
    # mep over the swept volume v5 - v2 = R (T_H - T_L r_p ... ) / P: v5 = R T_H / P1, v2 = R T_L / (r_p P1)
    v5, v2 = R_AIR * T_H / P1, R_AIR * T_L / (r_p * P1)
    assert c.states[5].v == pytest.approx(v5, rel=1e-9) and c.states[2].v == pytest.approx(v2, rel=1e-9)
    assert r.mep == pytest.approx(r.w_net / (v5 - v2), rel=1e-9)
    # regenerator: q_23 = cp (T_H - T_L) = -q_56
    assert c.value("q_23") == pytest.approx(fl.cp * (T_H - T_L), rel=1e-9)
    assert c.value("q_56") == pytest.approx(-fl.cp * (T_H - T_L), rel=1e-9)
    # second law
    assert r.eta_carnot == pytest.approx(1 - T_L / T_H, rel=1e-12)
    assert r.eta_ii == pytest.approx(1.0, rel=1e-9)
    assert math.fsum(s.ds for s in c.steps) == pytest.approx(0.0, abs=1e-9 * fl.cp)
    assert math.fsum(c.states[s.b].u - c.states[s.a].u for s in c.steps) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("gamma", (1.4, 1.67))
@pytest.mark.parametrize("r_p", (3.0, 10.0))
@pytest.mark.parametrize("eps", (0.0, 0.5, 0.8))
def test_imperfect_regeneration_closed_form(gamma, r_p, eps):
    """eta = R (T_H - T_L) ln r_p / (R T_H ln r_p + (1 - eps) cp (T_H - T_L))."""
    fl = IdealGas(R=R_AIR, gamma=gamma)
    T_L, T_H = 300.0, 1500.0
    c = Cycle("ericsson", fluid=fl, T1=T_L, P1=1e5, r_p=r_p, T4=T_H, eps=eps)
    ln = math.log(r_p)
    q_in = R_AIR * T_H * ln + (1 - eps) * fl.cp * (T_H - T_L)
    w = R_AIR * (T_H - T_L) * ln
    assert c.results.q_in == pytest.approx(q_in, rel=1e-9)
    assert c.results.w_net == pytest.approx(w, rel=1e-9)
    assert c.results.eta_th == pytest.approx(w / q_in, rel=1e-9)
    assert c.value("q_34") == pytest.approx((1 - eps) * fl.cp * (T_H - T_L), rel=1e-9)
    assert c.value("q_23") == pytest.approx(eps * fl.cp * (T_H - T_L), rel=1e-9, abs=1e-6)
    assert c.results.eta_ii < 1.0 - 1e-6


def test_backward_inputs():
    ref = Cycle("ericsson", T1=300, P1=1e5, r_p=6, T4=1400)
    back = Cycle("ericsson", T1=300, P1=1e5, r_p=6, q_in=ref.results.q_in)
    assert back.states[4].T == pytest.approx(1400.0, rel=1e-9)
    back2 = Cycle("ericsson", T1=300, P1=1e5, T4=1400, w_net=ref.results.w_net)
    assert back2.params["r_p"] == pytest.approx(6.0, rel=1e-9)


# --- validity ------------------------------------------------------------------------------------------------------

def test_inverted_temperatures_trip_the_regenerator_rule():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("ericsson", T1=900, P1=1e5, r_p=5, T4=400)
    msg = str(ei.value)
    assert "regenerator (cold side 2->3, hot side 5->6)" in msg
    assert "T5" in msg and "T2" in msg and "raise T4" in msg
    assert "lower r_p" not in msg


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
def test_r_p_one_is_zero_heat(check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("ericsson", T1=300, P1=1e5, r_p=1.0, T4=1200, check=check)
    msg = str(ei.value)
    for part in ("zero heat input", "r_p = 1", "degenerate limit", "raise r_p above 1"):
        assert part in msg, (part, msg)


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
def test_equal_temperatures_are_rejected(check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("ericsson", T1=300, P1=1e5, r_p=5, T4=300, check=check)
    msg = str(ei.value)
    for part in ("no temperature difference", "T_max = T_min = 300 K", "eta_ii", "raise T4"):
        assert part in msg, (part, msg)


def test_eps_outside_bounds_is_an_input_error():
    with pytest.raises(InputError, match="eps"):
        Cycle("ericsson", T1=300, P1=1e5, r_p=5, T4=1200, eps=-0.1)


def test_requirements_under_and_over():
    req = requirements("ericsson", T1=300, P1=1e5, r_p=5)
    assert req.status == "under" and req.missing == 1
    assert {"T4", "q_in", "w_net", "eta_th"} <= set(req.add_one_of)
    assert "T2" not in req.add_one_of and "T2" in req.determined
    assert "P4" not in req.add_one_of and "P4" in req.determined         # isobaric 2->3->4: P4 = r_p P1
    over = requirements("ericsson", T1=300, P1=1e5, r_p=5, T4=1200, q_in=1e6)
    assert over.status == "over" and {"T4", "q_in"} <= set(over.drop_one_of)
    over2 = requirements("ericsson", T1=300, P1=1e5, P2=5e5, r_p=5)
    assert over2.status == "over" and {"P2", "r_p"} <= set(over2.drop_one_of)


# --- describe and plots ----------------------------------------------------------------------------------------------

def test_describe():
    text = describe("ericsson")
    for part in ("Ericsson", "isothermal", "regenerator", "eps", "default 1", "ericsson needs 4 inputs",
                 "Cengel & Boles 5e Ch. 9 Problem 65"):
        assert part in text, (part, text)


def test_plots_isotherms_horizontal_in_Ts_and_isobars_horizontal_in_Pv():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        c = Cycle("ericsson", T1=300, P1=1e5, r_p=5, T4=1200, eps=0.8)
        ax = c.plot("Ts")
        L = {l.get_gid(): l for l in ax.lines if (l.get_gid() or "").startswith("step")}
        assert len(L) == 6
        for gid, T in (("step 1-2", 300.0), ("step 4-5", 1200.0)):
            y = np.asarray(L[gid].get_ydata())
            assert np.ptp(y) <= 1e-9 * T and y[0] == pytest.approx(T, rel=1e-12)
            assert np.ptp(L[gid].get_xdata()) > 100.0
        ax2 = c.plot("Pv")
        L2 = {l.get_gid(): l for l in ax2.lines if (l.get_gid() or "").startswith("step")}
        v, P = np.asarray(L2["step 4-5"].get_xdata()), np.asarray(L2["step 4-5"].get_ydata())
        Pv = P * v
        assert np.ptp(Pv) <= 1e-9 * np.max(Pv)                              # P v = R T on the isotherm
        y23 = np.asarray(L2["step 2-3"].get_ydata())
        assert np.ptp(y23) <= 1e-9 * np.max(y23)                            # isobar horizontal in P-v
    finally:
        plt.close("all")


def test_r_p_one_with_imperfect_regenerator_is_zero_net_work():
    """r_p = 1 with eps < 1: the external isobaric steps still carry heat, so q_in > 0 but w_net = 0 (eta_th = 0):
    the cycle only heats and cools the gas. Rejected as not a power cycle in check='raise'."""
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("ericsson", T1=300, P1=1e5, r_p=1.0, T4=1200, eps=0.5)
    msg = str(ei.value)
    for part in ("zero net work", "w_net = 0.0 kJ/kg", "not a power cycle", "r_p = 1", "raise r_p above 1"):
        assert part in msg, (part, msg)
    with pytest.warns(UserWarning, match="zero net work"):
        c = Cycle("ericsson", T1=300, P1=1e5, r_p=1.0, T4=1200, eps=0.5, check="warn")
    assert c.results.w_net == pytest.approx(0.0, abs=1e-6)
