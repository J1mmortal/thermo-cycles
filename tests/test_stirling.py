# tests/test_stirling.py  (v0.2, T0096): the ideal Stirling cycle with a regenerator of effectiveness eps.
#
# Numbering (catalogue convention, state 1 = start of compression; regenerator pair as brayton_regen):
#   1->2 isothermal heat_out (compression at T_L)      2->3 isochoric internal (regenerator cold side)
#   3->4 isochoric heat_in (external top-up, 0 at eps 1) 4->5 isothermal heat_in (expansion at T_H)
#   5->6 isochoric internal (regenerator hot side)      6->1 isochoric heat_out (external, 0 at eps 1)
# Cengel's numbering (SM 9-80E, 9-81): his 1 = our 4 (= 3 at eps 1), his 2 = our 5, his 3 = our 1 (= 6),
# his 4 = our 2.
import math

import numpy as np
import pytest

from thermo_cycles import Cycle, IdealGas, describe, get_cycle, requirements
from thermo_cycles.errors import InputError, InvalidCycleError

PSI = 6894.757293168          # Pa per psi (exact definition via lbf and inch)
FT3 = 0.028316846592          # m^3 per ft^3 (exact)
LBM = 0.45359237              # kg per lbm (exact)
R_AIR = 287.0                 # library AIR: cp 1004.5 - cv 717.5


def test_registered_with_regenerator_pair():
    d = get_cycle("stirling")
    assert d.system == "closed" and d.states == (1, 2, 3, 4, 5, 6)
    assert [(s.a, s.b, s.kind, s.role) for s in d.steps] == [
        (1, 2, "isothermal", "heat_out"), (2, 3, "isochoric", "internal"), (3, 4, "isochoric", "heat_in"),
        (4, 5, "isothermal", "heat_in"), (5, 6, "isochoric", "internal"), (6, 1, "isochoric", "heat_out")]
    assert d.regenerators == (((2, 3), (5, 6)),)
    p = {x.name: x for x in d.params}
    assert p["eps"].default == 1.0 and p["eps"].bounds == (0.0, 1.0)
    assert p["r_c"].bounds[0] == 1.0
    assert (d.v_max_state, d.v_min_state) == (1, 2)


# --- textbook references --------------------------------------------------------------------------------------

def test_cengel_7e_sm_9_81():
    """Cengel & Boles 7e SM 9-81: air (R 0.287), his state 3 = 298 K and 50 kPa, v3/v4 = 12, P1 = 3600 kPa.
    Printed: T1 1788 K, q_in 1275 kJ/kg, q_out 212.5 kJ/kg, w_net 1063 kJ/kg (his numbering)."""
    c = Cycle("stirling", T1=298.0, P1=50e3, r_c=12.0, P4=3600e3)
    assert c.states[4].T == pytest.approx(1788.0, rel=1e-9)      # independent of R: T = P v / R, v4 = v1/12
    assert c.states[5].T == pytest.approx(1788.0, rel=1e-9)
    r = c.results
    assert r.q_in == pytest.approx(1275e3, rel=1e-3)
    assert r.q_out == pytest.approx(212.5e3, rel=1e-3)
    assert r.w_net == pytest.approx(1063e3, rel=1e-3)
    # perfect regeneration: the external isochoric steps carry no heat, states 3 = 4 and 6 = 1
    assert c.value("q_34") == pytest.approx(0.0, abs=1e-6) and c.value("q_61") == pytest.approx(0.0, abs=1e-6)
    assert c.states[3].T == pytest.approx(1788.0, rel=1e-9) and c.states[6].T == pytest.approx(298.0, rel=1e-9)
    assert r.eta_th == pytest.approx(1 - 298.0 / 1788.0, rel=1e-9)


def test_cengel_7e_sm_9_80E():
    """Cengel & Boles 7e SM 9-80E: T_L 510 R, 10 psia at the largest volume 0.5 ft^3, smallest volume 0.06 ft^3,
    net work 2.5 Btu of 6 Btu heat (eta 2.5/6). Printed: T_H 874 R, m 0.02647 lbm, P_max 143 psia."""
    c = Cycle("stirling", T1=510 / 1.8, P1=10 * PSI, r_c=0.5 / 0.06, eta_th=2.5 / 6)
    assert c.states[4].T * 1.8 == pytest.approx(874.0, rel=1e-3)          # exact 874.29 R
    assert c.results.P_max / PSI == pytest.approx(143.0, rel=2e-3)        # exact 142.86 psia
    assert c.results.P_max == pytest.approx(c.states[4].P, rel=1e-12)
    mass = 0.5 * FT3 / c.states[1].v / LBM
    assert mass == pytest.approx(0.02647, rel=1e-3)


# --- closed forms (rtol 1e-9) ------------------------------------------------------------------------------------

GAMMAS = (1.3, 1.4, 1.67)
RATIOS = (2.0, 5.0, 12.0)
T_HOT = (600.0, 1200.0, 1800.0)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("r_c", RATIOS)
@pytest.mark.parametrize("T_H", T_HOT)
def test_ideal_regeneration_is_carnot(gamma, r_c, T_H):
    fl = IdealGas(R=R_AIR, gamma=gamma)
    T_L = 300.0
    c = Cycle("stirling", fluid=fl, T1=T_L, P1=1e5, r_c=r_c, T4=T_H)
    r = c.results
    assert r.eta_th == pytest.approx(1 - T_L / T_H, rel=1e-9)
    assert r.q_in == pytest.approx(R_AIR * T_H * math.log(r_c), rel=1e-9)      # isothermal expansion 4->5
    assert c.value("q_45") == pytest.approx(R_AIR * T_H * math.log(r_c), rel=1e-9)
    assert r.q_out == pytest.approx(R_AIR * T_L * math.log(r_c), rel=1e-9)
    assert r.w_net == pytest.approx(R_AIR * (T_H - T_L) * math.log(r_c), rel=1e-9)
    assert r.w_net == pytest.approx(r.q_in - r.q_out, rel=1e-9)
    assert r.mep == pytest.approx(r.w_net / (c.states[1].v - c.states[2].v), rel=1e-9)
    # regenerator: q_23 = cv (T_H - T_L) = -q_56
    assert c.value("q_23") == pytest.approx(fl.cv * (T_H - T_L), rel=1e-9)
    assert c.value("q_56") == pytest.approx(-fl.cv * (T_H - T_L), rel=1e-9)
    # second law: perfect regeneration reaches Carnot
    assert r.eta_carnot == pytest.approx(1 - T_L / T_H, rel=1e-12)
    assert r.eta_ii == pytest.approx(1.0, rel=1e-9)
    # first law and state closure around the loop
    assert math.fsum(s.ds for s in c.steps) == pytest.approx(0.0, abs=1e-9 * fl.cp)
    assert math.fsum(c.states[s.b].u - c.states[s.a].u for s in c.steps) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("gamma", (1.4, 1.67))
@pytest.mark.parametrize("r_c", (3.0, 10.0))
@pytest.mark.parametrize("eps", (0.0, 0.5, 0.8))
def test_imperfect_regeneration_closed_form(gamma, r_c, eps):
    """eta = R (T_H - T_L) ln r / (R T_H ln r + (1 - eps) cv (T_H - T_L)): the regenerator gives eps of the
    isochoric heat cv (T_H - T_L), the rest comes from outside (step 3->4)."""
    fl = IdealGas(R=R_AIR, gamma=gamma)
    T_L, T_H = 300.0, 1500.0
    c = Cycle("stirling", fluid=fl, T1=T_L, P1=1e5, r_c=r_c, T4=T_H, eps=eps)
    ln = math.log(r_c)
    q_in = R_AIR * T_H * ln + (1 - eps) * fl.cv * (T_H - T_L)
    w = R_AIR * (T_H - T_L) * ln
    assert c.results.q_in == pytest.approx(q_in, rel=1e-9)
    assert c.results.w_net == pytest.approx(w, rel=1e-9)
    assert c.results.eta_th == pytest.approx(w / q_in, rel=1e-9)
    assert c.value("q_34") == pytest.approx((1 - eps) * fl.cv * (T_H - T_L), rel=1e-9)
    assert c.value("q_23") == pytest.approx(eps * fl.cv * (T_H - T_L), rel=1e-9, abs=1e-6)
    assert c.results.eta_ii < 1.0 - 1e-6


def test_eps_zero_has_no_regenerator_heat():
    c = Cycle("stirling", T1=300, P1=1e5, r_c=5, T4=1200, eps=0.0)
    assert c.value("q_23") == pytest.approx(0.0, abs=1e-6) and c.value("q_56") == pytest.approx(0.0, abs=1e-6)


def test_backward_inputs_q_in_and_eta():
    ref = Cycle("stirling", T1=300, P1=1e5, r_c=6, T4=1400)
    back = Cycle("stirling", T1=300, P1=1e5, r_c=6, q_in=ref.results.q_in)
    assert back.states[4].T == pytest.approx(1400.0, rel=1e-9)
    back2 = Cycle("stirling", T1=300, P1=1e5, T4=1400, w_net=ref.results.w_net)
    assert back2.params["r_c"] == pytest.approx(6.0, rel=1e-9)


# --- validity ---------------------------------------------------------------------------------------------------

def test_inverted_temperatures_trip_the_regenerator_rule():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("stirling", T1=900, P1=1e5, r_c=5, T4=400)
    msg = str(ei.value)
    assert "regenerator (cold side 2->3, hot side 5->6)" in msg
    assert "T5" in msg and "T2" in msg and "raise T4" in msg
    assert "lower r_c" not in msg          # r_c does not set T2 here: compression is isothermal


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
def test_r_c_one_is_zero_heat_and_zero_swept_volume(check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("stirling", T1=300, P1=1e5, r_c=1.0, T4=1200, check=check)
    msg = str(ei.value)
    for part in ("zero heat input", "r_c = 1", "degenerate limit", "zero swept volume"):
        assert part in msg, (part, msg)


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
def test_equal_temperatures_are_rejected(check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("stirling", T1=300, P1=1e5, r_c=5, T4=300, check=check)
    msg = str(ei.value)
    for part in ("no temperature difference", "T_max = T_min = 300 K", "eta_ii", "raise T4"):
        assert part in msg, (part, msg)


def test_eps_outside_bounds_is_an_input_error():
    with pytest.raises(InputError, match="eps"):
        Cycle("stirling", T1=300, P1=1e5, r_c=5, T4=1200, eps=1.2)


def test_requirements_under_and_over():
    req = requirements("stirling", T1=300, P1=1e5, r_c=5)
    assert req.status == "under" and req.missing == 1
    assert {"T4", "P4", "q_in", "w_net", "eta_th"} <= set(req.add_one_of)
    assert "T2" not in req.add_one_of and "T2" in req.determined        # isothermal: T2 = T1
    over = requirements("stirling", T1=300, P1=1e5, r_c=5, T4=1200, q_in=1e6)
    assert over.status == "over" and {"T4", "q_in"} <= set(over.drop_one_of)
    over2 = requirements("stirling", T1=300, T2=300, P1=1e5, r_c=5)
    assert over2.status == "over" and {"T1", "T2"} <= set(over2.drop_one_of)


# --- describe and plots -------------------------------------------------------------------------------------------

def test_describe():
    text = describe("stirling")
    for part in ("Stirling", "isothermal", "regenerator", "eps", "default 1", "stirling needs 4 inputs",
                 "Cengel & Boles 7e SM 9-81"):
        assert part in text, (part, text)


def test_plots_isotherms_horizontal_in_Ts_and_hyperbolic_in_Pv():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        c = Cycle("stirling", T1=300, P1=1e5, r_c=5, T4=1200, eps=0.8)
        ax = c.plot("Ts")
        L = {l.get_gid(): l for l in ax.lines if (l.get_gid() or "").startswith("step")}
        assert len(L) == 6
        for gid, T in (("step 1-2", 300.0), ("step 4-5", 1200.0)):
            y = np.asarray(L[gid].get_ydata())
            assert np.ptp(y) <= 1e-9 * T and y[0] == pytest.approx(T, rel=1e-12)
            assert np.ptp(L[gid].get_xdata()) > 100.0                      # s changes along the isotherm
        ax2 = c.plot("Pv")
        L2 = {l.get_gid(): l for l in ax2.lines if (l.get_gid() or "").startswith("step")}
        v, P = np.asarray(L2["step 4-5"].get_xdata()), np.asarray(L2["step 4-5"].get_ydata())
        Pv = P * v * (1e3 if np.max(P) < 1e5 else 1.0)                      # axis may be in kPa
        assert np.ptp(Pv) <= 1e-9 * np.max(Pv)                             # P v = R T on the isotherm
        assert np.ptp(L2["step 2-3"].get_xdata()) <= 1e-9 * c.states[2].v  # isochore vertical in P-v
    finally:
        plt.close("all")
