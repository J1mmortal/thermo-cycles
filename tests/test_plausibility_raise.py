"""Fix 1 (v0.1.1): implausible inputs (state T < 100 K, state P < 1000 Pa) raise InputError by default.

Opt-out: plausibility="warn" (the v0.1 behaviour) or "off" on Cycle, solve_cycle, optimise, requirements;
with_inputs forwards it. Trial T0089 finding 1: T1 = 15 (meant deg C) returned eta 0.387 instead of 0.291.
"""

import warnings

import pytest

from thermo_cycles import Cycle, optimise, requirements
from thermo_cycles.errors import InputError, InputPlausibilityWarning
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.fluids import AIR
from thermo_cycles.solver import solve_cycle

BRAYTON = dict(P1=101300, r_p=8, T3=1250)


def test_celsius_number_raises_by_default_with_the_fix():
    with pytest.raises(InputError) as ei:
        Cycle("brayton", T1=15, **BRAYTON)
    msg = str(ei.value)
    assert "T1 = 15 K: did you mean 15 deg C (288.15 K)?" in msg
    assert "write T1=degC(15)" in msg
    assert "plausibility='warn'" in msg


def test_bar_number_raises_by_default():
    with pytest.raises(InputError, match=r"P1 = 1.013 Pa: did you mean 1.013 bar or 1.013 kPa\?"):
        Cycle("brayton", T1=288.15, P1=1.013, r_p=8, T3=1250)


def test_any_state_is_checked():
    with pytest.raises(InputError, match="P3 = 500 Pa"):
        Cycle("otto", T1=300, P1=1e5, r_c=8, P3=500)


def test_thresholds_are_strict():
    # exactly 100 K and 1000 Pa are accepted (thresholds unchanged from v0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        requirements("otto", T1=100.0, P1=1000.0, r_c=8)


def test_warn_restores_v01_behaviour():
    with pytest.warns(InputPlausibilityWarning, match="T1 = 15 K"):
        c = Cycle("brayton", T1=15, plausibility="warn", **BRAYTON)
    assert c.results.eta_th == pytest.approx(1 - 8 ** (-0.4 / 1.4), rel=1e-9)


def test_off_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = Cycle("brayton", T1=15, plausibility="off", **BRAYTON)
    assert c.states[1].T == 15


def test_bad_mode_rejected():
    with pytest.raises(InputError, match="plausibility must be one of"):
        Cycle("brayton", T1=288.15, plausibility="ignore", **BRAYTON)


def test_with_inputs_forwards_the_mode():
    c = Cycle("brayton", T1=288.15, plausibility="off", **BRAYTON)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c2 = c.with_inputs(T1=15)
    assert c2.states[1].T == 15
    c3 = Cycle("brayton", T1=288.15, **BRAYTON)
    with pytest.raises(InputError, match="T1 = 15 K"):
        c3.with_inputs(T1=15)


def test_solve_cycle_keyword():
    defn = get_cycle("brayton")
    with pytest.raises(InputError, match="T1 = 15 K"):
        solve_cycle(defn, AIR, dict(T1=15, **BRAYTON))
    with pytest.warns(InputPlausibilityWarning):
        solve_cycle(defn, AIR, dict(T1=15, **BRAYTON), plausibility="warn")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_cycle(defn, AIR, dict(T1=15, **BRAYTON), plausibility="off")


def test_requirements_keyword():
    with pytest.raises(InputError, match="T1 = 15 K"):
        requirements("brayton", T1=15, **BRAYTON)
    assert requirements("brayton", T1=15, plausibility="off", **BRAYTON).status == "ok"


def test_optimise_raises_before_the_search():
    with pytest.raises(InputError, match="P1 = 100 Pa"):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=100, T3=1500), grid=5)
    with pytest.raises(InputError, match="T3 = 50 K"):
        optimise("otto", vary={"T3": (50, 1500)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5, r_c=8), grid=5)


def test_optimise_warn_checks_once_and_off_is_silent():
    with pytest.warns(InputPlausibilityWarning) as rec:
        res = optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=100, T3=1500),
                       grid=5, plausibility="warn")
    assert len([w for w in rec if issubclass(w.category, InputPlausibilityWarning)]) == 1
    assert res.x["r_c"] == pytest.approx(10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=100, T3=1500),
                 grid=5, plausibility="off")


def test_optimise_search_never_raises_for_accepted_points():
    # the bounds pass the pre-check; no grid point may then raise a plausibility error
    res = optimise("otto", vary={"T3": (100, 1500)}, maximise="eta_th", fixed=dict(T1=100, P1=1000, r_c=8),
                   grid=7)
    assert res.feasible
