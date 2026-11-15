"""Margins (spec section 11, plan Task 11).

The plan's tests use ``atkinson`` (Task 9, not in this base); they are kept verbatim and skipped until that
cycle is registered. The Otto versions below run now. Number sources: see REPORT.md of T0083.
"""
import math
import warnings

import pytest

from thermo_cycles import Cycle, list_cycles
from thermo_cycles.errors import (InputError, LimitViolationError, LimitViolationWarning, UnknownInputError)
from thermo_cycles.fluids import AIR
from thermo_cycles.margins import Margin, MarginReport, SecondLaw, evaluate_limits, parse_limit_key
from thermo_cycles.solver import system_for

KW = dict(T1=300, P1=1e5, r_c=8, T3=1800)
needs_atkinson = pytest.mark.skipif("atkinson" not in list_cycles(), reason="atkinson arrives with plan Task 9")


# --- plan tests, verbatim (atkinson) --------------------------------------------------------------------
@needs_atkinson
def test_margin_values_and_location():
    c = Cycle("atkinson", limits={"T_max": 2000, "P_max": 8e6, "eta_th_min": 0.5}, **KW)
    m = c.margins["T_max"]
    assert (m.kind, m.limit, m.value, m.where) == ("max", 2000, pytest.approx(1800), 3)
    assert m.margin == pytest.approx(200) and m.margin_pct == pytest.approx(10.0) and m.ok
    assert c.margins["eta_th_min"].margin == pytest.approx(c.results.eta_th - 0.5)
    assert c.margins.ok and c.margins.violated == ()
    sl = c.margins.second_law
    assert sl.eta_ii == pytest.approx(c.results.eta_th / c.results.eta_carnot)
    assert sl.gap == pytest.approx(c.results.eta_carnot - c.results.eta_th)


@needs_atkinson
def test_violation_policies():
    with pytest.warns(LimitViolationWarning, match="P_max"):
        c = Cycle("atkinson", limits={"P_max": 1e6}, **KW)
    assert not c.margins["P_max"].ok and c.margins.worst.key == "P_max"
    with pytest.raises(LimitViolationError):
        Cycle("atkinson", limits={"P_max": 1e6}, on_violation="raise", **KW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Cycle("atkinson", limits={"P_max": 1e6}, on_violation="ignore", **KW)


@needs_atkinson
def test_state_specific_and_unknown_keys():
    c = Cycle("atkinson", limits={"P2_max": 2e6, "r_e_max": 20}, **KW)
    assert c.margins["P2_max"].where == 2 and c.margins["r_e_max"].ok
    with pytest.raises(UnknownInputError):
        Cycle("atkinson", limits={"Tmax": 2000}, **KW)


@needs_atkinson
def test_no_limits_still_has_second_law():
    c = Cycle("atkinson", **KW)
    assert len(c.margins) == 0 and c.margins.second_law.eta_carnot == pytest.approx(1 - 300 / 1800)


# --- the same checks on Otto (runs in this base) ---------------------------------------------------------
# Hand numbers, Otto T1=300 K, P1=1e5 Pa, r_c=8, T3=1800 K, gamma=1.4:
#   T2 = 300*8**0.4 = 689.2 K; P2 = 1e5*8**1.4 = 1.838e6 Pa; P3 = P2*T3/T2 = 4.8e6 Pa (= P_max);
#   eta_th = 1 - 8**-0.4 = 0.56472; eta_carnot = 1 - 300/1800 = 5/6.
def test_otto_margin_values_and_location():
    c = Cycle("otto", limits={"T_max": 2000, "P_max": 8e6, "eta_th_min": 0.5}, **KW)
    m = c.margins["T_max"]
    assert (m.kind, m.limit, m.value, m.where) == ("max", 2000, pytest.approx(1800), 3)
    assert m.margin == pytest.approx(200) and m.margin_pct == pytest.approx(10.0) and m.ok
    p = c.margins["P_max"]
    assert p.value == pytest.approx(4.8e6, rel=1e-3) and p.where == 3
    assert p.margin_pct == pytest.approx(40.0, rel=1e-3)
    e = c.margins["eta_th_min"]
    assert e.kind == "min" and e.where is None and e.margin == pytest.approx(c.results.eta_th - 0.5)
    assert e.value == pytest.approx(1 - 8 ** -0.4, rel=1e-9)
    assert c.margins.ok and c.margins.violated == ()
    sl = c.margins.second_law
    assert isinstance(sl, SecondLaw)
    assert sl.eta_th == pytest.approx(c.results.eta_th)
    assert sl.eta_carnot == pytest.approx(5 / 6)
    assert sl.eta_ii == pytest.approx(c.results.eta_th / c.results.eta_carnot)
    assert sl.gap == pytest.approx(c.results.eta_carnot - c.results.eta_th)


def test_otto_violation_policies():
    with pytest.warns(LimitViolationWarning, match="P_max"):
        c = Cycle("otto", limits={"P_max": 1e6}, **KW)
    assert not c.margins["P_max"].ok and c.margins.worst.key == "P_max"
    assert c.margins.violated == (c.margins["P_max"],) and not c.margins.ok
    with pytest.raises(LimitViolationError) as ei:
        Cycle("otto", limits={"P_max": 1e6}, on_violation="raise", **KW)
    assert [m.key for m in ei.value.violated] == ["P_max"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Cycle("otto", limits={"P_max": 1e6}, on_violation="ignore", **KW)


def test_one_warning_lists_all_violated_limits():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Cycle("otto", limits={"P_max": 1e6, "T_max": 1000, "eta_th_min": 0.5}, **KW)
    lw = [x for x in w if issubclass(x.category, LimitViolationWarning)]
    assert len(lw) == 1
    text = str(lw[0].message)
    assert "P_max" in text and "T_max" in text and "eta_th_min" not in text


def test_otto_state_specific_and_unknown_keys():
    c = Cycle("otto", limits={"P2_max": 2e6, "T4_max": 1000, "mep_min": 1e5, "eta_ii_min": 0.5,
                              "eta_carnot_max": 0.9, "q_23_max": 1e6}, **KW)
    assert c.margins["P2_max"].where == 2 and c.margins["P2_max"].ok
    assert c.margins["T4_max"].where == 4 and c.margins["T4_max"].value == pytest.approx(c.states[4].T)
    assert c.margins["mep_min"].value == pytest.approx(c.results.mep)
    assert c.margins["eta_ii_min"].value == pytest.approx(c.results.eta_ii)
    assert c.margins["eta_carnot_max"].value == pytest.approx(5 / 6)
    assert c.margins["q_23_max"].value == pytest.approx(c.results.q_in)
    with pytest.raises(UnknownInputError):
        Cycle("otto", limits={"Tmax": 2000}, **KW)
    with pytest.raises(UnknownInputError, match="Q_max"):
        Cycle("otto", limits={"Q_max": 2000}, **KW)
    with pytest.raises(UnknownInputError):
        Cycle("otto", limits={"T_limit": 2000}, **KW)


def test_property_letter_keys_take_the_extreme_with_location():
    with pytest.warns(LimitViolationWarning):
        c = Cycle("otto", limits={"T_min": 250, "P_min": 2e5, "v_max": 0.8, "s_max": 1e4, "h_min": 0}, **KW)
    m = c.margins
    assert (m["T_min"].value, m["T_min"].where, m["T_min"].kind) == (pytest.approx(300), 1, "min")
    assert m["T_min"].margin == pytest.approx(50) and m["T_min"].ok
    assert m["P_min"].value == pytest.approx(1e5) and m["P_min"].where == 1 and not m["P_min"].ok
    assert m["P_min"].margin == pytest.approx(-1e5) and m["P_min"].margin_pct == pytest.approx(-50.0)
    assert m["v_max"].value == pytest.approx(c.results.v_max) and m["v_max"].where in (1, 4)
    assert m["h_min"].value == pytest.approx(min(s.h for s in c.states.values()))
    assert m["s_max"].value == pytest.approx(max(s.s for s in c.states.values()))
    assert m.worst.key == "P_min"          # -50 % is the lowest margin_pct of the violated ones


def test_otto_no_limits_still_has_second_law():
    c = Cycle("otto", **KW)
    assert len(c.margins) == 0 and c.margins.second_law.eta_carnot == pytest.approx(1 - 300 / 1800)
    assert c.margins.ok and c.margins.violated == () and c.margins.worst is None
    assert "second law" in c.margins.table().lower()


def test_table_is_fixed_width_text():
    c = Cycle("otto", limits={"T_max": 2000, "eta_th_min": 0.5}, **KW)
    t = c.margins.table()
    assert isinstance(t, str) and "T_max" in t and "eta_th_min" in t and "ok" in t.lower()
    assert "eta_ii" in t
    with pytest.warns(LimitViolationWarning):
        t2 = Cycle("otto", limits={"P_max": 1e6}, **KW).margins.table()
    assert "VIOLATED" in t2


def test_mapping_protocol_and_immutability():
    c = Cycle("otto", limits={"T_max": 2000, "P_max": 8e6}, **KW)
    r = c.margins
    assert isinstance(r, MarginReport) and list(r) == ["T_max", "P_max"] and len(r) == 2 and "T_max" in r
    assert isinstance(r["T_max"], Margin)
    with pytest.raises(KeyError):
        r["nope"]
    with pytest.raises(Exception):
        r["T_max"].limit = 1


def test_with_inputs_keeps_limits_and_policy():
    c = Cycle("otto", limits={"T_max": 2000}, on_violation="raise", **KW)
    c2 = c.with_inputs(T3=1900)
    assert c2.margins["T_max"].margin == pytest.approx(100)
    with pytest.raises(LimitViolationError):
        c.with_inputs(T3=2100)


def test_zero_limit_gives_nan_margin_pct_not_a_crash():
    # h1 = cp*300 > 0, so h_min = 0 is satisfied; margin_pct = 100*margin/|0| is undefined -> nan
    c = Cycle("otto", limits={"h_min": 0}, **KW)
    assert c.margins["h_min"].ok and math.isnan(c.margins["h_min"].margin_pct)


def test_bad_limit_values_raise_inputerror():
    for bad in (float("nan"), float("inf"), "hot", None, True):
        with pytest.raises(InputError, match="T_max"):
            Cycle("otto", limits={"T_max": bad}, **KW)


def test_case_insensitive_keys_and_limits_property():
    c = Cycle("otto", limits={"r_c_max": 10, "t3_max": 2000}, **KW)
    assert c.margins["r_c_max"].value == 8 and c.margins["t3_max"].value == pytest.approx(1800)
    assert dict(c.limits) == {"r_c_max": 10, "t3_max": 2000}


def test_parse_limit_key():
    system = system_for(Cycle("otto", **KW).definition, AIR)
    assert parse_limit_key("T_max", system) == ("T", "max")
    assert parse_limit_key("P3_max", system) == ("P3", "max")
    assert parse_limit_key("eta_th_min", system) == ("eta_th", "min")
    assert parse_limit_key("eta_ii_min", system) == ("eta_ii", "min")
    assert parse_limit_key("eta_carnot_max", system) == ("eta_carnot", "max")
    assert parse_limit_key("t3_max", system) == ("T3", "max")
    for bad in ("Tmax", "T", "_max", "T_maximum", "zzz_max", "eta_ii_max_min", 3):
        with pytest.raises(UnknownInputError):
            parse_limit_key(bad, system)


def test_evaluate_limits_directly():
    c = Cycle("otto", **KW)
    r = evaluate_limits(c.cycle_state, {"T_max": 1700})
    assert not r.ok and r["T_max"].margin == pytest.approx(-100)
    assert r["T_max"].margin_pct == pytest.approx(-100 / 1700 * 100)
    assert evaluate_limits(c.cycle_state, {}).second_law.eta_ii == pytest.approx(c.results.eta_ii)


def test_hidden_state_extreme_brayton_2s():
    # Brayton with eta_c < 1: the isentropic end state 2s has the smallest specific volume of the cycle
    # (same P as state 2, lower T), so a v_min limit is reached at the hidden state "2s" (T2 report, Task 11).
    from thermo_cycles import Cycle
    c = Cycle("brayton", T1=300, P1=1e5, r_p=10, T3=1400, eta_c=0.85, eta_t=0.9, limits={"v_min": 0.1})
    m = c.margins["v_min"]
    assert m.where == "2s" and m.kind == "min"
    R, g = c.fluid.R, c.fluid.gamma
    T2s = 300 * 10 ** ((g - 1) / g)                            # isentropic compression, hand calculation
    assert m.value == pytest.approx(R * T2s / 10e5, rel=1e-9)  # v = R T / P at P2 = 10 bar
    assert m.value == pytest.approx(c.hidden_states["2s"].v) and m.value < c.states[2].v
    assert m.ok == (m.value >= 0.1)
