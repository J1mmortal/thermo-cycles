"""Extra Task 3 tests: every inverse of every relation, errors, scales, paths."""
import math

import pytest

from thermo_cycles.fluids import AIR
from thermo_cycles.fluids.base import Fluid
from thermo_cycles.model.relations import Relation, default_scale, equal, fixed, linear, product, ratio
from thermo_cycles.model.steps import KINDS
from thermo_cycles.state import State


def mk(id, **two):
    return State(id=id, **AIR.state(**two))


def vals_of(*states):
    return {f"{p}{s.id}": getattr(s, p) for s in states for p in ("T", "P", "v", "u", "h", "s")}


def check_all(rels, vals):
    """residual ~ 0 and every inverse reproduces the value."""
    for r in rels:
        assert abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10, r.name
        for var in r.solvers:
            got = r.solve_for(var, {k: vals[k] for k in r.vars if k != var})
            assert got == pytest.approx(vals[var], rel=1e-9, abs=1e-9), (r.name, var)


def test_ideal_gas_is_a_fluid_with_relations():
    assert isinstance(AIR, Fluid)
    for r in AIR.relations("2s"):
        assert set(r.solvers) == set(r.vars)
        assert all(v.endswith("2s") for v in r.vars)


def test_helper_errors():
    with pytest.raises(KeyError):
        Relation("x", ("a", "b"), lambda a, b: a - b, {"a": lambda **k: k["b"]}).solve_for("b", {"a": 1})
    with pytest.raises(ValueError):
        equal("a", "a", "n")
    with pytest.raises(ValueError):
        linear("a", {})
    with pytest.raises(ValueError):
        linear("a", {"a": 1.0})
    with pytest.raises(ValueError):
        linear("a", {"b": 0.0})
    with pytest.raises(ValueError):
        fixed("T1", float("nan"))
    with pytest.raises(ValueError):
        Relation("x", ("a", "a"), lambda a, b: 0.0, {})
    r = ratio("r", "a", "b")
    with pytest.raises(ValueError):
        r.solve_for("r", {"a": 1.0})
    with pytest.raises(ZeroDivisionError):
        r.solve_for("r", {"a": 1.0, "b": 0.0})
    assert r.solve_for("r", {"a": 6.0, "b": 3.0, "unrelated": 9.0}) == 2.0


def test_equal_product_linear_all_inverses():
    check_all([equal("T2", "T1", "e"), product("w", "P", "dv"), linear("q", {"a": 2.0, "b": -3.0})],
              {"T2": 5.0, "T1": 5.0, "w": 6.0, "P": 2.0, "dv": 3.0, "q": 2 * 4.0 - 3 * 1.0, "a": 4.0, "b": 1.0})


def test_scales():
    assert [default_scale(v) for v in ("T1", "P2", "v3", "u4", "h5", "q_23", "w_12", "s1", "r_c", "eta_th")] == \
        [300.0, 1e5, 1.0, 1e5, 1e5, 1e5, 1e5, 100.0, 1.0, 1.0]
    assert default_scale("mep") == 1e5 and default_scale("dT_regen") == 300.0
    assert ratio("eta_th", "w_net", "q_in").scale == 1e5   # follows the numerator


@pytest.mark.parametrize("system", ["closed", "flow"])
def test_isochoric_all_inverses(system):
    s1 = mk(1, T=400.0, P=2e5); s2 = mk(2, T=1200.0, v=s1.v)
    v = vals_of(s1, s2)
    if system == "closed":
        v["q_12"] = s2.u - s1.u; v["w_12"] = 0.0
    else:
        v["w_12"] = -s1.v * (s2.P - s1.P); v["q_12"] = s2.h - s1.h + v["w_12"]
    check_all(KINDS["isochoric"].relations("1", "2", system=system, role="heat_in", efficiency=None), v)
    # energy balance: q - w = du (closed) / dh (flow)
    d = (s2.u - s1.u) if system == "closed" else (s2.h - s1.h)
    assert v["q_12"] - v["w_12"] == pytest.approx(d, rel=1e-12)


@pytest.mark.parametrize("kind", ["isentropic", "isobaric", "isothermal"])
@pytest.mark.parametrize("system", ["closed", "flow"])
def test_energy_balance_and_inverses(kind, system):
    a = mk(1, T=500.0, P=1e6)
    b = {"isentropic": lambda: mk(2, P=3e5, s=a.s), "isobaric": lambda: mk(2, T=900.0, P=a.P),
         "isothermal": lambda: mk(2, T=a.T, P=3e5)}[kind]()
    v = vals_of(a, b)
    e = "u" if system == "closed" else "h"
    d = getattr(b, e) - getattr(a, e)
    if kind == "isentropic":
        v["q_12"], v["w_12"] = 0.0, -d
    elif kind == "isobaric":
        v["q_12"] = b.h - a.h
        v["w_12"] = a.P * (b.v - a.v) if system == "closed" else 0.0
    else:
        v["q_12"] = a.T * (b.s - a.s); v["w_12"] = v["q_12"] - d
    check_all(KINDS[kind].relations("1", "2", system=system, role="x", efficiency=None), v)
    assert v["q_12"] - v["w_12"] == pytest.approx(d, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("role,eta_name,eta", [("compression", "eta_c", 0.87), ("expansion", "eta_t", 0.85)])
def test_adiabatic_inverses_and_values(role, eta_name, eta):
    k = KINDS["adiabatic"]
    a = mk(1, T=293.0, P=1e5) if role == "compression" else mk(1, T=1400.0, P=8e5)
    p2 = 8e5 if role == "compression" else 1e5
    bs = mk("2s", P=p2, s=a.s)
    hb = a.h + (bs.h - a.h) / eta if role == "compression" else a.h - eta * (a.h - bs.h)
    b = mk(2, P=p2, h=hb)
    v = vals_of(a, b, bs); v[eta_name] = eta; v["q_12"] = 0.0; v["w_12"] = -(b.h - a.h)
    rels = k.relations("1", "2", system="flow", role=role, efficiency=eta_name)
    assert len(rels) == 5   # s_bs, P_bs, efficiency, q, w
    for r in rels:
        assert abs(r.residual(*(v[x] for x in r.vars))) / r.scale < 1e-10, r.name
    # eta=1 singular inverse for h_a raises instead of returning nonsense
    eff = next(r for r in rels if "efficiency" in r.name)
    with pytest.raises(ZeroDivisionError):
        eff.solve_for("h1", {"h2s": 1.0, "h2": 2.0, "eta_c" if role == "compression" else "eta_t": 1.0})
    for var in ("h2s", "h2", eta_name):
        assert eff.solve_for(var, {x: v[x] for x in eff.vars if x != var}) == pytest.approx(v[var], rel=1e-9)


def test_adiabatic_without_efficiency_and_bad_role():
    rels = KINDS["adiabatic"].relations("3", "4", system="closed", role="expansion", efficiency=None)
    assert {x for r in rels for x in r.vars} >= {"s4s", "P4s", "h4s", "h4"}
    with pytest.raises(ValueError):
        KINDS["adiabatic"].relations("3", "4", system="closed", role="heat_in", efficiency=None)
    with pytest.raises(ValueError):
        KINDS["isobaric"].relations("1", "2", system="open", role="x", efficiency=None)


def test_reversible_flags_and_hidden():
    assert KINDS["adiabatic"].reversible is False and all(KINDS[k].reversible for k in KINDS if k != "adiabatic")
    assert KINDS["adiabatic"].hidden_states("4") == ("4s",)


def test_path_rejects_inconsistent_endpoints_and_short_n():
    a = mk(1, T=300.0, P=1e5); b = mk(2, T=600.0, P=8e5)
    with pytest.raises(ValueError, match="isentropic"):
        KINDS["isentropic"].path(AIR, a, b)
    with pytest.raises(ValueError):
        KINDS["isobaric"].path(AIR, a, mk(2, T=400.0, P=1e5), n=1)


def test_isothermal_path_lies_on_isotherm_with_pv_const():
    a = mk(1, T=500.0, P=1e6); b = mk(2, T=500.0, P=1e5)
    for s in KINDS["isothermal"].path(AIR, a, b, n=20):
        assert s.T == pytest.approx(500.0) and s.P * s.v == pytest.approx(287.0 * 500.0)


def test_isentropic_path_follows_poisson():
    a = mk(1, T=300.0, P=1e5); b = mk(2, v=a.v / 8, s=a.s)
    for s in KINDS["isentropic"].path(AIR, a, b, n=20):
        assert s.T / 300.0 == pytest.approx((s.P / 1e5) ** (0.4 / 1.4), rel=1e-12)
