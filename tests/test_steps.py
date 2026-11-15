import math, pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.steps import KINDS
from thermo_cycles.state import State

def mk(id, **two):
    return State(id=id, **AIR.state(**two))

def test_kinds_present():
    assert set(KINDS) == {"isentropic", "isochoric", "isobaric", "isothermal", "adiabatic"}
    assert KINDS["adiabatic"].hidden_states("2") == ("2s",) and KINDS["isobaric"].hidden_states("2") == ()

def values_of(*states):
    return {f"{p}{s.id}": getattr(s, p) for s in states for p in ("T", "P", "v", "u", "h", "s")}

@pytest.mark.parametrize("system", ["closed", "flow"])
def test_isentropic_relations_and_work(system):
    s1 = mk(1, T=300.0, P=1e5); s2 = mk(2, v=s1.v / 8, s=s1.s)
    rels = KINDS["isentropic"].relations("1", "2", system=system, role="compression", efficiency=None)
    vals = values_of(s1, s2)
    du = s2.u - s1.u; dh = s2.h - s1.h
    vals["q_12"] = 0.0; vals["w_12"] = -(du if system == "closed" else dh)
    for r in rels:
        assert abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10

def test_isobaric_closed_heat_is_dh_and_work_is_Pdv():
    s2 = mk(2, T=900.0, P=5e6); s3 = mk(3, T=1800.0, P=5e6)
    rels = KINDS["isobaric"].relations("2", "3", system="closed", role="heat_in", efficiency=None)
    vals = values_of(s2, s3); vals["q_23"] = s3.h - s2.h; vals["w_23"] = 5e6 * (s3.v - s2.v)
    assert all(abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10 for r in rels)

def test_isothermal_heat_is_T_ds():
    a = mk(1, T=500.0, P=1e6); b = mk(2, T=500.0, P=1e5)
    rels = KINDS["isothermal"].relations("1", "2", system="closed", role="expansion", efficiency=None)
    vals = values_of(a, b); vals["q_12"] = 500.0 * (b.s - a.s); vals["w_12"] = vals["q_12"]  # du = 0 ideal gas
    assert vals["q_12"] == pytest.approx(287.0 * 500.0 * math.log(10))
    assert all(abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10 for r in rels)

def test_adiabatic_compression_efficiency():  # Cengel 9-110 numbers: T1 293, rp 8, eta_c 0.87 -> T2 566.3
    rels = KINDS["adiabatic"].relations("1", "2", system="flow", role="compression", efficiency="eta_c")
    names = {v for r in rels for v in r.vars}
    assert {"s2s", "P2s", "h2s", "h1", "h2", "eta_c", "q_12", "w_12"} <= names
    s1 = mk(1, T=293.0, P=1e5); s2s = mk("2s", P=8e5, s=s1.s)
    T2 = 293.0 + (s2s.T - 293.0) / 0.87
    assert T2 == pytest.approx(566.3, rel=3e-4)

@pytest.mark.parametrize("kind,end", [
    ("isentropic", dict(P=2e6)), ("isochoric", dict(T=1500.0)), ("isobaric", dict(T=1500.0)), ("isothermal", dict(P=3e5))])
def test_paths_endpoints_and_monotonic_T(kind, end):
    a = mk(1, T=600.0, P=1e6)
    if kind == "isentropic": b = mk(2, P=end["P"], s=a.s)
    elif kind == "isochoric": b = mk(2, T=end["T"], v=a.v)
    elif kind == "isobaric": b = mk(2, T=end["T"], P=a.P)
    else: b = mk(2, T=a.T, P=end["P"])
    path = KINDS[kind].path(AIR, a, b, n=50)
    assert len(path) == 50
    for p in ("T", "P", "v", "s"):
        assert getattr(path[0], p) == pytest.approx(getattr(a, p), rel=1e-9)
        assert getattr(path[-1], p) == pytest.approx(getattr(b, p), rel=1e-9)
    Ts = [s.T for s in path]
    assert Ts == sorted(Ts) or Ts == sorted(Ts, reverse=True)
    if kind == "isentropic":
        assert max(s.s for s in path) - min(s.s for s in path) < 1e-9

def test_adiabatic_path_is_none():
    a = mk(1, T=300.0, P=1e5); b = mk(2, T=600.0, P=8e5)
    assert KINDS["adiabatic"].path(AIR, a, b) is None
