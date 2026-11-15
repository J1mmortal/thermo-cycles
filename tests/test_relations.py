import pytest
from thermo_cycles.model.relations import ratio, linear, equal, product, fixed
from thermo_cycles.fluids import AIR

def test_ratio_all_inverses():
    r = ratio("r_c", "v1", "v2")
    vals = {"r_c": 10.0, "v1": 0.8, "v2": 0.08}
    assert r.residual(*(vals[v] for v in r.vars)) == pytest.approx(0, abs=1e-12)
    for var in r.vars:
        others = {k: v for k, v in vals.items() if k != var}
        assert r.solve_for(var, others) == pytest.approx(vals[var])

def test_linear_and_fixed():
    l = linear("q_in", {"q_23": 1.0, "q_34": 1.0})
    assert l.solve_for("q_34", {"q_in": 5.0, "q_23": 2.0}) == pytest.approx(3.0)
    f = fixed("T1", 300.0)
    assert f.vars == ("T1",) and f.solve_for("T1", {}) == 300.0

def test_ideal_gas_relations_consistent_with_state():
    st = AIR.state(T=900.0, P=2e6)
    vals = {f"{k}7": v for k, v in st.items()}
    rels = AIR.relations("7")
    assert [r.name for r in rels] == ["eos 7", "u(T) 7", "h(T) 7", "s(T,P) 7"]
    for r in rels:
        assert abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-12
        for var in r.vars:
            assert r.solve_for(var, {k: vals[k] for k in r.vars if k != var}) == pytest.approx(vals[var], rel=1e-12)
