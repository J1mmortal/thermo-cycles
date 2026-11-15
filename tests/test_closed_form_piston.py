# tests/test_closed_form_piston.py  (spec §15.2)
import itertools, pytest
from thermo_cycles import Cycle, IdealGas

G = [1.3, 1.4, 1.67]

@pytest.mark.parametrize("g,r,q", list(itertools.product(G, [4, 8, 12, 20], [0.5e6, 1.8e6])))
def test_otto(g, r, q):
    c = Cycle("otto", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, q_in=q)
    assert c.results.eta_th == pytest.approx(1 - r ** (1 - g), rel=1e-9)

@pytest.mark.parametrize("g,r,co", list(itertools.product(G, [12, 18, 22], [1.5, 2, 3])))
def test_diesel(g, r, co):
    c = Cycle("diesel", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, cutoff=co)
    assert c.results.eta_th == pytest.approx(1 - r ** (1 - g) * (co ** g - 1) / (g * (co - 1)), rel=1e-9)

@pytest.mark.parametrize("g,r,a,co", list(itertools.product(G, [10, 14], [1.2, 1.5], [1.2, 1.8])))
def test_dual(g, r, a, co):
    c = Cycle("dual", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, alpha=a, cutoff=co)
    exp = 1 - r ** (1 - g) * (a * co ** g - 1) / ((a - 1) + g * a * (co - 1))
    assert c.results.eta_th == pytest.approx(exp, rel=1e-9)

def test_dual_limits():
    d = Cycle("dual", T1=300, P1=1e5, r_c=14, alpha=1.0, cutoff=2.0)
    di = Cycle("diesel", T1=300, P1=1e5, r_c=14, cutoff=2.0)
    assert d.results.eta_th == pytest.approx(di.results.eta_th, rel=1e-9)
    o = Cycle("dual", T1=300, P1=1e5, r_c=8, alpha=3.0, cutoff=1.0)
    ot = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=o.states[3].T)
    assert o.results.eta_th == pytest.approx(ot.results.eta_th, rel=1e-9)

@pytest.mark.parametrize("g,r,T3", list(itertools.product(G, [6, 8, 10], [1500, 2400])))
def test_atkinson(g, r, T3):
    c = Cycle("atkinson", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, T3=T3)
    re = c.params["r_e"]
    assert c.results.eta_th == pytest.approx(1 - g * (re - r) / (re ** g - r ** g), rel=1e-9)
    assert 0 < c.results.eta_th < c.results.eta_carnot
