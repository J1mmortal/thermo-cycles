# tests/test_closed_form_brayton.py
import itertools, pytest
from thermo_cycles import Cycle, IdealGas

@pytest.mark.parametrize("g,rp,T3", list(itertools.product([1.3, 1.4, 1.67], [4, 8, 16], [1200, 1600])))
def test_brayton(g, rp, T3):
    c = Cycle("brayton", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_p=rp, T3=T3)
    x = rp ** ((g - 1) / g)
    assert c.results.eta_th == pytest.approx(1 - 1 / x, rel=1e-9)
    assert c.results.bwr == pytest.approx(300 / T3 * x, rel=1e-9)
    assert c.results.w_net == pytest.approx(c.results.q_in - c.results.q_out, rel=1e-12)

@pytest.mark.parametrize("rp", [3, 5, 8])
def test_ideal_regeneration(rp):
    c = Cycle("brayton_regen", T1=300, P1=1e5, r_p=rp, T4=1400, eps=1.0)
    assert c.results.eta_th == pytest.approx(1 - 300 / 1400 * rp ** (0.4 / 1.4), rel=1e-9)
    assert sum(s.q for s in c.steps if s.role == "internal") == pytest.approx(0, abs=1e-6)
