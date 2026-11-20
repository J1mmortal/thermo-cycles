# tests/test_reference_otto.py  (spec §15.1 O1, O2)
import pytest
from thermo_cycles import Cycle

def test_O1_waterloo_me354():
    c = Cycle("otto", T1=288, P1=100e3, r_c=10, q_in=1800e3)
    exp = {"T2": 723.4, "P2": 2511.9e3, "T3": 3230.4, "P3": 11216.6e3, "T4": 1286, "P4": 446.5e3}
    for k, v in exp.items():
        assert c.value(k) == pytest.approx(v, rel=3e-3), k
    # P4 = 446.5 kPa (445.6 on the source sheet is a typo): isochoric 4->1, P4 = P1 T4 / T1
    assert c.value("P4") == pytest.approx(100e3 * c.value("T4") / 288, rel=1e-9)
    r = c.results
    assert r.q_out == pytest.approx(716.6e3, rel=3e-3) and r.w_net == pytest.approx(1083.4e3, rel=3e-3)
    assert r.eta_th == pytest.approx(0.602, rel=3e-3) and r.mep == pytest.approx(1456.4e3, rel=3e-3)
    assert r.bwr is None and r.eta_carnot == pytest.approx(1 - 288 / c.states[3].T)
    assert r.eta_ii == pytest.approx(r.eta_th / r.eta_carnot)

def test_O2_cengel_9_39_backward():
    c = Cycle("otto", T1=308, P1=100e3, r_c=9.5, T4=800)
    assert c.states[2].T == pytest.approx(757.9, rel=3e-3) and c.states[2].P == pytest.approx(2338e3, rel=3e-3)
    assert c.states[3].T == pytest.approx(1969, rel=3e-3) and c.states[3].P == pytest.approx(6072e3, rel=3e-3)
    assert c.results.eta_th == pytest.approx(0.594, rel=3e-3) and c.results.mep == pytest.approx(652e3, rel=3e-3)
