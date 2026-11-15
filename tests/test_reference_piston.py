# tests/test_reference_piston.py  (spec §15.1 D1, D2, U1, A1)
import pytest
from thermo_cycles import Cycle, IdealGas

def test_D1_cengel8e_9_57_states():
    c = Cycle("diesel", T1=343, P1=97e3, r_c=22, cutoff=1.8)
    assert [c.states[i].T for i in (2, 3, 4)] == pytest.approx([1181, 2126, 781], rel=3e-3)
    assert c.results.eta_th == pytest.approx(0.66884, rel=1e-4)

def test_D2_cengel7e_9_58_states_1_to_3():
    c = Cycle("diesel", T1=293, P1=95e3, r_c=20, T3=2200)
    assert c.states[2].T == pytest.approx(971.1, rel=3e-3)
    assert c.params["cutoff"] == pytest.approx(2.265, rel=3e-3) and c.results.q_in == pytest.approx(1235e3, rel=3e-3)

def test_U1_cengel7e_9_54_dual():                 # Cengel x->3, 3->4, 4->5
    c = Cycle("dual", T1=253, P1=80e3, r_c=14, alpha=1.5, cutoff=1.2)
    assert c.states[1].v == pytest.approx(0.9076, rel=3e-3)
    assert c.states[2].P == pytest.approx(3219e3, rel=3e-3) and c.states[3].P == pytest.approx(4829e3, rel=3e-3)
    assert c.states[4].P == pytest.approx(4829e3, rel=3e-3)
    assert [c.states[i].T for i in (2, 3, 4, 5)] == pytest.approx([727.1, 1091, 1309, 490.0], rel=3e-3)
    r = c.results
    assert (r.q_in, r.q_out, r.eta_th) == pytest.approx((480.4e3, 170.2e3, 0.646), rel=3e-3)

def test_A1_rajput_ex_13_31_pressure_inputs():
    c = Cycle("atkinson", fluid=IdealGas(cp=920, gamma=1.22), T1=300, P1=1e5, P2=4e5, P3=16e5)
    assert [c.states[i].T for i in (2, 3, 4)] == pytest.approx([385.2, 1540.8, 934.9], rel=1e-3)
    assert c.results.q_out == pytest.approx(584.1e3, rel=3e-3)
    assert c.results.q_in == pytest.approx(866.7e3, rel=1e-2)     # book's k=1.22 vs cp/cv=1.2267
    assert c.results.w_net == pytest.approx(282.6e3, rel=2e-2)
    assert c.results.eta_th == pytest.approx(0.326, rel=1.5e-2)
    assert c.states[4].P == pytest.approx(1e5) and c.params["r_e"] > c.params["r_c"]

def test_atkinson_probe_case():
    c = Cycle("atkinson", T1=300, P1=1e5, r_c=8, T3=1800)
    assert c.params["r_e"] == pytest.approx(15.8813, rel=1e-5) and c.states[4].T == pytest.approx(595.550, rel=1e-5)
    assert c.results.eta_th == pytest.approx(0.627497, rel=1e-5) and c.results.mep == pytest.approx(312.25e3, rel=1e-4)
