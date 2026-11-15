# tests/test_regression_old.py  (spec §15.4; values from commit 9c1505b AtkinsonSolver DESIGN mode)
import pytest
from thermo_cycles import Cycle

def test_old_atkinson_design_mode_is_diesel():
    c = Cycle("diesel", T1=298.0, P1=1e5, r_c=20.0, T3=2480.988)
    assert c.states[2].T == pytest.approx(987.707297, rel=1e-6) and c.states[2].P == pytest.approx(6628908.03468, rel=1e-6)
    assert c.states[3].v == pytest.approx(0.107415, rel=1e-5) and c.states[4].T == pytest.approx(1081.960052, rel=1e-6)
    assert c.states[4].P == pytest.approx(363073.84284, rel=1e-6) and c.states[1].s == pytest.approx(87.464217, rel=1e-6)
    r = c.results
    assert (r.eta_th, r.w_net, r.mep) == pytest.approx((0.6250058918, 937509.1289, 1153861.6499), rel=1e-6)
