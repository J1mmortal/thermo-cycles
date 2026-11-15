# tests/test_results.py
import pytest
from thermo_cycles import Cycle

def test_energy_and_entropy_closure():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    assert c.results.w_net == pytest.approx(c.results.q_in - c.results.q_out, rel=1e-12)
    assert sum(s.ds for s in c.steps) == pytest.approx(0, abs=1e-9)
    assert [ (s.a, s.b) for s in c.steps ] == [(1, 2), (2, 3), (3, 4), (4, 1)]
    assert c.steps[0].w < 0 and c.steps[2].w > 0 and c.steps[1].q > 0 and c.steps[3].q < 0
    assert c.cycle_state.report.max_residual < 1e-9
