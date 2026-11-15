# tests/test_custom_cycle.py  (spec §15.1 C1, Cengel 7e SM 9-24)
import pytest
from thermo_cycles import Cycle, IdealGas, register_cycle
from thermo_cycles.model.cycle_def import CycleDef, StepDef, ParamDef
from thermo_cycles.model.relations import ratio

def test_C1_three_step_cycle():
    register_cycle(CycleDef(
        name="test_9_24", title="Cengel 9-24", aliases=(), system="closed", states=(1, 2, 3),
        steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
               StepDef("isochoric", 3, 1, "heat_out")),
        params=(ParamDef("r_c", "v1/v2", ratio("r_c", "v1", "v2")),), v_max_state=1, v_min_state=2), replace=True)
    c = Cycle("test_9_24", fluid=IdealGas(R=300, cv=700), T1=293, P1=1e5, r_c=5)
    assert (c.states[2].T, c.states[3].T) == pytest.approx((584.4, 2922), rel=3e-3)
    w = {(s.a, s.b): s.w for s in c.steps}; q = {(s.a, s.b): s.q for s in c.steps}
    assert w[(1, 2)] == pytest.approx(-204.0e3, rel=3e-3) and w[(2, 3)] == pytest.approx(701.3e3, rel=3e-3)
    assert q[(2, 3)] == pytest.approx(2338e3, rel=3e-3) and q[(3, 1)] == pytest.approx(-1840.3e3, rel=3e-3)
    k = 1000 / 700
    assert c.results.eta_th == pytest.approx((1 - 1 / k) - (1 - 5 ** (1 - k)) / (k * 4), rel=1e-9)
