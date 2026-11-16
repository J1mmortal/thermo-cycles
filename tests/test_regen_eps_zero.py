"""5b (v0.1.1): brayton_regen with eps = 0 is the plain Brayton cycle (no heat exchanged), so the regenerator
hot/cold ordering rule must not fire (T0091 open issue 4: r_p = 14 was rejected)."""

import pytest

from thermo_cycles import Cycle
from thermo_cycles.errors import InvalidCycleError

BASE = dict(T1=300, P1=1e5, eta_c=1.0, eta_t=1.0)


@pytest.mark.parametrize("r_p", [12, 14])
@pytest.mark.parametrize("effs", [dict(), dict(eta_c=0.85, eta_t=0.88)])
def test_eps_zero_equals_brayton(r_p, effs):
    kw = {**BASE, **effs}
    plain = Cycle("brayton", r_p=r_p, T3=1300, **kw)
    regen = Cycle("brayton_regen", r_p=r_p, T4=1300, eps=0.0, **kw)
    assert regen.results.eta_th == pytest.approx(plain.results.eta_th, rel=1e-9, abs=1e-12)
    assert regen.results.w_net == pytest.approx(plain.results.w_net, rel=1e-9)
    assert regen.states[3].T == pytest.approx(regen.states[2].T, rel=1e-9)
    assert regen.states[6].T == pytest.approx(regen.states[5].T, rel=1e-9)


def test_t0091_case_r_p_14():
    # T0091: brayton (T1 = 300 K, T3 = 1300 K, ideal) at r_p = 14 has eta_th = 1 - 14**(-0.4/1.4) = 0.5295
    regen = Cycle("brayton_regen", T1=300, P1=1e5, r_p=14, T4=1300, eps=0)
    assert regen.results.eta_th == pytest.approx(1 - 14 ** (-0.4 / 1.4), rel=1e-9)
    assert round(regen.results.eta_th, 4) == 0.5295


def test_small_positive_eps_above_cut_off_still_rejected():
    # the ordering rule still applies whenever heat is exchanged
    with pytest.raises(InvalidCycleError, match="regenerator"):
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=14, T4=1300, eps=0.01)
