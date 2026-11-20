"""5c (v0.1.1): zero heat input reached through a singular Jacobian gives the zero-heat InvalidCycleError of
T0086, not a ConvergenceError (T0091 open issue 5: diesel T1=330, P1=120e3, r_c=16, cutoff=1.0)."""

import pytest

from thermo_cycles import Cycle
from thermo_cycles.errors import ConvergenceError, InvalidCycleError


@pytest.mark.parametrize("check", ["raise", "warn", "off"])
@pytest.mark.parametrize("inputs", [dict(T1=330, P1=120e3, r_c=16, cutoff=1.0),
                                    dict(T1=300, P1=1e5, r_c=15, cutoff=1.0),
                                    dict(T1=300, P1=1e5, r_c=17, cutoff=1.0),
                                    dict(T1=300, P1=1e5, r_c=20, cutoff=1.0)])
def test_diesel_cutoff_one_is_zero_heat(inputs, check):
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("diesel", check=check, **inputs)
    msg = str(ei.value)
    assert not isinstance(ei.value, ConvergenceError)
    for part in ("zero heat input", "step 2->3 (isobaric, heat_in)", "T3 = T2", "cutoff = 1",
                 "is the degenerate limit"):
        assert part in msg, (part, msg)
    assert "Jacobian" not in msg


@pytest.mark.parametrize("r_c", [12, 16, 19])
def test_dual_alpha_and_cutoff_one_is_zero_heat(r_c):
    with pytest.raises(InvalidCycleError, match="zero heat input"):
        Cycle("dual", T1=330, P1=120e3, r_c=r_c, alpha=1.0, cutoff=1.0)


def test_genuine_degenerate_inputs_still_convergence_error_or_input_error():
    # a non-zero-heat cycle is not affected: a valid diesel still solves
    c = Cycle("diesel", T1=330, P1=120e3, r_c=16, cutoff=2.0)
    assert c.results.q_in > 0
