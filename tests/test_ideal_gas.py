import itertools, math
import pytest
from thermo_cycles.fluids import IdealGas, AIR
from thermo_cycles.errors import FluidError, DegenerateStateError

def test_air_constants():
    assert (AIR.R, AIR.gamma) == (287.0, 1.4)
    assert AIR.cp == pytest.approx(1004.5) and AIR.cv == pytest.approx(717.5)

@pytest.mark.parametrize("kw,expect", [
    (dict(cp=920, gamma=1.22), dict(cv=920/1.22, R=920-920/1.22)),       # Rajput Ex 13.31
    (dict(R=300, cv=700), dict(cp=1000, gamma=1000/700)),                 # Cengel 9-24
    (dict(cp=1004.5, cv=717.5, R=287.0), dict(gamma=1.4)),                # 3 consistent values
])
def test_constructor_any_two(kw, expect):
    g = IdealGas(**kw)
    for k, v in expect.items():
        assert getattr(g, k) == pytest.approx(v, rel=1e-9)

def test_inconsistent_and_underspecified():
    with pytest.raises(FluidError):
        IdealGas(cp=1005, cv=718, R=280)
    with pytest.raises(FluidError):
        IdealGas(R=287)

def test_old_code_state1_values():
    st = AIR.state(T=298.0, P=1e5)        # old code: V=0.85526, S=87.464217
    assert st["v"] == pytest.approx(0.85526, rel=1e-6)
    assert st["s"] == pytest.approx(87.464217, rel=1e-7)
    assert st["u"] == pytest.approx(717.5 * 298) and st["h"] == pytest.approx(1004.5 * 298)

INDEPENDENT = [p for p in itertools.combinations(("T","P","v","u","h","s"), 2)
               if set(p) not in ({"T","u"}, {"T","h"}, {"u","h"})]

@pytest.mark.parametrize("a,b", INDEPENDENT)
def test_round_trip_all_pairs(a, b):
    ref = AIR.state(T=1234.5, P=3.21e6)
    got = AIR.state(**{a: ref[a], b: ref[b]})
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-10, abs=1e-9)

@pytest.mark.parametrize("a,b", [("T","u"), ("T","h"), ("u","h")])
def test_degenerate_pairs(a, b):
    ref = AIR.state(T=500, P=1e5)
    assert AIR.independent(a, b) is False
    with pytest.raises(DegenerateStateError):
        AIR.state(**{a: ref[a], b: ref[b]})
