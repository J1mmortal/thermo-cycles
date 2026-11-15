"""Extra (not in the plan): FluidBase unsupported-pair fallback via brentq, and call validation."""
import pytest
from thermo_cycles.fluids import AIR, FluidBase, IdealGas
from thermo_cycles.errors import FluidError, UnsupportedPairError


class TPOnly(FluidBase):
    """Toy fluid that only evaluates (T, P), like CoolProp refusing (T, h)."""
    name = "toy"
    supported_pairs = (frozenset(("T", "P")),)

    def _evaluate(self, **pair):
        return AIR.state(**pair)

    def bracket(self, prop):
        return {"P": (1e3, 1e8)}[prop]


def test_fallback_solves_unsupported_pair():
    ref = AIR.state(T=800.0, P=2e6)
    # (T, s) is not supported directly; solve P so that s matches
    got = TPOnly().state(T=800.0, s=ref["s"])
    assert got["P"] == pytest.approx(2e6, rel=1e-9)


def test_fallback_without_bracket_raises():
    class NoBracket(TPOnly):
        def bracket(self, prop):
            return super(TPOnly, self).bracket(prop)
    with pytest.raises(UnsupportedPairError):
        NoBracket().state(T=800.0, s=1000.0)


def test_fluidbase_relations_is_none():
    assert FluidBase().relations("1") is None  # IdealGas overrides it since Task 3


@pytest.mark.parametrize("kw", [dict(T=300), dict(T=300, P=1e5, v=1), dict(T=300, x=1), dict(T=300, t=310),
                                dict(T=float("nan"), P=1e5), dict(T=-5, P=1e5), dict(T=300, P="a")])
def test_bad_calls_raise_fluid_error(kw):
    with pytest.raises(FluidError):
        AIR.state(**kw)


def test_inconsistent_message_names_constant():
    with pytest.raises(FluidError, match="inconsistent"):
        IdealGas(cp=1005, cv=718, R=280)
