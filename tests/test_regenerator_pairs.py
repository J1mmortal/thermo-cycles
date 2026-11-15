# tests/test_regenerator_pairs.py  (feat: explicit regenerator pairs on CycleDef, spec 8.1 / 9.7)
#
# Regenerator stream pairs are declared on CycleDef.regenerators as ((cold a, cold b), (hot a, hot b)) step
# tuples. The spec 9.7 rule uses the declared sides, never the step order; every internal step must be in
# exactly one declared pair (definition error otherwise).
import dataclasses

import pytest

from thermo_cycles import Cycle, get_cycle
from thermo_cycles.catalogue.gas_turbine import _effectiveness
from thermo_cycles.errors import InvalidCycleError
from thermo_cycles.model import cycle_def
from thermo_cycles.model.cycle_def import CycleDef, ParamDef, StepDef, register_cycle, validate_cycle
from thermo_cycles.model.relations import linear, ratio


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(cycle_def, "_REGISTRY", dict(cycle_def._REGISTRY))
    return cycle_def._REGISTRY


def _hot_first(**changes) -> CycleDef:
    """brayton_regen renumbered to start at the turbine exit, so the hot side (1->2) comes before the cold side
    (4->5) in step order. Old -> new numbering: 5->1, 6->2, 1->3, 2->4, 3->5, 4->6."""
    base = dict(
        name="regen_hot_first", title="regenerative Brayton, hot side listed first", aliases=(), system="flow",
        states=(1, 2, 3, 4, 5, 6),
        steps=(StepDef("isobaric", 1, 2, "internal"), StepDef("isobaric", 2, 3, "heat_out"),
               StepDef("adiabatic", 3, 4, "compression", "eta_c"), StepDef("isobaric", 4, 5, "internal"),
               StepDef("isobaric", 5, 6, "heat_in"), StepDef("adiabatic", 6, 1, "expansion", "eta_t")),
        params=(ParamDef("r_p", "pressure ratio P4/P3 (> 1)", ratio("r_p", "P4", "P3"), bounds=(1.0, float("inf"))),
                ParamDef("eta_c", "compressor efficiency", None, default=1.0, bounds=(0.0, 1.0)),
                ParamDef("eta_t", "turbine efficiency", None, default=1.0, bounds=(0.0, 1.0)),
                ParamDef("eps", "regenerator effectiveness (h5 - h4)/(h1 - h4)",
                         _effectiveness("eps", "h4", "h5", "h1"), bounds=(0.0, 1.0))),
        extra_relations=(linear("h5", {"h4": 1.0, "h1": 1.0, "h2": -1.0}, "h5 - h4 = h1 - h2"),),
        regenerators=(((4, 5), (1, 2)),),
    )
    base.update(changes)
    return CycleDef(**base)


# --- the catalogue declares its pair ---------------------------------------------------------------------

def test_brayton_regen_declares_its_pair():
    assert get_cycle("brayton_regen").regenerators == (((2, 3), (5, 6)),)
    assert get_cycle("brayton").regenerators == ()


def test_default_is_no_regenerator():
    assert get_cycle("otto").regenerators == ()


# --- hot side listed first, declared correctly ------------------------------------------------------------

def test_hot_side_first_valid_cycle_solves_and_matches_brayton_regen(registry):
    register_cycle(_hot_first())
    c = Cycle("regen_hot_first", T3=300, P3=1e5, r_p=5, T6=1400, eps=0.8)   # order-based pairing flagged this
    ref = Cycle("brayton_regen", T1=300, P1=1e5, r_p=5, T4=1400, eps=0.8)
    assert c.results.eta_th == pytest.approx(ref.results.eta_th, rel=1e-9)
    assert c.states[2].T == pytest.approx(ref.states[6].T, rel=1e-9)        # hot outlet
    assert c.states[5].T == pytest.approx(ref.states[3].T, rel=1e-9)        # cold outlet


def test_hot_side_first_invalid_regenerator_is_reported_with_declared_sides(registry):
    register_cycle(_hot_first())
    # r_p = 40: the turbine exit (new T1) is colder than the compressor exit (new T4): no eps in [0, 1] works.
    with pytest.raises(InvalidCycleError, match="regenerator") as ei:
        Cycle("regen_hot_first", T3=300, P3=1e5, r_p=40, T6=1200, eps=0.8)
    msg = next(p for p in ei.value.problems if p.startswith("regenerator"))
    assert "cold side 4->5, hot side 1->2" in msg
    assert "T1" in msg and "T4" in msg and "r_p" in msg


def test_declaring_the_sides_the_wrong_way_round_is_caught(registry):
    # Same physics, sides swapped in the declaration: the valid cycle above now violates the rule, loudly.
    register_cycle(_hot_first(regenerators=(((1, 2), (4, 5)),)))
    with pytest.raises(InvalidCycleError, match="regenerator"):
        Cycle("regen_hot_first", T3=300, P3=1e5, r_p=5, T6=1400, eps=0.8)


# --- definition errors ------------------------------------------------------------------------------------

@pytest.mark.parametrize("regens, match", [
    ((), "internal step 1->2 is not in any declared regenerator pair"),
    ((((4, 5), (2, 3)),), "step 2->3 has role 'heat_out', not 'internal'"),
    ((((4, 5), (7, 8)),), "step 7->8 is not a step of the cycle"),
    ((((4, 5), (4, 5)),), "cold and hot side are the same step 4->5"),
    ((((4, 5), (1, 2)), ((1, 2), (4, 5))), "in more than one regenerator pair"),
    ((((4, 5),),), "must be a pair"),
    (((4, 5), (1, 2)), "must be a pair"),
    ((((4, 5), (1, "2")),), "integers"),
    ([((4, 5), (1, 2))], "tuple"),
])
def test_bad_regenerator_declarations_are_definition_errors(regens, match):
    with pytest.raises(ValueError, match=match):
        validate_cycle(_hot_first(regenerators=regens))


def test_regenerators_on_a_cycle_without_internal_steps():
    otto = get_cycle("otto")
    with pytest.raises(ValueError, match="role 'heat_in', not 'internal'"):
        validate_cycle(dataclasses.replace(otto, name="o2", regenerators=(((2, 3), (4, 1)),)))


def test_undeclared_internal_step_fails_at_registration(registry):
    defn = dataclasses.replace(get_cycle("brayton_regen"), name="regen_undeclared", regenerators=())
    with pytest.raises(ValueError, match=r"internal step 2->3 is not in any declared regenerator pair.*"
                                         r"regenerators="):
        register_cycle(defn)
