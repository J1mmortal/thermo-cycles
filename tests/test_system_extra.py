"""Extra Task 5 tests: registry validation, candidate ranking, scan ranges, generic closure and
forward solves of the built systems (through the Task 6 block solver since Task 6)."""

import math

import pytest

import thermo_cycles
from thermo_cycles.errors import UnknownCycleError, UnknownInputError
from thermo_cycles.fluids import AIR, FluidBase, IdealGas
from thermo_cycles.model import cycle_def
from thermo_cycles.model.cycle_def import CycleDef, ParamDef, StepDef, get_cycle, list_cycles, register_cycle
from thermo_cycles.model.relations import ratio
from thermo_cycles.solver.structure import dm_partition
from thermo_cycles.solver.numeric import solve_numeric
from thermo_cycles.solver.system import EquationSystem, build_system


@pytest.fixture
def registry(monkeypatch):
    """A private copy of the registry, so tests can register cycles without leaking them."""
    monkeypatch.setattr(cycle_def, "_REGISTRY", dict(cycle_def._REGISTRY))
    return cycle_def._REGISTRY


def _graph(sys_, inputs):
    rels = sys_.with_inputs(inputs)
    idx = {v: i for i, v in enumerate(sys_.variables)}
    return rels, [frozenset(idx[v] for v in r.vars) for r in rels]


def _forward(sys_, inputs):
    """Solve with the Task 6 block solver (replaces the provisional tear + brentq of Task 5).

    Returns the values, the max scaled residual and the variables of the blocks that are not 1x1 explicit."""
    x, rep = solve_numeric(sys_, inputs)
    loops = [tuple(sorted(b)) for b, m in zip(rep.blocks, rep.methods) if m != "explicit"]
    return x, rep.max_residual, loops


def _brayton(fluid):
    """Brayton per spec 8.2 (built here, not registered; the catalogue entry is Task 10)."""
    return CycleDef(
        name="brayton_t5", title="", aliases=(), system="flow", states=(1, 2, 3, 4),
        steps=(StepDef("adiabatic", 1, 2, "compression", "eta_c"), StepDef("isobaric", 2, 3, "heat_in"),
               StepDef("adiabatic", 3, 4, "expansion", "eta_t"), StepDef("isobaric", 4, 1, "heat_out")),
        params=(ParamDef("r_p", "pressure ratio P2/P1", ratio("r_p", "P2", "P1"), bounds=(1.0, math.inf)),
                ParamDef("eta_c", "compressor isentropic efficiency", None, 1.0, (0.0, 1.0)),
                ParamDef("eta_t", "turbine isentropic efficiency", None, 1.0, (0.0, 1.0))))


# --- registry ---------------------------------------------------------------------------------------


def test_registry_lookup_and_top_level_exports():
    assert get_cycle("otto") is get_cycle("Otto") is thermo_cycles.get_cycle("OTTO")
    assert thermo_cycles.list_cycles() == sorted(thermo_cycles.list_cycles())
    assert thermo_cycles.catalogue.OTTO is get_cycle("otto")


def test_unknown_cycle_message_and_type():
    with pytest.raises(UnknownCycleError, match="did you mean: otto|Did you mean: otto") as ei:
        get_cycle("Oto")
    assert isinstance(ei.value, ValueError) and ei.value.suggestions == ("otto",)
    with pytest.raises(UnknownCycleError) as ei:
        get_cycle("zzzzzz")
    assert ei.value.suggestions == ()
    with pytest.raises(UnknownCycleError):
        get_cycle(3)


def test_alias_and_replace(registry):
    d = CycleDef(name="otto2", title="", aliases=("o2",), system="closed", states=(1, 2, 3, 4),
                 steps=get_cycle("otto").steps, params=get_cycle("otto").params, v_max_state=1, v_min_state=2)
    register_cycle(d)
    assert get_cycle("O2") is d and "otto2" in list_cycles()
    with pytest.raises(ValueError, match="replace=True"):
        register_cycle(d)
    d2 = CycleDef(**{**d.__dict__, "aliases": ("o_two",)})
    register_cycle(d2, replace=True)
    assert get_cycle("o_two") is d2
    with pytest.raises(UnknownCycleError):
        get_cycle("o2")                      # old alias removed with the replaced definition
    clash = CycleDef(**{**d.__dict__, "name": "other", "aliases": ("OTTO",)})
    with pytest.raises(ValueError, match="already used"):
        register_cycle(clash)


def _otto_like(**changes):
    base = dict(name="t", title="", aliases=(), system="closed", states=(1, 2, 3, 4),
                steps=get_cycle("otto").steps, params=get_cycle("otto").params, v_max_state=1, v_min_state=2)
    base.update(changes)
    return CycleDef(**base)


@pytest.mark.parametrize("changes, match", [
    ({"steps": (StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 2, "heat_out"))}, "closed loop"),
    ({"steps": (StepDef("isentropic", 1, 2, "compression"), StepDef("isochor", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out"))}, "unknown kind"),
    ({"steps": (StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out"))}, "unknown role"),
    ({"steps": (StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "internal"))}, "heat_out"),
    ({"steps": (StepDef("isentropic", 1, 2, "compression", "eta_c"), StepDef("isochoric", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out"))}, "no efficiency"),
    ({"steps": (StepDef("adiabatic", 1, 2, "compression", "eta_c"), StepDef("isochoric", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out"))},
     "not a declared parameter"),
    ({"steps": (StepDef("adiabatic", 1, 2, "heat_in"), StepDef("isochoric", 2, 3, "heat_in"),
                StepDef("isentropic", 3, 4, "expansion"), StepDef("isochoric", 4, 1, "heat_out"))}, "role"),
    ({"states": (1, 2, 3, 3)}, "duplicate state"),
    ({"v_max_state": None}, "v_max_state"),
    ({"v_min_state": 1}, "differ"),
    ({"system": "open"}, "system"),
    ({"kind": "refrigeration"}, "kind"),
    ({"name": "bad name"}, "identifier"),
    ({"params": (ParamDef("T1", "clash", ratio("T1", "v1", "v2")),)}, "clashes"),
    ({"params": (ParamDef("Q_IN", "clash", ratio("Q_IN", "v1", "v2")),)}, "clashes"),
    ({"params": (ParamDef("r_c", "a", ratio("r_c", "v1", "v2")), ParamDef("r_c", "b", ratio("r_c", "v1", "v2")))},
     "duplicate parameter"),
    ({"params": (ParamDef("r_c", "x", ratio("r_x", "v1", "v2")),)}, "does not contain"),
    ({"params": (ParamDef("r_c", "x", ratio("r_c", "v1", "v9")),)}, "unknown variables"),
    ({"params": (ParamDef("r_c", "x", ratio("r_c", "v1", "v2"), bounds=(5.0, 1.0)),)}, "bounds"),
    ({"params": (ParamDef("r_c", "x", ratio("r_c", "v1", "v2"), default=0.5, bounds=(1.0, math.inf)),)},
     "outside its bounds"),
    ({"params": (ParamDef("k_free", "unused", None, 1.0),)}, "not used"),
    ({"extra_relations": (ratio("r_c", "v1", "bwr"),)}, "unknown variables"),
    ({"suggest": ("bwr",)}, "suggest"),
    ({"typical_inputs": (("T1", "P9"),)}, "typical_inputs"),
])
def test_register_rejects(registry, changes, match):
    with pytest.raises(ValueError, match=match):
        register_cycle(_otto_like(**changes))
    assert "t" not in list_cycles()


def test_register_rejects_flow_cycle_without_expansion(registry):
    d = CycleDef(name="f", title="", aliases=(), system="flow", states=(1, 2, 3),
                 steps=(StepDef("adiabatic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
                        StepDef("isobaric", 3, 1, "heat_out")))
    with pytest.raises(ValueError, match="expansion"):
        register_cycle(d)


# --- the Otto system --------------------------------------------------------------------------------


def test_otto_candidates_ranked_per_9_4():
    s = build_system(get_cycle("otto"), AIR)
    assert s.candidates[:17] == ("r_c", "T3", "q_in", "P3", "T4", "T1", "P1", "v1", "T2", "P2", "v2", "v3",
                                 "P4", "v4", "q_out", "w_net", "eta_th")
    assert s.candidates[17:22] == ("mep", "s1", "s2", "s3", "s4")
    assert s.candidates[22:] == ("q_12", "w_12", "q_23", "w_23", "q_34", "w_34", "q_41", "w_41")
    assert not any(c[0] in "uh" for c in s.candidates)          # folded into T_k for the ideal gas
    assert set(s.equivalent) == {f"{p}{k}" for p in "uh" for k in range(1, 5)}


def test_otto_kinds_and_scan_ranges_per_9_5():
    s = build_system(get_cycle("otto"), AIR)
    assert s.scan_range["T1"] == (1.0, 1e5, "log") and s.scan_range["P2"] == (1.0, 1e10, "log")
    assert s.scan_range["v4"] == (1e-6, 1e4, "log") and s.scan_range["r_c"] == (1.0, 1e4, "log")
    assert s.scan_range["eta_th"] == (1e-6, 1.0, "log") and s.scan_range["s1"] == (-1e9, 1e9, "asinh")
    assert {s.var_kind[v] for v in ("u1", "h2", "s3", "q_in", "q_out", "w_net", "mep", "w_12")} == {"signed"}
    assert set(s.var_kind) == set(s.variables) == set(s.scan_range)
    assert s.defaults == {} and s.aliases == {"r": "r_c"} and s.cycle is get_cycle("otto")
    with pytest.raises(TypeError):
        s.var_kind["T1"] = "signed"          # read-only mappings
    assert isinstance(s, EquationSystem)


def test_with_inputs_unknown_name():
    s = build_system(get_cycle("otto"), AIR)
    with pytest.raises(UnknownInputError) as ei:
        s.with_inputs({"rc": 10.0})
    assert "r_c" in ei.value.suggestions and ei.value.name == "rc"
    with pytest.raises(ValueError):
        s.with_inputs({"T1": math.nan})


def test_otto_O1_forward_chain_is_explicit():
    """Spec 15.1 O1 (Waterloo ME354 tutorial wk 8): r_c 10, T1 288 K, P1 100 kPa, q_in 1800 kJ/kg.

    The BLT order of the built system is a chain of 1x1 explicit blocks except where a state is fixed
    by (s, v): with the ideal-gas relations (eos, u(T), h(T), s(T,P)) that is a 2x2 loop (eos + s(T,P))
    for states 2 and 4. Spec 9.5 step 2 says this case is fully closed-form; it is not (see REPORT).
    The values must match the printed ones within the reference tolerance (the tutorial uses
    cv 0.718, the library cv = 287/0.4 = 717.5 J/kg K)."""
    s = build_system(get_cycle("otto"), AIR)
    x, res, loops = _forward(s, {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    assert res < 1e-9
    assert sorted(loops) == [("P2", "T2"), ("P4", "T4")]       # eos k + s(T,P) k, k = 2, 4
    exp = {"T2": 723.4, "P2": 2511.9e3, "T3": 3230.4, "P3": 11216.6e3, "T4": 1286, "P4": 446.5e3,
           "q_out": 716.6e3, "w_net": 1083.4e3, "eta_th": 0.602, "mep": 1456.4e3}
    for k, v in exp.items():
        assert x[k] == pytest.approx(v, rel=3e-3), k
    # closed forms (rtol 1e-9): T2 = T1 r^(k-1), eta = 1 - r^(1-k), heat and work balance
    assert x["T2"] == pytest.approx(288.0 * 10 ** 0.4, rel=1e-9)
    assert x["eta_th"] == pytest.approx(1 - 10 ** -0.4, rel=1e-9)
    assert x["w_net"] == pytest.approx(x["q_in"] - x["q_out"], rel=1e-9)
    assert x["q_12"] == x["q_34"] == 0.0 and x["w_23"] == x["w_41"] == 0.0


def test_otto_O2_backward_inputs_are_square():
    """Spec 15.1 O2 (T4 given instead of q_in): structurally well-determined, no free or surplus part."""
    s = build_system(get_cycle("otto"), AIR)
    _, eq_vars = _graph(s, {"T1": 308.0, "P1": 1e5, "r_c": 9.5, "T4": 800.0})
    dm = dm_partition(eq_vars, len(s.variables))
    assert dm.rank == len(s.variables) == len(eq_vars)
    assert not (dm.under_vars or dm.over_eqs)


def test_otto_three_inputs_leave_one_free():
    s = build_system(get_cycle("otto"), AIR)
    _, eq_vars = _graph(s, {"T1": 288.0, "P1": 1e5, "r_c": 10.0})
    dm = dm_partition(eq_vars, len(s.variables))
    assert dm.rank == len(s.variables) - 1 and not dm.over_eqs


# --- generic fluid closure --------------------------------------------------------------------------


class _OpaqueAir(FluidBase):
    """AIR without sparse relations: forces the generic per-state closure."""
    name = "opaque air"

    def state(self, **two):
        return AIR.state(**two)


def test_generic_closure_counts_and_residuals():
    s = build_system(get_cycle("otto"), _OpaqueAir())
    assert (len(s.variables), len(s.relations)) == (38, 34)
    closures = [r for r in s.relations if r.name.startswith("closure ")]
    assert len(closures) == 16 and all(len(r.vars) == 6 for r in closures)
    assert s.equivalent == {} and "u3" in s.candidates and "h3" in s.candidates
    # residuals vanish at the ideal-gas solution of O1; the inverse reproduces the property
    x, _, _ = _forward(build_system(get_cycle("otto"), AIR), {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    for r in closures:
        assert abs(r.residual(*(x[v] for v in r.vars))) / r.scale < 1e-9
        (var,) = r.solvers
        assert r.solve_for(var, x) == pytest.approx(x[var], rel=1e-9)
    _, eq_vars = _graph(s, {"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    assert dm_partition(eq_vars, len(s.variables)).rank == 38


# --- a flow cycle with non-ideal steps (counts per spec 8.1 / 8.2 / 9.1) -------------------------------


def test_flow_cycle_counts_defaults_and_bwr():
    s = build_system(_brayton(AIR), AIR)
    # 6 states (4 main + 2s + 4s) x 6 props + 8 step q/w + 3 params + 7 results
    assert len(s.variables) == 36 + 8 + 3 + 7 == 54
    # 24 fluid + 2 x 5 adiabatic + 2 x 3 isobaric + 1 parameter definition + 4 sums + 3 flow sums
    assert len(s.relations) == 24 + 10 + 6 + 1 + 4 + 3 == 48
    assert {"T2s", "T4s", "bwr", "w_compression", "w_expansion"} <= set(s.variables) and "mep" not in s.variables
    assert s.defaults == {"eta_c": 1.0, "eta_t": 1.0}
    assert s.var_kind["eta_c"] == "positive" and s.scan_range["eta_c"] == (1e-6, 1.0, "log")
    assert s.scan_range["r_p"] == (1.0, 1e4, "log") and s.var_kind["bwr"] == "positive"
    assert "T2s" not in s.candidates and "r_p" in s.candidates and "bwr" in s.candidates
    assert len(s.with_inputs({"T1": 300.0})) == 48 + 1 + 2
    assert len(s.with_inputs({"T1": 300.0, "eta_c": 0.9})) == 48 + 2 + 1   # default overridden, not doubled


def test_brayton_B1_forward_chain():
    """Spec 15.1 B1 (thermostate cold-air Brayton example): r_p 10, T1 300 K, P1 1 bar, T3 1400 K,
    gamma 1.401662 -> T2 580.34 K, T4 723.71 K, bwr 0.4145, eta 0.4831."""
    gas = IdealGas(R=287.0, gamma=1.401662)
    s = build_system(_brayton(gas), gas)
    x, res, loops = _forward(s, {"T1": 300.0, "P1": 1e5, "r_p": 10.0, "T3": 1400.0})
    assert res < 1e-9 and loops == []          # (P, s)-fixed states are explicit: T from s(T,P)
    for k, v in {"T2": 580.34, "T4": 723.71, "bwr": 0.4145, "eta_th": 0.4831}.items():
        assert x[k] == pytest.approx(v, rel=3e-3), k
    assert x["w_compression"] < 0 < x["w_expansion"] and x["T2s"] == pytest.approx(x["T2"], rel=1e-12)


def test_fraction_kind_for_eps_and_signed_dT():
    d = CycleDef(
        name="x", title="", aliases=(), system="flow", states=(1, 2, 3, 4),
        steps=_brayton(AIR).steps,
        params=_brayton(AIR).params + (ParamDef("eps", "effectiveness", None, bounds=(0.0, 1.0)),),
        extra_relations=(ratio("eps", "T3", "T4"),))
    s = build_system(d, AIR)
    assert s.var_kind["eps"] == "fraction" and s.scan_range["eps"] == (0.0, 1.0, "asinh")
