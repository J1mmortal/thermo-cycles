"""Cycle definitions as data, validation and the cycle registry (spec section 8.1)."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from typing import Literal

from ..errors import UnknownCycleError
from ..state import PROPS
from .relations import Relation
from .steps import KINDS

ROLES = ("compression", "expansion", "heat_in", "heat_out", "internal")
SYSTEMS = ("closed", "flow")
CYCLE_KINDS = ("power",)

# Result names that live in (or next to) the equation system; parameters may not use them.
RESULT_NAMES = ("q_in", "q_out", "w_net", "eta_th", "mep", "bwr", "w_compression", "w_expansion",
                "eta_carnot", "eta_ii")


@dataclass(frozen=True)
class StepDef:
    kind: str
    a: int
    b: int
    role: str
    efficiency: "str | None" = None


@dataclass(frozen=True)
class ParamDef:
    """A catalogue parameter.

    ``relation`` defines it from state properties (e.g. ``ratio("r_c", "v1", "v2")``). It may be ``None``
    for a free parameter that only appears in a step (an adiabatic efficiency) or an extra relation; such a
    parameter normally carries a ``default``.
    """

    name: str
    description: str
    relation: "Relation | None"
    default: "float | None" = None
    bounds: "tuple[float, float] | None" = None   # validity range, also the scan range


@dataclass(frozen=True)
class CycleDef:
    """A cycle definition (spec 8.1).

    ``regenerators`` declares the regenerator stream pairs: one entry ``((a_c, b_c), (a_h, b_h))`` per
    regenerator, the cold side first (the step a_c -> b_c whose stream the regenerator heats), then the hot side
    (a_h -> b_h, the stream it cools). Steps are named by their state numbers, as in every message ("2→3"),
    not by position: a pair stays right when steps are added or the loop is renumbered, and a typo is caught
    because (a, b) must be a step of the cycle. Every step with role ``internal`` must be in exactly one pair,
    and pair members must be internal steps (``validate_cycle``). Example, ``brayton_regen``:
    ``regenerators=(((2, 3), (5, 6)),)``. The spec 9.7 regenerator rule uses these sides, never the step order.
    """

    name: str
    title: str
    aliases: "tuple[str, ...]"
    system: Literal["closed", "flow"]
    states: "tuple[int, ...]"
    steps: "tuple[StepDef, ...]"
    params: "tuple[ParamDef, ...]" = ()
    extra_relations: "tuple[Relation, ...]" = ()
    kind: Literal["power"] = "power"
    v_max_state: "int | None" = None
    v_min_state: "int | None" = None
    suggest: "tuple[str, ...]" = ()
    typical_inputs: "tuple[tuple[str, ...], ...]" = ()
    description: str = ""
    references: "tuple[str, ...]" = ()
    regenerators: "tuple[tuple[tuple[int, int], tuple[int, int]], ...]" = ()


# ---------------------------------------------------------------------------------------------------------
# names


def state_ids(defn: CycleDef) -> "tuple[str, ...]":
    """Main state ids as strings, then the hidden ones (``"2s"``) in step order."""
    main = tuple(str(k) for k in defn.states)
    hidden = []
    for st in defn.steps:
        for h in KINDS[st.kind].hidden_states(str(st.b)):
            if h not in hidden:
                hidden.append(h)
    return main + tuple(hidden)


def step_vars(st: StepDef) -> "tuple[str, str]":
    return f"q_{st.a}{st.b}", f"w_{st.a}{st.b}"


def result_vars(defn: CycleDef) -> "tuple[str, ...]":
    base = ("q_in", "q_out", "w_net", "eta_th")
    if defn.system == "closed":
        return base + ("mep",)
    return base + ("w_compression", "w_expansion", "bwr")


def _fail(defn_name, msg):
    raise ValueError(f"cycle definition {defn_name!r}: {msg}")


def validate_cycle(defn: CycleDef) -> None:
    """Check a definition per spec 8.1; raises ``ValueError`` naming the problem."""
    if not isinstance(defn, CycleDef):
        raise ValueError(f"expected a CycleDef, got {type(defn).__name__}")
    nm = defn.name
    if not isinstance(nm, str) or not nm.isidentifier():
        raise ValueError(f"cycle name must be a non-empty identifier string, got {nm!r}")
    if not isinstance(defn.aliases, tuple) or not all(isinstance(a, str) and a for a in defn.aliases):
        _fail(nm, f"aliases must be a tuple of non-empty strings, got {defn.aliases!r}")
    if defn.system not in SYSTEMS:
        _fail(nm, f"system must be one of {SYSTEMS}, got {defn.system!r}")
    if defn.kind not in CYCLE_KINDS:
        _fail(nm, f"kind must be one of {CYCLE_KINDS} in v0.1, got {defn.kind!r}")

    # states and steps: one closed loop through all states in order
    states = tuple(defn.states)
    if len(states) < 2:
        _fail(nm, f"needs at least 2 states, got {states}")
    if not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in states):
        _fail(nm, f"states must be integers >= 1, got {states}")
    if len(set(states)) != len(states):
        _fail(nm, f"duplicate state ids in {states}")
    steps = tuple(defn.steps)
    if not all(isinstance(s, StepDef) for s in steps):
        _fail(nm, "steps must be StepDef instances")
    if len(steps) != len(states):
        _fail(nm, f"{len(steps)} steps for {len(states)} states; the steps must form one closed loop "
                  f"through every state ({' -> '.join(map(str, states + states[:1]))})")
    for i, st in enumerate(steps):
        a, b = states[i], states[(i + 1) % len(states)]
        if (st.a, st.b) != (a, b):
            _fail(nm, f"step {i + 1} is {st.a}->{st.b}, expected {a}->{b}: the steps must form one closed loop "
                      f"through the states in order ({' -> '.join(map(str, states + states[:1]))})")
        if st.kind not in KINDS:
            _fail(nm, f"step {st.a}->{st.b}: unknown kind {st.kind!r}; known: {sorted(KINDS)}")
        if st.role not in ROLES:
            _fail(nm, f"step {st.a}->{st.b}: unknown role {st.role!r}; known: {ROLES}")
        if st.efficiency is not None and not KINDS[st.kind].hidden_states(str(st.b)):
            _fail(nm, f"step {st.a}->{st.b}: kind {st.kind!r} takes no efficiency parameter "
                      f"(use 'adiabatic' for a non-ideal step)")
        try:
            KINDS[st.kind].relations(str(st.a), str(st.b), system=defn.system, role=st.role,
                                     efficiency=st.efficiency)
        except ValueError as e:
            _fail(nm, str(e))
    roles = [s.role for s in steps]
    if "heat_in" not in roles:
        _fail(nm, "a power cycle needs at least one step with role 'heat_in'")
    if "heat_out" not in roles:
        _fail(nm, "a power cycle needs at least one step with role 'heat_out' (Kelvin-Planck)")
    if defn.system == "flow" and ("compression" not in roles or "expansion" not in roles):
        _fail(nm, "a flow cycle needs a 'compression' and an 'expansion' step (for w_compression, "
                  "w_expansion and bwr)")

    _validate_regenerators(defn, steps)

    # piston volumes for the MEP
    if defn.system == "closed":
        for label, k in (("v_max_state", defn.v_max_state), ("v_min_state", defn.v_min_state)):
            if k not in states:
                _fail(nm, f"a closed cycle needs {label} (one of the states {states}) for the MEP, got {k!r}")
        if defn.v_max_state == defn.v_min_state:
            _fail(nm, "v_max_state and v_min_state must differ")
    elif defn.v_max_state is not None or defn.v_min_state is not None:
        _fail(nm, "v_max_state / v_min_state only apply to closed cycles (the MEP)")

    # names
    sids = state_ids(defn)
    reserved = {f"{p}{sid}" for sid in sids for p in PROPS}
    reserved |= {n for st in steps for n in step_vars(st)}
    reserved |= set(RESULT_NAMES)
    params = tuple(defn.params)
    if not all(isinstance(p, ParamDef) for p in params):
        _fail(nm, "params must be ParamDef instances")
    pnames = [p.name for p in params]
    if len(set(pnames)) != len(pnames):
        _fail(nm, f"duplicate parameter names in {pnames}")
    lowered = {r.lower(): r for r in reserved}
    for p in params:
        if not isinstance(p.name, str) or not p.name.isidentifier():
            _fail(nm, f"parameter name must be an identifier, got {p.name!r}")
        if p.name.lower() in lowered:
            _fail(nm, f"parameter name {p.name!r} clashes with the property or result name "
                      f"{lowered[p.name.lower()]!r}")
    known = {f"{p}{sid}" for sid in sids for p in PROPS} | {n for st in steps for n in step_vars(st)}
    known |= set(result_vars(defn)) | set(pnames)
    for p in params:
        if p.bounds is not None:
            if len(p.bounds) != 2 or any(math.isnan(float(x)) for x in p.bounds) or not p.bounds[0] < p.bounds[1]:
                _fail(nm, f"parameter {p.name!r}: bounds must be (lo, hi) with lo < hi, got {p.bounds!r}")
        if p.default is not None:
            d = float(p.default)
            if not math.isfinite(d):
                _fail(nm, f"parameter {p.name!r}: default must be finite, got {p.default!r}")
            if p.bounds is not None and not p.bounds[0] <= d <= p.bounds[1]:
                _fail(nm, f"parameter {p.name!r}: default {d:g} lies outside its bounds {p.bounds}")
        if p.relation is not None:
            if not isinstance(p.relation, Relation):
                _fail(nm, f"parameter {p.name!r}: relation must be a Relation or None")
            if p.name not in p.relation.vars:
                _fail(nm, f"parameter {p.name!r}: its relation {p.relation.name!r} does not contain it "
                          f"(vars {p.relation.vars})")
            unknown = [v for v in p.relation.vars if v not in known]
            if unknown:
                _fail(nm, f"parameter {p.name!r}: relation {p.relation.name!r} uses unknown variables {unknown}")
    for r in defn.extra_relations:
        if not isinstance(r, Relation):
            _fail(nm, "extra_relations must be Relation instances")
        unknown = [v for v in r.vars if v not in known]
        if unknown:
            _fail(nm, f"extra relation {r.name!r} uses unknown variables {unknown}")
    used = {st.efficiency for st in steps if st.efficiency is not None}
    used |= {v for r in defn.extra_relations for v in r.vars}
    for st in steps:
        if st.efficiency is not None and st.efficiency not in pnames:
            _fail(nm, f"step {st.a}->{st.b}: efficiency {st.efficiency!r} is not a declared parameter")
    for p in params:
        if p.relation is None and p.name not in used:
            _fail(nm, f"parameter {p.name!r} has no relation and is not used by any step or extra relation")

    # suggestions and typical inputs must name real variables
    variables = known
    bad = [s for s in defn.suggest if s not in variables]
    if bad:
        _fail(nm, f"suggest names unknown variables {bad}")
    for ti in defn.typical_inputs:
        bad = [s for s in ti if s not in variables]
        if bad:
            _fail(nm, f"typical_inputs {ti} name unknown variables {bad}")


def _validate_regenerators(defn: CycleDef, steps: "tuple[StepDef, ...]") -> None:
    """Spec 8.1 ``regenerators``: well-formed ((a_c, b_c), (a_h, b_h)) pairs of distinct internal steps, and every
    internal step in exactly one pair."""
    nm = defn.name
    regens = defn.regenerators
    form = "((a_c, b_c), (a_h, b_h)): the cold-side step, then the hot-side step"
    if not isinstance(regens, tuple):
        _fail(nm, f"regenerators must be a tuple of pairs {form}, got {type(regens).__name__} {regens!r}")
    by_ab = {(st.a, st.b): st for st in steps}
    seen: "dict[tuple[int, int], int]" = {}
    for i, pair in enumerate(regens):
        where = f"regenerators[{i}] = {pair!r}"
        if not (isinstance(pair, tuple) and len(pair) == 2
                and all(isinstance(side, tuple) and len(side) == 2 for side in pair)):
            _fail(nm, f"{where} must be a pair of steps {form}")
        if not all(isinstance(k, int) and not isinstance(k, bool) for side in pair for k in side):
            _fail(nm, f"{where}: state numbers must be integers")
        cold, hot = pair
        if cold == hot:
            _fail(nm, f"{where}: cold and hot side are the same step {cold[0]}→{cold[1]}")
        for side, (a, b) in (("cold", cold), ("hot", hot)):
            st = by_ab.get((a, b))
            if st is None:
                _fail(nm, f"{where}: {side} side step {a}→{b} is not a step of the cycle "
                          f"(steps: {', '.join(f'{s.a}→{s.b}' for s in steps)})")
            if st.role != "internal":
                _fail(nm, f"{where}: {side} side step {a}→{b} has role {st.role!r}, not 'internal'")
            if (a, b) in seen:
                _fail(nm, f"step {a}→{b} is in more than one regenerator pair (regenerators[{seen[(a, b)]}] and "
                          f"regenerators[{i}])")
            seen[(a, b)] = i
    for st in steps:
        if st.role == "internal" and (st.a, st.b) not in seen:
            others = [f"({s.a}, {s.b})" for s in steps if s.role == "internal" and s is not st]
            _fail(nm, f"internal step {st.a}→{st.b} is not in any declared regenerator pair. Declare it with "
                      f"regenerators=(((a_c, b_c), (a_h, b_h)),), cold side (the heated stream) first"
                      + (f"; other internal steps: {', '.join(others)}" if others else "") + ".")


# ---------------------------------------------------------------------------------------------------------
# registry

_REGISTRY: "dict[str, CycleDef]" = {}


def _keys(defn: CycleDef) -> "list[str]":
    return [defn.name.lower()] + [a.lower() for a in defn.aliases]


def register_cycle(defn: CycleDef, *, replace: bool = False) -> None:
    """Validate ``defn`` (spec 8.1) and add it to the registry.

    Raises ``ValueError`` on a bad definition, or when its name or an alias is already taken
    (case-insensitive) and ``replace`` is false. With ``replace=True`` a cycle of the same name is replaced.
    """
    validate_cycle(defn)
    keys = _keys(defn)
    if len(set(keys)) != len(keys):
        raise ValueError(f"cycle definition {defn.name!r}: name and aliases must be unique "
                         f"(case-insensitive), got {keys}")
    for other in _REGISTRY.values():
        if replace and other.name == defn.name:
            continue
        clash = set(keys) & set(_keys(other))
        if clash:
            if other.name.lower() == defn.name.lower() and not replace:
                raise ValueError(f"cycle {other.name!r} is already registered; pass replace=True to replace it")
            raise ValueError(f"cycle definition {defn.name!r}: name/alias {sorted(clash)} already used by "
                             f"cycle {other.name!r}")
    _REGISTRY[defn.name] = defn


def list_cycles() -> "list[str]":
    """Registered cycle names, sorted."""
    return sorted(_REGISTRY)


def get_cycle(name: str) -> CycleDef:
    """Look a cycle up by exact name, then case-insensitively, then by alias.

    Raises ``UnknownCycleError`` with difflib suggestions (canonical names) otherwise.
    """
    if not isinstance(name, str):
        raise UnknownCycleError(f"cycle name must be a string, got {type(name).__name__} {name!r}. "
                                f"Known cycles: {', '.join(list_cycles())}", suggestions=())
    if name in _REGISTRY:
        return _REGISTRY[name]
    low = name.strip().lower()
    for defn in _REGISTRY.values():
        if low in _keys(defn):
            return defn
    lookup = {k: d.name for d in _REGISTRY.values() for k in _keys(d)}
    close = difflib.get_close_matches(low, list(lookup), n=5, cutoff=0.6)
    suggestions = tuple(dict.fromkeys(lookup[c] for c in close))
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise UnknownCycleError(f"unknown cycle {name!r}.{hint} Known cycles: {', '.join(list_cycles())}",
                            suggestions=suggestions)
