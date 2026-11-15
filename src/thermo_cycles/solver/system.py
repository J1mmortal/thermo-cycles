"""Compile a cycle definition into a system of scalar relations (spec sections 9.1, 9.4, 9.5)."""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..errors import UnknownInputError
from ..fluids.ideal_gas import IdealGas
from ..model.cycle_def import CycleDef, result_vars, state_ids, step_vars, validate_cycle
from ..model.relations import Relation, default_scale, fixed, linear, product
from ..model.steps import KINDS, _prod_diff
from ..state import PROPS

# Scan ranges per spec 9.5: (lo, hi, spacing).
RANGE_T = (1.0, 1e5, "log")
RANGE_P = (1.0, 1e10, "log")
RANGE_V = (1e-6, 1e4, "log")
RANGE_RATIO = (1.0, 1e4, "log")
RANGE_EFFICIENCY = (1e-6, 1.0, "log")
RANGE_FRACTION = (0.0, 1.0, "asinh")       # eps: linear 0..1, treated as asinh
RANGE_SIGNED = (-1e9, 1e9, "asinh")
RANGE_BWR = (1e-6, 1e4, "log")             # not in 9.5: positive, but below 1 for a working gas turbine

# Old-style names that are neither the exact nor a case-insensitive match (spec 3).
_ALIAS_TABLE = {"r": "r_c"}


@dataclass(frozen=True)
class EquationSystem:
    variables: "tuple[str, ...]"
    relations: "tuple[Relation, ...]"
    var_kind: "Mapping[str, str]"                         # "positive" | "signed" | "fraction"
    scan_range: "Mapping[str, tuple[float, float, str]]"  # (lo, hi, "log" | "asinh")
    candidates: "tuple[str, ...]"                         # user-facing input names, ranked per 9.4
    defaults: "Mapping[str, float]"
    aliases: "Mapping[str, str]"
    cycle: CycleDef
    equivalent: "Mapping[str, str]"                       # u_k, h_k -> T_k for an ideal gas

    def with_inputs(self, inputs: "Mapping[str, float]") -> "tuple[Relation, ...]":
        """The system's relations plus one ``fixed`` relation per input and per default not overridden.

        Names must be exact variable names (resolution of case and aliases is ``resolve_inputs``' job).
        """
        known = set(self.variables)
        for name in inputs:
            if name not in known:
                sugg = tuple(difflib.get_close_matches(str(name), self.variables, n=5, cutoff=0.6))
                hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
                raise UnknownInputError(f"{self.cycle.name}: unknown input {name!r}.{hint}",
                                        name=name, suggestions=sugg, accepted=self.variables)
        given = tuple(fixed(n, v) for n, v in inputs.items())
        dflt = tuple(fixed(n, v) for n, v in self.defaults.items() if n not in inputs)
        return self.relations + given + dflt


# ---------------------------------------------------------------------------------------------------------


def _closure(fluid, sid: str) -> "list[Relation]":
    """Generic per-state closure for a fluid without sparse relations (spec 9.1).

    Four dense relations over all six properties of the state: each one says that one of the four
    non-pair properties equals ``fluid.state(pair)``. Any two properties fix the rest, so every relation
    lists all six variables. Names start with ``"closure "`` so the block solver can spot the pattern.
    """
    pair = ("T", "s") if fluid.independent("T", "s") else ("P", "h")
    names = {p: f"{p}{sid}" for p in PROPS}
    vars_ = tuple(names[p] for p in PROPS)
    i0, i1 = PROPS.index(pair[0]), PROPS.index(pair[1])
    rels = []
    for p in PROPS:
        if p in pair:
            continue
        ip = PROPS.index(p)

        def residual(*vals, ip=ip, p=p):
            return vals[ip] - fluid.state(**{pair[0]: vals[i0], pair[1]: vals[i1]})[p]

        def solver(p=p, **k):
            return fluid.state(**{pair[0]: k[names[pair[0]]], pair[1]: k[names[pair[1]]]})[p]

        rels.append(Relation(f"closure {sid} {p}({pair[0]},{pair[1]})", vars_, residual,
                             {names[p]: solver},
                             default_scale(names[p])))
    return rels


def _bwr_relation() -> Relation:
    """bwr * w_expansion + w_compression = 0 (bwr = -w_compression / w_expansion)."""
    nm = "bwr = -w_compression/w_expansion"

    def div(x, y, var):
        if y == 0:
            raise ZeroDivisionError(f"relation {nm!r}: cannot solve for {var!r}, the divisor is zero")
        return x / y

    return Relation(nm, ("bwr", "w_compression", "w_expansion"),
                    lambda b, wc, we: b * we + wc,
                    {"bwr": lambda **k: div(-k["w_compression"], k["w_expansion"], "bwr"),
                     "w_compression": lambda **k: -k["bwr"] * k["w_expansion"],
                     "w_expansion": lambda **k: div(-k["w_compression"], k["bwr"], "w_expansion")},
                    1e5)


def _param_kind(p) -> "tuple[str, tuple[float, float, str]]":
    n = p.name
    if n.startswith("r_") or n in ("cutoff", "alpha"):
        kind, rng = "positive", RANGE_RATIO
    elif n.startswith("eta"):
        kind, rng = "positive", RANGE_EFFICIENCY
    elif n == "eps":
        kind, rng = "fraction", RANGE_FRACTION
    elif n.startswith("dT"):
        kind, rng = "signed", RANGE_SIGNED
    elif p.bounds is not None and p.bounds[0] > 0:
        hi = p.bounds[1] if math.isfinite(p.bounds[1]) else max(1e4, 10.0 * p.bounds[0])
        kind, rng = "positive", (p.bounds[0], hi, "log")
    elif p.bounds is not None and p.bounds[0] == 0 and math.isfinite(p.bounds[1]):
        kind, rng = "fraction", (0.0, p.bounds[1], "asinh")
    else:
        kind, rng = "signed", RANGE_SIGNED
    if p.bounds is not None:   # bounds are also the scan range: intersect
        lo = max(rng[0], p.bounds[0]) if math.isfinite(p.bounds[0]) else rng[0]
        hi = min(rng[1], p.bounds[1]) if math.isfinite(p.bounds[1]) else rng[1]
        if lo < hi:
            rng = (lo, hi, rng[2])
    return kind, rng


def build_system(defn: CycleDef, fluid) -> EquationSystem:
    """Variables and relations of ``defn`` with ``fluid`` (spec 9.1); inputs are added by ``with_inputs``."""
    validate_cycle(defn)
    if fluid is None or not callable(getattr(fluid, "relations", None)):
        raise TypeError(f"build_system needs a fluid with a relations(sid) method, got {fluid!r}")
    sids = state_ids(defn)
    main = tuple(str(k) for k in defn.states)
    variables: "list[str]" = []
    relations: "list[Relation]" = []
    kind: "dict[str, str]" = {}
    rng: "dict[str, tuple[float, float, str]]" = {}

    def add(var, k, r):
        variables.append(var)
        kind[var] = k
        rng[var] = r

    # states (main, then hidden): 6 properties and the fluid's 4 relations
    for sid in sids:
        for p in PROPS:
            if p == "T":
                add(f"T{sid}", "positive", RANGE_T)
            elif p == "P":
                add(f"P{sid}", "positive", RANGE_P)
            elif p == "v":
                add(f"v{sid}", "positive", RANGE_V)
            else:
                add(f"{p}{sid}", "signed", RANGE_SIGNED)
        fr = fluid.relations(sid)
        relations.extend(_closure(fluid, sid) if fr is None else fr)

    # steps: q_ab, w_ab and the kind's relations
    for st in defn.steps:
        for v in step_vars(st):
            add(v, "signed", RANGE_SIGNED)
        relations.extend(KINDS[st.kind].relations(str(st.a), str(st.b), system=defn.system, role=st.role,
                                                  efficiency=st.efficiency))

    # parameters and their definitions
    for p in defn.params:
        add(p.name, *_param_kind(p))
        if p.relation is not None:
            relations.append(p.relation)

    # cycle sums
    for r in result_vars(defn):
        if r == "eta_th":
            add(r, "positive", RANGE_EFFICIENCY)
        elif r == "bwr":
            add(r, "positive", RANGE_BWR)
        else:
            add(r, "signed", RANGE_SIGNED)
    q = {st: step_vars(st)[0] for st in defn.steps}
    w = {st: step_vars(st)[1] for st in defn.steps}
    relations.append(linear("q_in", {q[s]: 1.0 for s in defn.steps if s.role == "heat_in"},
                            "q_in = sum q(heat_in)"))
    relations.append(linear("q_out", {q[s]: -1.0 for s in defn.steps if s.role == "heat_out"},
                            "q_out = -sum q(heat_out)"))
    relations.append(linear("w_net", {w[s]: 1.0 for s in defn.steps}, "w_net = sum w"))
    relations.append(product("w_net", "eta_th", "q_in", "eta_th = w_net/q_in"))
    if defn.system == "closed":
        relations.append(_prod_diff("w_net", "mep", f"v{defn.v_max_state}", f"v{defn.v_min_state}",
                                    f"mep = w_net/(v{defn.v_max_state} - v{defn.v_min_state})"))
    else:
        relations.append(linear("w_compression", {w[s]: 1.0 for s in defn.steps if s.role == "compression"},
                                "w_compression = sum w(compression)"))
        relations.append(linear("w_expansion", {w[s]: 1.0 for s in defn.steps if s.role == "expansion"},
                                "w_expansion = sum w(expansion)"))
        relations.append(_bwr_relation())

    relations.extend(defn.extra_relations)

    # every relation must only use known variables
    vset = set(variables)
    if len(vset) != len(variables):
        raise ValueError(f"cycle {defn.name!r}: duplicate variable names in the equation system")
    for r in relations:
        bad = [v for v in r.vars if v not in vset]
        if bad:
            raise ValueError(f"cycle {defn.name!r}: relation {r.name!r} uses unknown variables {bad}")

    # ideal-gas equivalents: u_k and h_k carry the same information as T_k
    equivalent = {}
    if isinstance(fluid, IdealGas):
        for sid in sids:
            equivalent[f"u{sid}"] = f"T{sid}"
            equivalent[f"h{sid}"] = f"T{sid}"

    # candidate ranking (spec 9.4): suggest order, then T, P, v by state, then results, then u/h/s,
    # then step q/w. Hidden-state properties are accepted as inputs but never suggested.
    ranked: "list[str]" = list(defn.suggest)
    ranked += [f"{p}{sid}" for sid in main for p in ("T", "P", "v")]
    ranked += [p.name for p in defn.params]
    ranked += [r for r in ("q_in", "q_out", "w_net", "eta_th", "mep", "bwr") if r in vset]
    ranked += [f"{p}{sid}" for sid in main for p in ("u", "h", "s") if f"{p}{sid}" not in equivalent]
    ranked += [v for st in defn.steps for v in step_vars(st)]
    ranked += [r for r in ("w_compression", "w_expansion") if r in vset]
    candidates = tuple(dict.fromkeys(ranked))

    defaults = {p.name: float(p.default) for p in defn.params if p.default is not None}
    aliases = {a: t for a, t in _ALIAS_TABLE.items() if t in vset and a not in vset}

    return EquationSystem(
        variables=tuple(variables), relations=tuple(relations),
        var_kind=MappingProxyType(kind), scan_range=MappingProxyType(rng), candidates=candidates,
        defaults=MappingProxyType(defaults), aliases=MappingProxyType(aliases), cycle=defn,
        equivalent=MappingProxyType(equivalent))
