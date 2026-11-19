"""Compile a cycle definition into a system of scalar relations (spec sections 9.1, 9.4, 9.5)."""

from __future__ import annotations

import difflib
import math
import numbers
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..errors import CycleDefinitionError, InputError, InputPlausibilityWarning, UnknownInputError
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
    fluid: object = None                                  # the fluid, for "closure" blocks (Task 6)

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
        raise CycleDefinitionError(f"cycle {defn.name!r}: duplicate variable names in the equation system")
    for r in relations:
        bad = [v for v in r.vars if v not in vset]
        if bad:
            raise CycleDefinitionError(f"cycle {defn.name!r}: relation {r.name!r} uses unknown variables {bad}")

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
        equivalent=MappingProxyType(equivalent), fluid=fluid)


# ---------------------------------------------------------------------------------------------------------
# Input resolution (spec 3 look-up order, 9.8 validation; plan Task 7)

# Old spellings that mean different things in different books: suggest every meaning the cycle has.
# Cengel & Boles call the cutoff ratio r_c and the dual-cycle pressure ratio r_p (spec 3, naming clash).
_AMBIGUOUS_OLD = {"rc": ("r_c", "cutoff"), "rp": ("r_p", "alpha"), "re": ("r_e",)}
_AMBIGUOUS_NOTE = {
    "rc": "(Cengel & Boles call the cutoff ratio r_c; here r_c is the compression ratio v1/v2.)",
    "rp": "(Cengel & Boles call the dual-cycle pressure ratio r_p; here that is alpha, and r_p is the "
          "Brayton pressure ratio.)"}
# Quantities computed after the solve (spec 9.1): they involve max/min and cannot be inputs.
_POST_SOLVE = ("eta_carnot", "eta_ii", "t_max", "t_min", "p_max", "p_min", "v_max", "v_min")

_UNITS = {"T": "K", "P": "Pa", "v": "m^3/kg", "u": "J/kg", "h": "J/kg", "s": "J/(kg*K)"}


def _cycle_name(system: EquationSystem) -> str:
    return system.cycle.name if system.cycle is not None else "system"


def _unit(system: EquationSystem, name: str) -> str:
    """SI unit of a variable, for messages ('' for ratios and efficiencies)."""
    sids = state_ids(system.cycle) if system.cycle is not None else ()
    if name[:1] in _UNITS and name[1:] in sids:
        return _UNITS[name[:1]]
    if name in ("q_in", "q_out", "w_net", "w_compression", "w_expansion") or name[:2] in ("q_", "w_"):
        return "J/kg"
    if name == "mep":
        return "Pa"
    if name.startswith("dT"):
        return "K"
    return ""


def _with_unit(value: float, unit: str) -> str:
    return f"{value:g} {unit}" if unit else f"{value:g}"


def known_names_text(system: EquationSystem, noun: str = "inputs") -> str:
    """Compact list of every name ``resolve_name`` accepts, grouped (v0.1.1), e.g. "Known inputs: state
    properties T, P, v, u, h, s at states 1, 2, 3, 4 (e.g. T3; hidden states 2s, 4s); parameters r_p, ...;
    step heat and work q_12, w_12, ...; cycle quantities q_in, ... (see describe('brayton'))."
    """
    defn = system.cycle
    if defn is None:
        return f"Known {noun}: {', '.join(system.variables)}."
    sids = state_ids(defn)
    main = [str(k) for k in defn.states]
    hidden = [h for h in sids if h not in main]
    state_vars = {f"{p}{sid}" for sid in sids for p in _STATE_PROPS}
    params = [p.name for p in defn.params]
    steps = [v for v in system.variables if v[:2] in ("q_", "w_") and v[2:].isdigit()]
    other = [v for v in system.variables if v not in state_vars and v not in params and v not in steps]
    parts = [f"state properties {', '.join(_STATE_PROPS)} at states {', '.join(main)} (e.g. T{main[-1]}"
             + (f"; hidden states {', '.join(hidden)})" if hidden else ")")]
    if params:
        parts.append(f"parameters {', '.join(params)}")
    if steps:
        parts.append(f"step heat and work {', '.join(steps)}")
    if other:
        parts.append(f"cycle quantities {', '.join(other)}")
    return f"Known {noun}: " + "; ".join(parts) + f" (see describe({defn.name!r}))."


_STATE_PROPS = ("T", "P", "v", "u", "h", "s")


def resolve_name(system: EquationSystem, name: object) -> str:
    """Canonical variable name for ``name`` (spec 3): exact, then unique case-insensitive, then alias.

    The alias table is matched exactly (``"R"`` is not taken for the alias ``"r"``: it is far more likely
    the gas constant). Anything else raises ``UnknownInputError`` with difflib suggestions; old spellings
    that mean different things in different books (``rc``: r_c or cutoff) suggest every meaning.
    """
    cyc = _cycle_name(system)
    if not isinstance(name, str):
        raise UnknownInputError(f"{cyc}: input names must be strings, got {name!r}.",
                                name=name, suggestions=(), accepted=system.variables)
    if name in system.variables:
        return name
    low = name.lower()
    ci = [v for v in system.variables if v.lower() == low]
    if len(ci) == 1:
        return ci[0]
    if name in system.aliases:
        return system.aliases[name]
    if ci:      # two variables differing only in case (not in the v0.1 catalogue)
        raise UnknownInputError(f"{cyc}: input {name!r} matches {', '.join(ci)} when case is ignored; "
                                f"use the exact name.", name=name, suggestions=tuple(ci),
                                accepted=system.variables)
    if low in _POST_SOLVE:
        raise UnknownInputError(f"{cyc}: {name!r} is computed after the solve (it involves a max or min over "
                                f"the cycle) and cannot be an input.", name=name, suggestions=(),
                                accepted=system.variables)
    key = low.replace("_", "")
    known = set(system.variables)
    meanings = [s for s in _AMBIGUOUS_OLD.get(key, ()) if s in known]
    sugg = list(meanings)
    lower_map = {v.lower(): v for v in system.variables}
    lower_map.update({a.lower(): t for a, t in system.aliases.items()})
    for m in difflib.get_close_matches(low, list(lower_map), n=5, cutoff=0.6):
        if lower_map[m] not in sugg:
            sugg.append(lower_map[m])
    hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
    if len(meanings) > 1 and key in _AMBIGUOUS_NOTE:
        hint += " " + _AMBIGUOUS_NOTE[key]
    sid = name[1:]
    if (system.cycle is not None and name[:1] in _STATE_PROPS and sid and sid not in state_ids(system.cycle)
            and (sid.isdigit() or sid[:-1].isdigit())):
        hint += f" {cyc} has no state {sid} (states: {', '.join(str(k) for k in system.cycle.states)})."
    raise UnknownInputError(f"{cyc}: unknown input {name!r}.{hint}\n  {known_names_text(system)}", name=name,
                            suggestions=tuple(sugg), accepted=system.variables)


def _to_float(cyc: str, name: str, value: object) -> float:
    """One value per spec 9.8: real and finite; ints and numpy scalars accepted; bools, strings not."""
    if isinstance(value, (bool, np.bool_)):
        raise InputError(f"{cyc}: {name} = {value!r} is a bool; give a number.")
    if isinstance(value, str):
        raise InputError(f"{cyc}: {name} = {value!r} is a string; give a number ({name}=300, not '300').")
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        raise InputError(f"{cyc}: {name} = {value!r} is complex; inputs must be real numbers.")
    if not isinstance(value, numbers.Real):
        raise InputError(f"{cyc}: {name} = {value!r} ({type(value).__name__}) is not a real number; "
                         f"give one scalar value.")
    x = float(value)
    if not math.isfinite(x):
        raise InputError(f"{cyc}: {name} = {x!r} is not finite; inputs must be finite real numbers.")
    return x


PLAUSIBILITY_MODES = ("raise", "warn", "off")
PLAUSIBLE_P_MIN = 1000.0     # Pa: a given state pressure below this usually means kPa or bar (spec 3, 9.8)
PLAUSIBLE_T_MIN = 100.0      # K: a given state temperature below this usually means deg C (spec 3, 9.8)


def check_plausibility_mode(plausibility: object) -> str:
    """``plausibility`` validated against ``PLAUSIBILITY_MODES`` (else ``InputError``)."""
    if not isinstance(plausibility, str) or plausibility not in PLAUSIBILITY_MODES:
        raise InputError(f"plausibility must be one of {PLAUSIBILITY_MODES}, got {plausibility!r}.")
    return plausibility


def state_mixup_text(defn, given: "Mapping[str, float]") -> "str | None":
    """The ``CycleDef.state_mixups`` message for the first given input at or above its limit, else None.

    Example (brayton_regen): "T3 = 1250 K is the regenerator cold outlet in brayton_regen; the turbine inlet is
    T4. Did you mean T4=1250?" """
    for mx in getattr(defn, "state_mixups", ()) or ():
        x = given.get(mx.given)
        if x is not None and x >= mx.limit:
            unit = " K" if mx.given[:1] == "T" else ""
            return (f"{mx.given} = {x:g}{unit} is the {mx.given_role} in {defn.name}; the {mx.meant_role} is "
                    f"{mx.meant}. Did you mean {mx.meant}={x:g}?")
    return None


def resolve_inputs(system: EquationSystem, raw: "Mapping[str, object]",
                   plausibility: str = "raise") -> "dict[str, float]":
    """Canonical names and validated float values of user inputs (spec 3 and 9.8), in the given order.

    - Names: exact, then a unique case-insensitive match, then the alias table; else ``UnknownInputError``.
      Two spellings of one variable raise ``InputError``.
    - Values: real and finite (ints and numpy scalars are converted with ``float()``); a variable of
      positive kind must be > 0; every value must lie inside the variable's physical range (the solver's
      scan range, spec 9.5), else ``InputError`` naming the input. Without that check a value such as
      T4 = 2e5 K reaches the solver and the no-solution message blames another input.
    - Plausibility: a given state pressure below 1 000 Pa or state temperature below 100 K usually means kPa,
      bar or degrees Celsius were passed. ``plausibility="raise"`` (default) raises ``InputError`` with the
      fix, ``"warn"`` emits ``InputPlausibilityWarning`` instead (the v0.1 behaviour), ``"off"`` skips it.
    """
    check_plausibility_mode(plausibility)
    cyc = _cycle_name(system)
    sids = state_ids(system.cycle) if system.cycle is not None else ()
    out: "dict[str, float]" = {}
    spelled: "dict[str, str]" = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise InputError(f"{cyc}: input names must be strings, got {key!r} = {value!r}.")
        name = resolve_name(system, key)
        if name in out:
            raise InputError(f"{cyc}: {spelled[name]!r} and {key!r} both give {name}; pass it once.")
        spelled[name] = key
        x = _to_float(cyc, name, value)
        unit = _unit(system, name)
        if system.var_kind.get(name) == "positive" and not x > 0:
            raise InputError(f"{cyc}: {name} = {_with_unit(x, unit)} must be > 0.")
        lo, hi, _ = system.scan_range[name]
        if not lo <= x <= hi:
            hint = ""
            if not unit and hi == 1.0 and 1.0 < x <= 100.0:     # a percentage given for a fraction
                hint = f"; efficiencies and ratios are fractions ({x / 100:g}, not {x:g})"
            raise InputError(f"{cyc}: {name} = {_with_unit(x, unit)} is outside the range thermo_cycles can "
                             f"solve for it ({lo:g} to {_with_unit(hi, unit)}). Check the value and its "
                             f"units: inputs are SI (K, Pa, m^3/kg, J/kg){hint}.")
        if plausibility != "off":
            text = None
            if name[:1] == "P" and name[1:] in sids and x < PLAUSIBLE_P_MIN:
                alt = f"{x:g} bar or {x:g} kPa" if x < 50.0 else f"{x:g} kPa"
                text = (f"{name} = {x:g} Pa: did you mean {alt}? thermo_cycles works in SI "
                        f"(write {name}={x:g}*kPa or {x:g}*bar with thermo_cycles.units).")
            elif name[:1] == "T" and name[1:] in sids and x < PLAUSIBLE_T_MIN:
                text = (f"{name} = {x:g} K: did you mean {x:g} deg C ({x + 273.15:g} K)? thermo_cycles "
                        f"works in SI (write {name}=degC({x:g}) with thermo_cycles.units).")
            if text is not None:
                if plausibility == "raise":
                    raise InputError(f"{cyc}: {text} If {_with_unit(x, unit or 'SI')} is really meant, pass "
                                     f"plausibility='warn' or plausibility='off'.")
                warnings.warn(text, InputPlausibilityWarning, stacklevel=2)
        out[name] = x
    if plausibility != "off" and system.cycle is not None:
        text = state_mixup_text(system.cycle, out)
        if text is not None:
            if plausibility == "raise":
                raise InputError(f"{cyc}: {text} If {text.split(' is the ')[0]} is really meant, pass "
                                 f"plausibility='warn' or plausibility='off'.")
            warnings.warn(text, InputPlausibilityWarning, stacklevel=2)
    return out
