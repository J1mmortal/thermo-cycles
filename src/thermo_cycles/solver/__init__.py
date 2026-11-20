"""The general solver: structure, numeric block solve, input advice, and ``solve_cycle`` that ties them together
(spec section 9)."""

from __future__ import annotations

import dataclasses
import warnings
from collections import OrderedDict
from types import MappingProxyType
from typing import Mapping

from ..errors import (ConvergenceError, CycleValidityWarning, InconsistentInputError, InvalidCycleError,
                      NoSolutionError)
from ..model.cycle_def import CycleDef
from .advice import Requirements, analyse, raise_for
from .numeric import SolveReport, _fmt, _solve, solve_numeric
from .system import (PLAUSIBILITY_MODES, state_mixup_text, EquationSystem, build_system, check_plausibility_mode, resolve_inputs,
                     resolve_name)

__all__ = ["solve_cycle", "system_for", "REDUNDANT_RTOL", "CHECK_MODES",
           "Requirements", "analyse", "raise_for", "SolveReport", "solve_numeric",
           "EquationSystem", "build_system", "resolve_inputs", "resolve_name", "PLAUSIBILITY_MODES",
           "check_plausibility_mode"]

REDUNDANT_RTOL = 1e-6                 # spec 9.6: |given - implied| <= 1e-6 max(|given|, 1)
CHECK_MODES = ("raise", "warn", "off")
_CACHE_SIZE = 32
_SYSTEMS: "OrderedDict[tuple[int, int], tuple[CycleDef, object, EquationSystem]]" = OrderedDict()


def system_for(defn: CycleDef, fluid) -> EquationSystem:
    """``build_system(defn, fluid)``, cached per (definition, fluid) object (small LRU).

    Caching keeps the numeric reference Jacobians of ``analyse`` (cached per system) alive between cycles of
    the same kind, so a second ``Cycle("otto", ...)`` does not rebuild them.
    """
    key = (id(defn), id(fluid))
    hit = _SYSTEMS.get(key)
    if hit is not None and hit[0] is defn and hit[1] is fluid:
        _SYSTEMS.move_to_end(key)
        return hit[2]
    system = build_system(defn, fluid)
    _SYSTEMS[key] = (defn, fluid, system)
    while len(_SYSTEMS) > _CACHE_SIZE:
        _SYSTEMS.popitem(last=False)
    return system


def _drop_redundant(system: EquationSystem, given: "dict[str, float]", req: Requirements):
    """Spec 9.6 step 1: drop inputs of ``drop_one_of`` until the status is no longer ``over``.

    Order: the input given last is dropped first (see the module notes in REPORT/commit: the spec's
    "reverse ranking order" would drop T1 before T3 in Otto(T1, P1, r_c, q_in, T3), contradicting both the
    plan's test and the spec's own example "r_e = 14 given, but T1, P1, r_c, T3 imply r_e = 15.88").
    """
    active = dict(given)
    dropped: "dict[str, float]" = {}
    while req.status == "over":
        choice = next((n for n in reversed(list(active)) if n in req.drop_one_of), None)
        if choice is None:            # only defaulted parameters conflict: nothing the user gave can go
            break
        dropped[choice] = active.pop(choice)
        req = analyse(system, list(active))
    return active, dropped, req


# Results that are a quotient (eta_th = w_net/q_in, mep = w_net/(v_max - v_min), bwr = -w_compression/w_expansion)
# and appear in no other relation: when one of them hits a zero divisor the rest of the cycle is still determined.
_QUOTIENT_RESULTS = ("eta_th", "mep", "bwr")
_UNDEFINED_NOTE = ("  (raised for every check mode, 'raise', 'warn' and 'off': an undefined eta_th, mep or bwr is "
                   "never returned as a result.)")


def _undefined_error(defn: CycleDef, problems: "list[str]", undefined: "list[str]") -> InvalidCycleError:
    what = ", ".join(dict.fromkeys(p.split(":", 1)[0] for p in undefined))
    msg = (f"{defn.name}: the cycle has {what}, so a result is undefined:\n  - " + "\n  - ".join(problems)
           + "\n" + _UNDEFINED_NOTE)
    return InvalidCycleError(msg, problems=tuple(problems))


def _diagnose_zero_divisor(defn: CycleDef, fluid, system: EquationSystem, active: "dict[str, float]",
                           err: "NoSolutionError | ConvergenceError",
                           any_failure: bool = False) -> "InvalidCycleError | None":
    """After ``err`` (a quotient result could not be computed, e.g. "eta_th = w_net/q_in ... the divisor is
    zero"), re-solve without the quotient results that are not inputs and check whether a result is undefined
    (``undefined_results``). Returns the ``InvalidCycleError`` to raise, or None to keep ``err``.

    ``any_failure`` (v0.1.1): also for a ``ConvergenceError`` (singular Jacobian at the solution found), whose
    variables are the whole system; zero heat input makes eta_th = w_net/q_in singular, so the degenerate
    guard used to fire before this diagnosis (diesel T1=330, P1=120e3, r_c=16, cutoff=1, T0091)."""
    from ..results import step_results, undefined_results, validity_problems

    failed = set(getattr(err, "variables", ()) or ())
    drop = {r for r in _QUOTIENT_RESULTS if r in system.variables and r not in active}
    if not any_failure and (not failed or not failed <= drop):
        return None
    if not drop:
        return None
    rels = [r for r in system.relations if not (set(r.vars) & drop)]
    if any(sum(1 for r in system.relations if v in r.vars) != 1 for v in drop):
        return None                                   # a quotient result is coupled elsewhere: not a leaf
    core = dataclasses.replace(system, variables=tuple(v for v in system.variables if v not in drop),
                               relations=tuple(rels))
    # Nested solve: no regularity filter. At a degenerate point (cutoff = 1, zero heat) the Jacobian of the core
    # system can be singular, but the values are only used to word the error, never returned as a result.
    try:
        values = _solve(core, active, None, {}, True)[0]
    except Exception:                                 # noqa: BLE001 - keep the original, more direct error
        return None
    steps = step_results(defn, values)
    undefined = undefined_results(defn, values, steps)
    if not undefined:
        return None
    return _undefined_error(defn, validity_problems(defn, values, steps), undefined)


_PROPS = ("T", "P", "v", "u", "h", "s")


def _state_of(name: str, main: "set[str]") -> "str | None":
    """Main state id of a state-property variable (``"T4"`` -> ``"4"``, ``"h5s"`` -> ``"5"``), else None."""
    if name[:1] not in _PROPS:
        return None
    sid = name[1:]
    if sid in main:
        return sid
    if sid.endswith("s") and sid[:-1] in main:        # hidden isentropic end state of a non-ideal step
        return sid[:-1]
    return None


def _explain_no_solution(defn: CycleDef, given: "Mapping[str, float]", err: NoSolutionError) -> str:
    """Lines naming the user inputs involved in a failed solve and the role of the state each sets (v0.1.1).

    Generic: the unsolved states come from ``err.variables``; the inputs involved are the given state
    properties at those states or one step away, the given parameters whose relation or step touches them, and
    the given whole-cycle quantities (q_in, w_net, ...). The words come from ``model.roles.state_roles``.
    """
    from ..model.roles import state_roles

    roles = state_roles(defn)
    main = {str(k) for k in defn.states}
    unsolved = [sid for sid in dict.fromkeys(_state_of(v, main) for v in (err.variables or ())) if sid]
    near = set(unsolved)
    for st in defn.steps:
        if str(st.a) in unsolved or str(st.b) in unsolved:
            near |= {str(st.a), str(st.b)}
    params = {p.name: p for p in defn.params}
    lines = []
    given_states = set()
    for name, value in given.items():
        sid = _state_of(name, main)
        if sid is not None:
            given_states.add(sid)
            if sid not in near:
                continue
            words = roles[int(sid)] if int(sid) in roles else "state"
            head, _, ctx = words.partition(" (")
            where = f"state {sid}, {ctx[:-1]}" if ctx else f"state {sid}"
            lines.append(f"{name} = {_fmt(name, value)} is the {head} in {defn.name} ({where}).")
        elif name in params:
            p = params[name]
            touched = set()
            if p.relation is not None:
                touched = {_state_of(v, main) for v in p.relation.vars}
            touched |= {str(x) for st in defn.steps if st.efficiency == name for x in (st.a, st.b)}
            if not unsolved or touched & near:
                lines.append(f"{name} = {_fmt(name, value)} is the {p.description}.")
        else:
            lines.append(f"{name} = {_fmt(name, value)} is a whole-cycle quantity.")
    if not lines:
        return ""
    out = ["  Inputs involved and the role of the state they set:"] + [f"    {x}" for x in lines]
    missing = [sid for sid in unsolved if sid not in given_states][:3]
    if missing:
        parts = []
        for sid in missing:
            words = roles.get(int(sid), "state")
            head, _, ctx = words.partition(" (")
            parts.append(f"the {head} is T{sid} (state {sid}" + (f", {ctx}" if ctx else ")"))
        out.append(f"  Not solved: {'; '.join(parts)}.")
    out.append(f"  Check that each value is given at the state you meant: describe({defn.name!r}) lists the role "
               f"of every state.")
    return "\n" + "\n".join(out)


_SENS_REL = 1e-4                      # relative perturbation of a given input for the heat-sign sensitivity
_THRESHOLD_STRETCH = (1.0, 1.05, 1.2, 1.5, 2.0, 4.0)   # multiples of the linear estimate tried as bracket ends


def _fmt_threshold(system: EquationSystem, name: str, x: float) -> str:
    from .system import _unit
    unit = _unit(system, name)
    if unit == "K":
        return f"{x:.1f} K"
    if unit == "Pa":
        return f"{x / 1e3:.1f} kPa"
    if unit == "J/kg":
        return f"{x / 1e3:.1f} kJ/kg"
    return f"{x:.4g}" + (f" {unit}" if unit else "")


def _heat_hints(defn: CycleDef, system: EquationSystem, active: "dict[str, float]", steps,
                problems: "list[str]") -> "dict[tuple[int, int], str]":
    """Input-aware advice for heat_in steps with q < 0 (v0.1.1, T0091 open issue 1).

    For each such step, perturb every given input (relative 1e-4) and re-solve: an input whose increase raises
    the step's q helps when raised, one whose increase lowers q helps when lowered, one without effect is left
    out. If every given input that the generic advice names ("Raise T3 / q_in or lower r_c") points the helpful
    way, the generic advice is kept (it may also name an input to swap in, such as q_in). Otherwise the advice
    lists only given inputs, each in the direction that helps and, where a nested solve finds it, the value at
    which the step's q reaches 0, e.g. "Raise T4 (to >= 2383.3 K), r_c (to >= 15.54), or lower P3 (...)".
    Returns ``{(a, b): advice}`` for the steps whose advice changes.
    """
    from math import isfinite

    from scipy.optimize import brentq

    from ..results import generic_heat_fix, step_results

    bad = [st for st in steps if st.role == "heat_in"
           and any(p.startswith(f"step {st.a}->{st.b} ({st.kind}, heat_in) has q =") for p in problems)]
    if not bad:
        return {}

    def q_of(inputs, ab):
        try:
            vals = _solve(system, inputs, None, {}, True)[0]
            return next(s.q for s in step_results(defn, vals) if (s.a, s.b) == ab)
        except Exception:                             # noqa: BLE001 - a failed nested solve gives no advice
            return float("nan")

    out = {}
    for st in bad:
        ab = (st.a, st.b)
        q0 = st.q
        raise_, lower_ = [], []
        for name, x in active.items():
            dx = _SENS_REL * (abs(x) if x != 0 else 1.0)
            dq = q_of({**active, name: x + dx}, ab) - q0
            if not isfinite(dq) or abs(dq) <= 1e-12 * max(abs(q0), 1.0):
                continue
            (raise_ if dq > 0 else lower_).append((name, x, dq / dx))
        g_raise, g_lower, g_text = generic_heat_fix(defn, st.b)
        named = {n: "up" for n in g_raise} | {n: "down" for n in g_lower}
        helps = {n: "up" for n, _, _ in raise_} | {n: "down" for n, _, _ in lower_}
        given_named = [n for n in named if n in active]
        if given_named and all(helps.get(n) == named[n] for n in given_named):
            continue                                   # the generic advice is right for these inputs
        if not raise_ and not lower_:
            continue

        def threshold(name, x, slope):
            lo_r, hi_r, _ = system.scan_range[name]
            est = x - q0 / slope
            for f in _THRESHOLD_STRETCH:
                xt = min(max(x + (est - x) * f, lo_r), hi_r)
                qt = q_of({**active, name: xt}, ab)
                if isfinite(qt) and qt >= 0:
                    try:
                        return brentq(lambda v: q_of({**active, name: v}, ab), x, xt, xtol=1e-12 * abs(xt) + 1e-12)
                    except (ValueError, RuntimeError):
                        return None
            return None

        def order(item):                               # the step's own states first, then parameters, then others
            n = item[0]
            own = n[1:] in (str(st.a), str(st.b)) and n[:1] in ("T", "P", "v", "u", "h", "s")
            return (0 if own else 1 if n in {p.name for p in defn.params} else 2)

        def words(items, sign):
            parts = []
            for name, x, slope in sorted(items, key=order):
                t = threshold(name, x, slope)
                parts.append(name if t is None else f"{name} (to {sign} {_fmt_threshold(system, name, t)})")
            return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]

        parts = []
        if raise_:
            parts.append("Raise " + words(raise_, ">="))
        if lower_:
            parts.append(("or lower " if parts else "Lower ") + words(lower_, "<="))
        out[ab] = ", ".join(parts) + "."
    return out


def solve_cycle(defn: CycleDef, fluid, inputs: Mapping[str, object], *, allow_redundant: bool = False,
                check: str = "raise", plausibility: str = "raise"):
    """Solve one cycle from user inputs and return a ``CycleState``.

    Order (spec 9.4-9.8, 10): resolve and validate the inputs (``resolve_inputs``), classify them (``analyse``)
    and raise ``UnderdeterminedError`` / ``OverdeterminedError`` (``raise_for``) before any numeric solve; with
    ``allow_redundant=True`` over-determined inputs are dropped and checked afterwards (spec 9.6,
    ``InconsistentInputError`` on a mismatch); then ``solve_numeric`` (candidate roots filtered by the validity
    checks, spec 9.5 step 3), the results, and the validity checks of spec 9.7: ``check="raise"`` raises
    ``InvalidCycleError`` listing every problem, ``"warn"`` emits ``CycleValidityWarning``, ``"off"`` skips them.
    ``plausibility`` ("raise" default, "warn", "off") governs given state temperatures below 100 K and
    pressures below 1000 Pa (``resolve_inputs``).

    Undefined results raise ``InvalidCycleError`` in every check mode, ``"off"`` included: zero heat input
    (eta_th = w_net/q_in), zero swept volume (mep) or zero expansion work (bwr), within 1e-9 of the cycle's scale
    (``results.undefined_results``). A divide-by-zero in one of these quotients during the solve is turned into
    the same error by re-solving the cycle without them, so the message names the steps and the fix instead of
    the divisor.
    """
    from ..cycle_state import CycleState          # late: cycle_state imports solver.numeric
    from ..results import compute_results, undefined_results, validity_problems

    if check not in CHECK_MODES:
        raise ValueError(f"check must be one of {CHECK_MODES}, got {check!r}")
    system = system_for(defn, fluid)
    given = resolve_inputs(system, inputs, plausibility)
    req = analyse(system, list(given))
    active, dropped = given, {}
    if req.status == "over" and allow_redundant:
        active, dropped, req = _drop_redundant(system, given, req)
    raise_for(req, defn.name)

    def validate(vals):
        return validity_problems(defn, vals, compute_results(defn, fluid, vals)[2])

    try:
        values, report = solve_numeric(system, active, validate=validate)
    except NoSolutionError as e:
        undefined_err = _diagnose_zero_divisor(defn, fluid, system, active, e)
        if undefined_err is not None:
            raise undefined_err from e
        extra = _explain_no_solution(defn, active, e)
        hint = state_mixup_text(defn, active)
        if hint:
            extra += "\n  Hint: " + hint
        if not extra:
            raise
        raise NoSolutionError(str(e) + extra, variables=e.variables, detail=e.detail) from e
    except ConvergenceError as e:
        undefined_err = _diagnose_zero_divisor(defn, fluid, system, active, e, any_failure=True)
        if undefined_err is not None:
            raise undefined_err from e
        raise

    checks = []
    for name, g in dropped.items():
        implied = float(values[name])
        checks.append((name, g, implied))
        if abs(g - implied) > REDUNDANT_RTOL * max(abs(g), 1.0):
            raise InconsistentInputError(
                f"{defn.name}: {name} = {_fmt(name, g)} given, but {', '.join(active)} imply "
                f"{name} = {_fmt(name, implied)}. Remove {name} or correct the input that is wrong.",
                name=name, given=g, implied=implied)

    states, hidden, steps, results = compute_results(defn, fluid, values)
    undefined = undefined_results(defn, values, steps)
    if undefined:                     # eta_th, mep or bwr is 0/0 or x/0: never returned, whatever ``check`` says
        raise _undefined_error(defn, validity_problems(defn, values, steps), undefined)
    if check != "off":
        problems = validity_problems(defn, values, steps)
        hints = _heat_hints(defn, system, active, steps, problems) if problems else {}
        if hints:
            problems = validity_problems(defn, values, steps, hints=hints)
        if problems:
            msg = (f"{defn.name}: the solved cycle is not physically valid:\n  - " + "\n  - ".join(problems))
            if check == "raise":
                raise InvalidCycleError(msg + "\n  (check='warn' or check='off' returns the states anyway.)",
                                        problems=tuple(problems))
            warnings.warn(msg, CycleValidityWarning, stacklevel=3)

    return CycleState(definition=defn, fluid=fluid, inputs=MappingProxyType(dict(given)),
                      values=MappingProxyType(dict(values)), states=MappingProxyType(states),
                      hidden_states=MappingProxyType(hidden), steps=steps, results=results,
                      report=dataclasses.replace(report, redundant=tuple(checks)))
