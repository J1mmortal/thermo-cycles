"""The general solver: structure, numeric block solve, input advice, and ``solve_cycle`` that ties them together
(spec section 9)."""

from __future__ import annotations

import dataclasses
import warnings
from collections import OrderedDict
from types import MappingProxyType
from typing import Mapping

from ..errors import InconsistentInputError, InvalidCycleError, CycleValidityWarning
from ..model.cycle_def import CycleDef
from .advice import Requirements, analyse, raise_for
from .numeric import SolveReport, _fmt, solve_numeric
from .system import EquationSystem, build_system, resolve_inputs, resolve_name

__all__ = ["solve_cycle", "system_for", "REDUNDANT_RTOL", "CHECK_MODES",
           "Requirements", "analyse", "raise_for", "SolveReport", "solve_numeric",
           "EquationSystem", "build_system", "resolve_inputs", "resolve_name"]

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


def solve_cycle(defn: CycleDef, fluid, inputs: Mapping[str, object], *, allow_redundant: bool = False,
                check: str = "raise"):
    """Solve one cycle from user inputs and return a ``CycleState``.

    Order (spec 9.4-9.8, 10): resolve and validate the inputs (``resolve_inputs``), classify them (``analyse``)
    and raise ``UnderdeterminedError`` / ``OverdeterminedError`` (``raise_for``) before any numeric solve; with
    ``allow_redundant=True`` over-determined inputs are dropped and checked afterwards (spec 9.6,
    ``InconsistentInputError`` on a mismatch); then ``solve_numeric`` (candidate roots filtered by the validity
    checks, spec 9.5 step 3), the results, and the validity checks of spec 9.7: ``check="raise"`` raises
    ``InvalidCycleError`` listing every problem, ``"warn"`` emits ``CycleValidityWarning``, ``"off"`` skips them.
    """
    from ..cycle_state import CycleState          # late: cycle_state imports solver.numeric
    from ..results import compute_results, validity_problems

    if check not in CHECK_MODES:
        raise ValueError(f"check must be one of {CHECK_MODES}, got {check!r}")
    system = system_for(defn, fluid)
    given = resolve_inputs(system, inputs)
    req = analyse(system, list(given))
    active, dropped = given, {}
    if req.status == "over" and allow_redundant:
        active, dropped, req = _drop_redundant(system, given, req)
    raise_for(req, defn.name)

    def validate(vals):
        return validity_problems(defn, vals, compute_results(defn, fluid, vals)[2])

    values, report = solve_numeric(system, active, validate=validate)

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
    if check != "off":
        problems = validity_problems(defn, values, steps)
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
