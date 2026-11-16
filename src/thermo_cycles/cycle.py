"""The user-facing ``Cycle`` class and ``requirements`` (spec section 4)."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping as _MappingABC
from types import MappingProxyType
from typing import Any, Mapping

from .errors import InputError, LimitViolationError, LimitViolationWarning, UnknownInputError
from .fluids import AIR
from .margins import check_limit_value, describe_violations, evaluate_limits, parse_limit_key
from .model.cycle_def import get_cycle
from .model.roles import state_roles
from .solver import (CHECK_MODES, analyse, check_plausibility_mode, resolve_inputs, resolve_name, solve_cycle,
                     system_for)

__all__ = ["Cycle", "requirements", "describe"]

VIOLATION_MODES = ("warn", "raise", "ignore")


def _fluid_or_air(fluid):
    return AIR if fluid is None else fluid


class Cycle:
    """A solved thermodynamic cycle: pick it by name, give any well-posed set of inputs (SI, per unit mass).

    Solves eagerly and is immutable afterwards; errors are raised at construction. ``with_inputs`` returns a
    new cycle.

    ``thermo_cycles.list_cycles()`` gives the names; ``print(thermo_cycles.describe(name))`` shows a cycle's
    steps, the role of every state (e.g. the turbine inlet is T3 in "brayton" but T4 in "brayton_regen"), its
    parameters and typical input sets. The README ("Quick start", "Which cycle?") has worked examples.

    Arguments: ``name``; inputs as keywords (state properties such as ``T1``, ``P1``, parameters such as ``r_c``,
    ``r_p``, ``eta_c``, or results such as ``q_in``), SI per unit mass; ``fluid`` (default AIR); ``limits``
    (e.g. ``{"T_max": 2500}``); ``allow_redundant``; ``check`` ("raise", "warn", "off") for the validity
    checks; ``on_violation`` ("warn", "raise", "ignore") for limits; ``plausibility`` ("raise", "warn", "off")
    for given temperatures below 100 K and pressures below 1000 Pa (usually deg C, kPa or bar).
    """

    __slots__ = ("_cs", "_limits", "_margins", "_allow_redundant", "_check", "_on_violation", "_plausibility")

    def __init__(self, name: str, *, fluid=None, limits: "Mapping[str, float] | None" = None,
                 allow_redundant: bool = False, check: str = "raise", on_violation: str = "warn",
                 plausibility: str = "raise", **inputs: float) -> None:
        check_plausibility_mode(plausibility)
        if check not in CHECK_MODES:
            raise InputError(f"check must be one of {CHECK_MODES}, got {check!r}.")
        if on_violation not in VIOLATION_MODES:
            raise InputError(f"on_violation must be one of {VIOLATION_MODES}, got {on_violation!r}.")
        if limits is not None and not isinstance(limits, _MappingABC):
            raise InputError(f"limits must be a mapping such as {{'T_max': 2500}}, got {type(limits).__name__}.")
        defn = get_cycle(name)
        fl = _fluid_or_air(fluid)
        if limits:      # unknown keys and bad values fail before the solve (spec 11)
            system = system_for(defn, fl)
            for key, val in limits.items():
                parse_limit_key(key, system)
                check_limit_value(key, val)
        cs = solve_cycle(defn, fl, inputs, allow_redundant=bool(allow_redundant), check=check,
                         plausibility=plausibility)
        report = evaluate_limits(cs, limits or {})
        if report.violated and on_violation != "ignore":
            message = describe_violations(report.violated)
            if on_violation == "raise":
                raise LimitViolationError(message, violated=report.violated)
            warnings.warn(message, LimitViolationWarning, stacklevel=2)
        object.__setattr__(self, "_cs", cs)
        object.__setattr__(self, "_limits", MappingProxyType(dict(limits or {})))
        object.__setattr__(self, "_margins", report)
        object.__setattr__(self, "_allow_redundant", bool(allow_redundant))
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_on_violation", on_violation)
        object.__setattr__(self, "_plausibility", plausibility)

    # immutability -------------------------------------------------------------------------------------
    def __setattr__(self, key, value):
        raise dataclasses.FrozenInstanceError(f"Cycle is immutable; use with_inputs(...) for a changed copy "
                                              f"(tried to set {key!r}).")

    def __delattr__(self, key):
        raise dataclasses.FrozenInstanceError(f"Cycle is immutable (tried to delete {key!r}).")

    # data ---------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._cs.definition.name

    @property
    def definition(self):
        return self._cs.definition

    @property
    def fluid(self):
        return self._cs.fluid

    @property
    def inputs(self) -> Mapping[str, float]:
        return self._cs.inputs

    @property
    def states(self):
        return self._cs.states

    @property
    def hidden_states(self):
        return self._cs.hidden_states

    @property
    def steps(self):
        return self._cs.steps

    @property
    def results(self):
        return self._cs.results

    @property
    def params(self) -> Mapping[str, float]:
        """Every catalogue parameter, given or derived (spec 10)."""
        return MappingProxyType({p.name: self._cs.values[p.name] for p in self.definition.params})

    @property
    def limits(self) -> Mapping[str, float]:
        return self._limits

    @property
    def margins(self):
        """``MarginReport`` for the ``limits`` given (empty without limits) with the second-law numbers (spec 11)."""
        return self._margins

    @property
    def cycle_state(self):
        return self._cs

    def value(self, name: str) -> float:
        """Any solved variable by name (case-insensitive and aliases as for inputs), or a cycle result such as
        ``eta_carnot`` or ``T_max``."""
        res = self.results.as_dict()
        if isinstance(name, str):
            key = name if name in res else {k.lower(): k for k in res}.get(name.lower())
            if key is not None and res[key] is not None:
                return res[key]
        system = system_for(self.definition, self.fluid)
        try:
            canonical = resolve_name(system, name)
        except UnknownInputError as e:
            text = str(e).replace("unknown input", "unknown variable").replace("Known inputs:", "Known variables:")
            text += f"\n  Cycle results are also accepted: {', '.join(res)}."
            raise UnknownInputError(text, name=e.name,
                                    suggestions=e.suggestions, accepted=e.accepted) from None
        return self._cs.values[canonical]

    def with_inputs(self, **changes: "float | None") -> "Cycle":
        """A new cycle with ``changes`` merged into the inputs; a value of None removes that input."""
        system = system_for(self.definition, self.fluid)
        merged = dict(self.inputs)
        for key, value in changes.items():
            canonical = resolve_name(system, key)
            if value is None:
                if canonical not in merged:
                    raise InputError(f"{self.name}: cannot remove {canonical}: it is not an input of this cycle "
                                     f"(inputs: {', '.join(merged)}).")
                del merged[canonical]
            else:
                merged[canonical] = value
        return Cycle(self.name, fluid=self.fluid, limits=dict(self._limits) or None,
                     allow_redundant=self._allow_redundant, check=self._check, on_violation=self._on_violation,
                     plausibility=self._plausibility, **merged)

    def plot(self, kind: str = "Ts", **kw) -> Any:
        """T-s ("Ts"), P-v ("Pv") or both plot of this cycle; see ``thermo_cycles.plotting.plot_cycle``."""
        from .plotting import plot_cycle
        return plot_cycle(self, kind, **kw)

    # text and export ------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.inputs.items())
        return f"Cycle({self.name!r}, {args})"

    def summary(self) -> str:
        """Text table of inputs, states, steps, results and parameters."""
        d = self.definition
        lines = [f"{d.name}: {d.title}", f"fluid: {self.fluid!r}",
                 "inputs: " + ", ".join(f"{k} = {_num(v)}" for k, v in self.inputs.items()), "",
                 "states   T [K]      P [kPa]     v [m3/kg]   u [kJ/kg]   h [kJ/kg]   s [J/(kg K)]"]
        for st in list(self.states.values()) + list(self.hidden_states.values()):
            lines.append(f"  {str(st.id):<5} {st.T:9.2f} {st.P / 1e3:12.2f} {st.v:12.5g} {st.u / 1e3:11.2f} "
                         f"{st.h / 1e3:11.2f} {st.s:11.2f}")
        lines += ["", "steps    kind         role          q [kJ/kg]   w [kJ/kg]   ds [J/(kg K)]"]
        for s in self.steps:
            lines.append(f"  {s.a}->{s.b:<3} {s.kind:<12} {s.role:<12} {s.q / 1e3:11.2f} {s.w / 1e3:11.2f} "
                         f"{s.ds:11.3f}")
        r = self.results
        lines += ["", "results",
                  f"  q_in = {r.q_in / 1e3:.2f} kJ/kg, q_out = {r.q_out / 1e3:.2f} kJ/kg, "
                  f"w_net = {r.w_net / 1e3:.2f} kJ/kg",
                  f"  eta_th = {r.eta_th:.5f}, eta_carnot = {r.eta_carnot:.5f}, eta_ii = {r.eta_ii:.5f}"]
        if r.mep is not None:
            lines.append(f"  mep = {r.mep / 1e3:.2f} kPa")
        if r.bwr is not None:
            lines.append(f"  bwr = {r.bwr:.5f}")
        lines.append(f"  T {r.T_min:.2f}..{r.T_max:.2f} K, P {r.P_min / 1e3:.2f}..{r.P_max / 1e3:.2f} kPa, "
                     f"v {r.v_min:.5g}..{r.v_max:.5g} m3/kg")
        if self.params:
            lines += ["", "parameters: " + ", ".join(f"{k} = {_num(v)}" for k, v in self.params.items())]
        lines += ["", f"margins: {self.margins.table()}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Plain dict (JSON-friendly values) of the cycle: inputs, parameters, states, steps, results."""
        return {
            "name": self.name,
            "fluid": repr(self.fluid),
            "inputs": dict(self.inputs),
            "params": dict(self.params),
            "states": {k: s.as_dict() for k, s in self.states.items()},
            "hidden_states": {k: s.as_dict() for k, s in self.hidden_states.items()},
            "steps": [dataclasses.asdict(s) for s in self.steps],
            "results": self.results.as_dict(),
        }


def _num(x: float) -> str:
    return f"{x:.6g}" if math.isfinite(x) else str(x)


def requirements(name: str, *, fluid=None, plausibility: str = "raise", **inputs: float):
    """What a cycle still needs, or which inputs conflict, for ``inputs`` (spec 9.4); returns ``Requirements``.

    Input names and values are resolved and validated as for ``Cycle`` (unknown names and bad values raise;
    ``plausibility`` as for ``Cycle``).
    """
    defn = get_cycle(name)
    system = system_for(defn, _fluid_or_air(fluid))
    given = resolve_inputs(system, inputs, check_plausibility_mode(plausibility))
    return analyse(system, list(given))


def describe(name: str) -> str:
    """Text description of a catalogue cycle (spec section 4): title, numbered steps, the role of every state
    in words (e.g. "4: turbine inlet (after 3->4 heat_in)"), parameters with
    descriptions and defaults, degrees of freedom, typical input sets and references.

    ``name`` is looked up as for ``Cycle`` (case-insensitive, aliases). Raises ``UnknownCycleError`` otherwise.
    """
    defn = get_cycle(name)
    needs = analyse(system_for(defn, AIR), []).missing
    out = [defn.title, "=" * len(defn.title)]
    if defn.description:
        out.append(defn.description)
    out += ["", f"Name: {defn.name}   System: {defn.system}   States: {', '.join(str(s) for s in defn.states)}",
            "", "Steps:"]
    out += [f"  {st.a}->{st.b} {st.kind} ({st.role})"
            + (f", efficiency {st.efficiency}" if st.efficiency else "") for st in defn.steps]
    out += ["", "State roles:"] + [f"  {sid}: {words}" for sid, words in state_roles(defn).items()]
    out += ["", "Parameters:"]
    if defn.params:
        for p in defn.params:
            extra = [f"default {p.default:g}"] if p.default is not None else []
            out.append(f"  {p.name}: {p.description}" + (f" [{', '.join(extra)}]" if extra else ""))
    else:
        out.append("  (none)")
    defaulted = [p for p in defn.params if p.default is not None]
    dof = f"{defn.name} needs {needs} inputs"
    if defaulted:
        dof += (f" (plus {', '.join(p.name for p in defaulted)}, "
                f"which default to {', '.join(f'{p.default:g}' for p in defaulted)})")
    out += ["", f"Degrees of freedom: {dof}."]
    if defn.typical_inputs:
        out += ["", "Typical input sets:"] + [f"  {', '.join(t)}" for t in defn.typical_inputs]
    if defn.references:
        out += ["", "References:"] + [f"  {r}" for r in defn.references]
    return "\n".join(out)
