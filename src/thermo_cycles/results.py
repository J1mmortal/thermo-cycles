"""Step and cycle results from a solved value dict (spec sections 5.3 and 10), and the physical validity checks
run after the solve (spec section 9.7)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Mapping

from .model.cycle_def import CycleDef, state_ids, step_vars
from .model.steps import KINDS
from .state import PROPS, State

__all__ = ["StepResult", "CycleResults", "compute_results", "validity_problems", "HEAT_SIGN_RTOL"]

# Spec 9.7: heat_in steps need q >= -1e-9 q_in (so q = 0 is a valid limiting case, e.g. Dual with alpha = 1),
# heat_out steps q <= 1e-9 q_in.
HEAT_SIGN_RTOL = 1e-9

# Parameters that set the compression; named in the advice of a heat-step sign problem (spec 9.7 example
# "Raise T3 / q_in or lower r_c").
_COMPRESSION_PARAMS = ("r_c", "r_p")


@dataclass(frozen=True)
class StepResult:
    """One process step a -> b. Signs per spec 3: q > 0 into the fluid, w > 0 done by the fluid."""

    a: int
    b: int
    kind: str
    role: str                  # compression | expansion | heat_in | heat_out | internal
    q: float                   # J/kg
    w: float                   # J/kg
    ds: float                  # J/(kg K)
    hidden: "str | None" = None   # e.g. "2s" for a non-ideal compression


@dataclass(frozen=True)
class CycleResults:
    """Cycle results (spec 5.3, 10). q_in and q_out are positive magnitudes; w_compression and w_expansion
    are signed sums over the steps with those roles."""

    q_in: float
    q_out: float
    w_net: float
    eta_th: float
    mep: "float | None"        # closed cycles only
    bwr: "float | None"        # flow cycles only
    w_compression: float
    w_expansion: float
    T_min: float
    T_max: float
    P_min: float
    P_max: float
    v_min: float
    v_max: float
    eta_carnot: float          # 1 - T_min/T_max over all main and hidden states
    eta_ii: float              # eta_th / eta_carnot

    def as_dict(self) -> "dict[str, float | None]":
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# ---------------------------------------------------------------------------------------------------------


def _state(sid: "int | str", values: Mapping[str, float]) -> State:
    return State(sid, *(float(values[f"{p}{sid}"]) for p in PROPS))


def _hidden_ids(defn: CycleDef) -> "tuple[str, ...]":
    return state_ids(defn)[len(defn.states):]


def _steps(defn: CycleDef, values: Mapping[str, float]) -> "tuple[StepResult, ...]":
    out = []
    for st in defn.steps:
        qn, wn = step_vars(st)
        hidden = KINDS[st.kind].hidden_states(str(st.b))
        out.append(StepResult(st.a, st.b, st.kind, st.role, float(values[qn]), float(values[wn]),
                              float(values[f"s{st.b}"]) - float(values[f"s{st.a}"]),
                              hidden[0] if hidden else None))
    return tuple(out)


def compute_results(defn: CycleDef, fluid, values: Mapping[str, float]
                    ) -> "tuple[dict[int, State], dict[str, State], tuple[StepResult, ...], CycleResults]":
    """States, hidden states, step results and cycle results from the solved ``values`` (spec 10).

    ``q_in``, ``q_out``, ``w_net``, ``eta_th``, ``mep`` and ``bwr`` are the solved values of the equation system
    (spec 9.1); the extremes, ``eta_carnot`` and ``eta_ii`` are computed over the main and hidden states. ``fluid``
    is the fluid the values were solved with (kept for fluids that report a vapour quality; unused for an
    ideal gas).
    """
    states = {k: _state(k, values) for k in defn.states}
    hidden = {h: _state(h, values) for h in _hidden_ids(defn)}
    steps = _steps(defn, values)
    every = list(states.values()) + list(hidden.values())
    Ts = [s.T for s in every]
    Ps = [s.P for s in every]
    vs = [s.v for s in every]
    T_min, T_max = min(Ts), max(Ts)
    eta_th = float(values["eta_th"])
    eta_carnot = 1.0 - T_min / T_max if T_max != 0 else math.nan
    eta_ii = eta_th / eta_carnot if eta_carnot != 0 else math.nan
    w_comp = math.fsum(s.w for s in steps if s.role == "compression")
    w_exp = math.fsum(s.w for s in steps if s.role == "expansion")
    closed = defn.system == "closed"
    results = CycleResults(
        q_in=float(values["q_in"]), q_out=float(values["q_out"]), w_net=float(values["w_net"]), eta_th=eta_th,
        mep=float(values["mep"]) if closed else None,
        bwr=None if closed else float(values["bwr"]),
        w_compression=float(values.get("w_compression", w_comp)),
        w_expansion=float(values.get("w_expansion", w_exp)),
        T_min=T_min, T_max=T_max, P_min=min(Ps), P_max=max(Ps), v_min=min(vs), v_max=max(vs),
        eta_carnot=eta_carnot, eta_ii=eta_ii)
    return states, hidden, steps, results


# ---------------------------------------------------------------------------------------------------------
# validity (spec 9.7)


def _kpa(x: float) -> str:
    return f"{x / 1e3:.1f} kPa"


def _K(x: float) -> str:
    return f"{x:.5g} K"


def validity_problems(defn: CycleDef, values: Mapping[str, float], steps: "tuple[StepResult, ...]") -> "list[str]":
    """Every physical-validity problem of a solved cycle (spec 9.7), as messages; empty when the cycle is valid.

    - all T, P, v > 0 (main and hidden states);
    - compression: P_b > P_a and w < 0; expansion: P_b < P_a and w > 0;
    - heat_in: q >= -1e-9 q_in (zero allowed) and q_in > 0 overall; heat_out: q <= 1e-9 q_in;
    - parameters inside their bounds.

    Messages name the step and say what to change, e.g. "step 2→3 (isochoric, heat_in) has q = -120.4 kJ/kg < 0:
    T3 = 600 K is below T2 = 689.2 K. Raise T3 / q_in or lower r_c."
    """
    problems: "list[str]" = []
    for sid in state_ids(defn):
        for p, unit in (("T", "K"), ("P", "Pa"), ("v", "m³/kg")):
            x = values[f"{p}{sid}"]
            if not x > 0:
                problems.append(f"state {sid}: {p}{sid} = {x:.6g} {unit} is not > 0.")

    q_in = float(values["q_in"])
    tol = HEAT_SIGN_RTOL * abs(q_in)
    lower = [p.name for p in defn.params if p.name in _COMPRESSION_PARAMS]
    for st in steps:
        a, b = st.a, st.b
        label = f"step {a}→{b} ({st.kind}, {st.role})"
        Pa, Pb = values[f"P{a}"], values[f"P{b}"]
        Ta, Tb = values[f"T{a}"], values[f"T{b}"]
        if st.role == "compression":
            if not Pb > Pa:
                problems.append(f"{label} has P{b} = {_kpa(Pb)} <= P{a} = {_kpa(Pa)}: a compression must raise "
                                f"the pressure.")
            if not st.w < 0:
                problems.append(f"{label} has w = {st.w / 1e3:.1f} kJ/kg >= 0: a compression must take work "
                                f"(w < 0).")
        elif st.role == "expansion":
            if not Pb < Pa:
                problems.append(f"{label} has P{b} = {_kpa(Pb)} >= P{a} = {_kpa(Pa)}: an expansion must lower "
                                f"the pressure.")
            if not st.w > 0:
                problems.append(f"{label} has w = {st.w / 1e3:.1f} kJ/kg <= 0: an expansion must deliver work "
                                f"(w > 0).")
        elif st.role == "heat_in":
            if not st.q >= -tol:
                why = f"T{b} = {_K(Tb)} is below T{a} = {_K(Ta)}. " if Tb < Ta else ""
                fix = " / ".join((f"T{b}", "q_in"))
                fix += f" or lower {' / '.join(lower)}" if lower else " or reduce the compression"
                problems.append(f"{label} has q = {st.q / 1e3:.1f} kJ/kg < 0: {why}Raise {fix}.")
        elif st.role == "heat_out":
            if not st.q <= tol:
                why = f"T{b} = {_K(Tb)} is above T{a} = {_K(Ta)}, so " if Tb > Ta else ""
                problems.append(f"{label} has q = +{st.q / 1e3:.1f} kJ/kg > 0: {why}the step adds heat "
                                f"instead of rejecting it.")
    if not q_in > 0:
        problems.append(f"q_in = {q_in / 1e3:.1f} kJ/kg is not > 0: the heat_in steps take in no net heat, so "
                        f"this is not a power cycle.")
    for p in defn.params:
        if p.bounds is None:
            continue
        x = values[p.name]
        lo, hi = p.bounds
        if not lo <= x <= hi:
            problems.append(f"parameter {p.name} = {x:.6g} is outside its bounds [{lo:g}, {hi:g}] "
                            f"({p.description}).")
    return problems
