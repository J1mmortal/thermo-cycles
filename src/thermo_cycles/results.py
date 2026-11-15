"""Step and cycle results from a solved value dict (spec sections 5.3 and 10), and the physical validity checks
run after the solve (spec section 9.7)."""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Mapping

from .model.cycle_def import CycleDef, state_ids, step_vars
from .model.steps import KINDS
from .state import PROPS, State

__all__ = ["StepResult", "CycleResults", "compute_results", "validity_problems", "HEAT_SIGN_RTOL",
           "step_results", "undefined_results", "ZERO_RTOL"]

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


def step_results(defn: CycleDef, values: Mapping[str, float]) -> "tuple[StepResult, ...]":
    """The ``StepResult`` of every step from solved ``values`` (needs only the state and step variables, not
    ``eta_th`` / ``mep`` / ``bwr``)."""
    return _steps(defn, values)


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
    - heat_in: q >= -1e-9 q_in (zero allowed) and q_in > 0 overall; heat_out: q <= 1e-9 q_in (with zero heat
      input the scale is the largest step |q| or |w| instead of q_in);
    - results defined: no zero heat input, no zero swept volume (closed), no zero expansion work (flow), see
      ``undefined_results``; ``solve_cycle`` raises on these in every check mode;
    - parameters inside their bounds;
    - regenerator (role ``internal`` steps): heat flows hot to cold, see ``_internal_problems``.

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
    undefined = undefined_results(defn, values, steps)
    # Spec 9.7 scales the heat-sign tolerance with q_in. With zero heat input that scale is itself ~0 and every
    # rounding-level q would be reported as a sign problem, so the cycle's step scale is used instead.
    tol = HEAT_SIGN_RTOL * (_step_scale(steps) if _zero_heat(values, steps) else abs(q_in))
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
    if not q_in > 0 and not _zero_heat(values, steps):
        problems.append(f"q_in = {q_in / 1e3:.1f} kJ/kg is not > 0: the heat_in steps take in no net heat, so "
                        f"this is not a power cycle.")
    problems += undefined
    problems += _internal_problems(defn, values, steps, q_in)
    for p in defn.params:
        if p.bounds is None:
            continue
        x = values[p.name]
        lo, hi = p.bounds
        if not lo <= x <= hi:
            problems.append(f"parameter {p.name} = {x:.6g} is outside its bounds [{lo:g}, {hi:g}] "
                            f"({p.description}).")
    return problems


# Spec 9.7 regenerator rule: comparisons of temperatures allow this relative round-off (eps = 1 makes T3 = T5
# and T6 = T2 up to rounding).
REGEN_T_RTOL = 1e-9
# Internal steps only move heat between themselves: their q must sum to zero within this fraction of the
# larger of q_in and the largest internal |q|.
INTERNAL_Q_RTOL = 1e-6


def _internal_problems(defn: CycleDef, values: Mapping[str, float], steps: "tuple[StepResult, ...]",
                       q_in: float) -> "list[str]":
    """Regenerator checks (spec 9.7) for the steps with role ``internal``.

    Internal steps are paired in definition order, (1st, 2nd), (3rd, 4th), ...; in each pair the first step is
    the cold side (the stream the regenerator heats, a_c -> b_c) and the second the hot side (a_h -> b_h).
    Counterflow, heat flowing hot to cold needs T(a_h) >= T(b_c) and T(b_h) >= T(a_c); for brayton_regen that
    is T5 >= T3 and T6 >= T2. Together with the energy balance this gives 0 <= eps <= 1 and T5 >= T2, except
    at eps = 1, where both hold with equality even when T5 < T2 (cold side cooled, hot side heated). So
    T(a_h) >= T(a_c) (T5 >= T2) is also checked explicitly (deviation from the letter of spec 9.7, which
    states T5 > T2 only as a consequence). The order matters: when T5 < T2 the solved "cold" side is in fact cooled, and pairing by the sign of q would
    hide that. The internal heats must also cancel (w_net = q_in - q_out).
    """
    internal = [st for st in steps if st.role == "internal"]
    problems: "list[str]" = []
    if not internal:
        return problems
    lower = [p.name for p in defn.params if p.name in _COMPRESSION_PARAMS]
    peak = [f"T{st.b}" for st in steps if st.role == "heat_in"]
    tuning = [p.name for p in defn.params if p.name in ("eps", "dT_regen")]
    for cold, hot in zip(internal[0::2], internal[1::2]):
        ci, co, hi, ho = cold.a, cold.b, hot.a, hot.b
        T = {k: float(values[f"T{k}"]) for k in (ci, co, hi, ho)}
        tol = REGEN_T_RTOL * max(abs(x) for x in T.values())
        label = f"regenerator (cold side {ci}→{co}, hot side {hi}→{ho})"
        bad = []
        if not T[hi] >= T[co] - tol:
            bad.append(f"hot inlet T{hi} = {_K(T[hi])} is below cold outlet T{co} = {_K(T[co])}")
        if not T[ho] >= T[ci] - tol:
            bad.append(f"hot outlet T{ho} = {_K(T[ho])} is below cold inlet T{ci} = {_K(T[ci])}")
        if not T[hi] >= T[ci] - tol and not bad:
            # eps = 1 gives T(b_c) = T(a_h) and T(b_h) = T(a_c), so the two checks above pass with equality
            # even when the streams are swapped; the inlet comparison catches that case.
            bad.append(f"hot inlet T{hi} = {_K(T[hi])} is below cold inlet T{ci} = {_K(T[ci])} (the "
                       f"\"cold\" side {ci}→{co} is cooled, q = {cold.q / 1e3:.1f} kJ/kg)")
        if not bad:
            continue
        msg = f"{label}: {' and '.join(bad)}, so heat would flow from the cold stream to the hot one. "
        if T[hi] < T[ci] - tol:
            fix = [f"lower {' / '.join(lower)}"] if lower else ["reduce the compression"]
            fix += [f"raise {' / '.join(peak)}"] if peak else []
            alt = defn.name[:-len("_regen")] if defn.name.endswith("_regen") else ""
            msg += (f"The hot stream enters colder than the cold stream (T{hi} < T{ci}), so no regenerator "
                    f"effectiveness in [0, 1] works: {' or '.join(fix)}"
                    + (f", or use the cycle {alt!r} without regeneration." if alt else
                       ", or remove the regenerator."))
        else:
            fix = []
            if "eps" in tuning:
                fix.append("lower eps (<= 1)")
            if "dT_regen" in tuning:
                fix.append("raise dT_regen (>= 0)")
            advice = " or ".join(fix) if fix else "reduce the regenerator heat transfer"
            msg += advice[:1].upper() + advice[1:] + "."
        problems.append(msg)
    q_sum = math.fsum(st.q for st in internal)
    scale = max([abs(q_in)] + [abs(st.q) for st in internal])
    if not abs(q_sum) <= INTERNAL_Q_RTOL * scale:
        names = ", ".join(f"{st.a}→{st.b}" for st in internal)
        problems.append(f"internal steps {names} exchange a net q = {q_sum / 1e3:.4g} kJ/kg with the "
                        f"surroundings: internal (regenerator) steps may only move heat between themselves.")
    return problems


# ---------------------------------------------------------------------------------------------------------
# undefined results (zero heat input and friends)

# A divisor of eta_th, mep or bwr counts as zero when it is within this fraction of the cycle's own scale
# (the same 1e-9 as the heat-sign rule of spec 9.7). Scale: the largest |q| or |w| of any step, or the larger
# specific volume for v_max - v_min. q_in itself cannot be the scale of its own zero test.
ZERO_RTOL = HEAT_SIGN_RTOL

_RATIO_GT = re.compile(r"\(>\s*([A-Za-z_]\w*)\s*\)")


def _step_scale(steps: "tuple[StepResult, ...]") -> float:
    return max([abs(s.q) for s in steps] + [abs(s.w) for s in steps] + [0.0])


def _zero_heat(values: Mapping[str, float], steps: "tuple[StepResult, ...]") -> bool:
    return abs(float(values["q_in"])) <= ZERO_RTOL * _step_scale(steps)


def _same(x: float, y: float) -> bool:
    return abs(x - y) <= ZERO_RTOL * max(abs(x), abs(y))


def _degenerate_params(defn: CycleDef, values: Mapping[str, float]) -> "list[str]":
    """Fix hints from the parameters: a ratio at its lower bound 1 (``r_c = 1``), or a parameter equal to the one
    its description says it must exceed (``r_e``: "(> r_c)")."""
    hints = []
    names = {p.name for p in defn.params}
    for p in defn.params:
        if p.name not in values:
            continue
        x = float(values[p.name])
        if p.bounds is not None and p.bounds[0] == 1.0 and _same(x, 1.0):
            hints.append(f"{p.name} = 1 ({p.description}) is the degenerate limit: raise {p.name} above 1")
        m = _RATIO_GT.search(p.description)
        if m and m.group(1) in names and m.group(1) in values and _same(x, float(values[m.group(1)])):
            o = m.group(1)
            hints.append(f"{p.name} = {o} = {x:.6g}: {p.name} must exceed {o}")
    return hints


def undefined_results(defn: CycleDef, values: Mapping[str, float], steps: "tuple[StepResult, ...]"
                      ) -> "list[str]":
    """Problems that leave a cycle result undefined (a zero divisor), as messages; empty when all are defined.

    - zero heat input: |q_in| <= 1e-9 x the largest step |q| or |w|, so eta_th = w_net/q_in is undefined;
    - closed cycles: v_max_state and v_min_state have the same volume (within 1e-9), so mep is undefined;
    - flow cycles: |w_expansion| <= 1e-9 x the largest step |q| or |w|, so bwr = -w_compression/w_expansion is
      undefined.

    Each message names the step(s), the states whose properties coincide and the fix. ``values`` need not contain
    eta_th, mep or bwr (``solve_cycle`` calls this on a re-solve without them after a divide-by-zero).
    """
    problems: "list[str]" = []
    tol = ZERO_RTOL * _step_scale(steps)
    hints = _degenerate_params(defn, values)
    lower = [p.name for p in defn.params if p.name in _COMPRESSION_PARAMS]
    if _zero_heat(values, steps):
        q_in = float(values["q_in"])
        parts = []
        for st in steps:
            if st.role != "heat_in":
                continue
            Ta, Tb = float(values[f"T{st.a}"]), float(values[f"T{st.b}"])
            why = f"T{st.b} = T{st.a} = {_K(Ta)}" if _same(Ta, Tb) else f"T{st.a} = {_K(Ta)}, T{st.b} = {_K(Tb)}"
            q = 0.0 if abs(st.q) <= tol else st.q
            parts.append(f"step {st.a}→{st.b} ({st.kind}, heat_in) has q = {q / 1e3:.1f} kJ/kg ({why})")
        q_show = 0.0 if abs(q_in) <= tol else q_in
        peak = " / ".join([f"T{st.b}" for st in steps if st.role == "heat_in"] + ["q_in"])
        fix = hints + [f"raise {peak} (heat must enter the cycle)"]
        problems.append(f"zero heat input: q_in = {q_show / 1e3:.1f} kJ/kg, so eta_th = w_net/q_in is undefined "
                        f"and this is not a power cycle. {'; '.join(parts)}. Fix: {'; or '.join(fix)}.")
    if defn.system == "closed":
        a, b = defn.v_max_state, defn.v_min_state
        va, vb = float(values[f"v{a}"]), float(values[f"v{b}"])
        if _same(va, vb):
            fix = hints or ([f"raise {' / '.join(lower)} above 1"] if lower else
                            ["give the states different volumes"])
            problems.append(f"zero swept volume: v{a} = v{b} = {va:.6g} m³/kg, so mep = w_net/(v{a} - v{b}) is "
                            f"undefined (no compression or expansion). Fix: {'; or '.join(fix)}.")
    else:
        w_exp = math.fsum(s.w for s in steps if s.role == "expansion")
        if abs(w_exp) <= tol:
            exps = ", ".join(f"{s.a}→{s.b}" for s in steps if s.role == "expansion")
            pr = ", ".join(f"P{s.a} = P{s.b} = {_kpa(float(values[f'P{s.a}']))}" for s in steps
                           if s.role == "expansion" and _same(float(values[f"P{s.a}"]), float(values[f"P{s.b}"])))
            fix = hints or ([f"raise {' / '.join(lower)} above 1"] if lower else
                            ["give the expansion a pressure drop"])
            problems.append(f"zero expansion work: w_expansion = 0.0 kJ/kg (step {exps}"
                            + (f"; {pr}" if pr else "") + "), so bwr = -w_compression/w_expansion is undefined. "
                            f"Fix: {'; or '.join(fix)}.")
    return problems
