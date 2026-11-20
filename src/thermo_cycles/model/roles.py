"""The role of each state of a cycle in words, derived from the cycle's steps (v0.1.1).

Used by ``describe()`` ("4: turbine inlet (after 3->4 heat_in)") and by the no-solution message of
``solve_cycle``, so a user can see which state number is which, e.g. that the turbine inlet is T3 in ``brayton``
but T4 in ``brayton_regen``. Nothing here is cycle-specific: the words come from the step roles (compression,
expansion, heat_in, heat_out), the regenerator pairs and the cycle's system (closed piston or steady flow).
"""

from __future__ import annotations

from .cycle_def import CycleDef

__all__ = ["state_roles", "state_role"]

_WORK = {  # (role, side, system) -> words; side "in" = the state enters the step, "out" = the step ends there
    ("compression", "in", "flow"): "compressor inlet", ("compression", "out", "flow"): "compressor outlet",
    ("expansion", "in", "flow"): "turbine inlet", ("expansion", "out", "flow"): "turbine outlet",
    ("compression", "in", "closed"): "start of compression", ("compression", "out", "closed"): "end of compression",
    ("expansion", "in", "closed"): "start of expansion", ("expansion", "out", "closed"): "end of expansion",
}
_HEAT = {"heat_in": "heat addition", "heat_out": "heat rejection", "internal": "internal heat exchange"}


def _regen_sides(defn: CycleDef) -> "dict[tuple[int, int], str]":
    sides = {}
    for cold, hot in getattr(defn, "regenerators", ()) or ():
        sides[tuple(cold)] = "cold"
        sides[tuple(hot)] = "hot"
    return sides


def _step_label(st, sides) -> str:
    side = sides.get((st.a, st.b))
    return f"{st.a}->{st.b} " + (f"regenerator {side} side" if side else st.role)


def _words(st, side: str, system: str, sides) -> "tuple[int, str]":
    """(priority, words) for a state that is the start (side "in") or end ("out") of step ``st``."""
    words = _WORK.get((st.role, side, system))
    if words is not None:
        return (0 if side == "in" else 1), words
    regen = sides.get((st.a, st.b))
    if regen is not None:
        return 2, f"regenerator {regen}-side {'inlet' if side == 'in' else 'outlet'}"
    what = _HEAT.get(st.role, st.role)
    return 3, f"{'start' if side == 'in' else 'end'} of {st.kind} {what}"


def state_roles(defn: CycleDef) -> "dict[int, str]":
    """``{state: words}`` for every main state, e.g. ``{3: "turbine inlet (after 2->3 heat_in)", ...}``.

    The main words come from the most telling step at the state (start of a work step, end of a work step,
    regenerator side, then heat step); the other step at the state is named in brackets.
    """
    sides = _regen_sides(defn)
    out: "dict[int, str]" = {}
    for sid in defn.states:
        cands = []
        for st in defn.steps:
            if st.a == sid:
                cands.append((*_words(st, "in", defn.system, sides), f"before {_step_label(st, sides)}"))
            if st.b == sid:
                cands.append((*_words(st, "out", defn.system, sides), f"after {_step_label(st, sides)}"))
        if not cands:
            out[sid] = "state"
            continue
        cands.sort(key=lambda c: c[0])
        main = cands[0][1]
        others = ", ".join(c[2] for c in cands[1:])
        out[sid] = f"{main} ({others})" if others else main
    return out


def state_role(defn: CycleDef, sid: "int | str") -> "str | None":
    """Words for one state (``3`` or ``"3"``); None for a hidden state such as ``"2s"`` or an unknown id."""
    roles = state_roles(defn)
    for k, v in roles.items():
        if str(k) == str(sid):
            return v
    return None
