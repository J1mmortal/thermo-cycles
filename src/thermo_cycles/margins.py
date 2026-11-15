"""Margins to user limits and the second-law efficiency report (spec section 11)."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from typing import Iterator, Literal, Mapping

from .errors import InputError, UnknownInputError
from .solver import resolve_name

__all__ = ["Margin", "SecondLaw", "MarginReport", "parse_limit_key", "evaluate_limits"]

_PROPS = ("T", "P", "v", "u", "h", "s")
_RESULT_ONLY = ("eta_ii", "eta_carnot")          # computed after the solve, not equation-system variables
_UNITS = {"T": "K", "P": "Pa", "v": "m3/kg", "u": "J/kg", "h": "J/kg", "s": "J/(kg K)"}


@dataclass(frozen=True)
class Margin:
    key: str
    kind: Literal["max", "min"]
    limit: float
    value: float
    where: "int | str | None"      # state id holding the extreme, or None
    margin: float                  # max: limit - value ; min: value - limit (same unit as the limit)
    margin_pct: float              # 100 * margin / |limit| (nan for a zero limit)
    ok: bool                       # margin >= 0


@dataclass(frozen=True)
class SecondLaw:
    eta_th: float
    eta_carnot: float
    eta_ii: float
    gap: float                     # eta_carnot - eta_th


class MarginReport(_MappingABC):
    """Margins by limit key (in the order given), plus the always-present second-law numbers."""

    def __init__(self, margins: "Mapping[str, Margin]", second_law: SecondLaw) -> None:
        self._margins = dict(margins)
        self.second_law = second_law

    def __getitem__(self, key: str) -> Margin:
        return self._margins[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._margins)

    def __len__(self) -> int:
        return len(self._margins)

    @property
    def violated(self) -> "tuple[Margin, ...]":
        return tuple(m for m in self._margins.values() if not m.ok)

    @property
    def ok(self) -> bool:
        return not self.violated

    @property
    def worst(self) -> "Margin | None":
        """The margin with the lowest ``margin_pct`` (None without limits); nan percentages (zero limit) rank last."""
        if not self._margins:
            return None
        return min(self._margins.values(),
                   key=lambda m: (math.inf if math.isnan(m.margin_pct) else m.margin_pct))

    def __repr__(self) -> str:
        return f"MarginReport({dict(self._margins)!r}, second_law={self.second_law!r})"

    def table(self) -> str:
        """Fixed-width text table: one row per limit, then the second-law efficiencies."""
        lines = []
        if self._margins:
            kw = max(max(len(k) for k in self._margins), 3)
            lines.append(f"{'limit':<{kw}}  {'kind':<4}  {'limit value':>12}  {'value':>12}  {'margin':>12}  "
                         f"{'margin %':>9}  {'at':<5}  status")
            for k, m in self._margins.items():
                lines.append(f"{k:<{kw}}  {m.kind:<4}  {m.limit:12.5g}  {m.value:12.5g}  {m.margin:12.5g}  "
                             f"{m.margin_pct:9.2f}  {'' if m.where is None else str(m.where):<5}  "
                             f"{'ok' if m.ok else 'VIOLATED'}")
        else:
            lines.append("(no limits)")
        s = self.second_law
        lines.append(f"second law: eta_th = {s.eta_th:.5f}, eta_carnot = {s.eta_carnot:.5f}, "
                     f"eta_ii = {s.eta_ii:.5f}, gap = {s.gap:.5f}")
        return "\n".join(lines)


def parse_limit_key(key: str, system) -> "tuple[str, str]":
    """Split a limit key into (name, "max" | "min").

    ``name`` is a property letter (T, P, v, u, h, s: extreme over all states), ``eta_ii`` / ``eta_carnot``, or the
    canonical name of any solved variable of ``system`` (case-insensitive and aliases as for inputs). Anything
    else raises ``UnknownInputError``.
    """
    accepted = tuple(system.variables) + _PROPS + _RESULT_ONLY
    if not isinstance(key, str):
        raise UnknownInputError(f"limit keys must be strings such as 'T_max' or 'eta_th_min', got {key!r}.",
                                name=key, suggestions=(), accepted=accepted)
    kind = key[-4:].lower()
    if kind not in ("_max", "_min"):
        raise UnknownInputError(f"unknown limit {key!r}: a limit key is '<name>_max' or '<name>_min', for example "
                                f"'T_max', 'P2_max', 'eta_th_min'. Did you mean {key}_max or {key}_min?",
                                name=key, suggestions=(f"{key}_max", f"{key}_min"), accepted=accepted)
    if len(key) == 4:
        raise UnknownInputError(f"unknown limit {key!r}: the name in front of '{kind}' is missing.",
                                name=key, suggestions=(), accepted=accepted)
    name, side = key[:-4], kind[1:]
    if name in _PROPS:
        return name, side
    if name.lower() in _RESULT_ONLY:
        return name.lower(), side
    try:
        return resolve_name(system, name), side
    except UnknownInputError as e:
        raise UnknownInputError(f"unknown limit {key!r}: {str(e).split(': ', 1)[-1]}", name=key,
                                suggestions=tuple(f"{s}_{side}" for s in e.suggestions),
                                accepted=accepted) from None


def check_limit_value(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise InputError(f"limit {key} must be a finite real number, got {value!r}.")
    return float(value)


def _state_id(cs, name: str) -> "int | str | None":
    """State id for a state-property variable such as ``P2`` or ``T2s``, else None."""
    if len(name) > 1 and name[0] in _PROPS:
        rest = name[1:]
        for sid in cs.states:
            if str(sid) == rest:
                return sid
        for sid in cs.hidden_states:
            if str(sid) == rest:
                return sid
    return None


def _extreme(cs, letter: str, side: str) -> "tuple[float, int | str]":
    every = list(cs.states.items()) + list(cs.hidden_states.items())
    pick = max if side == "max" else min
    sid, st = pick(every, key=lambda kv: getattr(kv[1], letter))
    return float(getattr(st, letter)), sid


def evaluate_limits(cs, limits: "Mapping[str, float]") -> MarginReport:
    """Margins of the solved ``cs`` (a ``CycleState``) to ``limits``, plus the second-law report."""
    from .solver import system_for

    system = system_for(cs.definition, cs.fluid)
    margins: "dict[str, Margin]" = {}
    for key, raw in limits.items():
        name, side = parse_limit_key(key, system)
        limit = check_limit_value(key, raw)
        where: "int | str | None" = None
        if name in _PROPS:
            value, where = _extreme(cs, name, side)
        elif name in _RESULT_ONLY:
            value = float(getattr(cs.results, name))
        else:
            value = float(cs.values[name])
            where = _state_id(cs, name)
        margin = limit - value if side == "max" else value - limit
        pct = 100.0 * margin / abs(limit) if limit != 0 else math.nan
        margins[key] = Margin(key, side, limit, value, where, margin, pct, bool(margin >= 0))
    r = cs.results
    return MarginReport(margins, SecondLaw(r.eta_th, r.eta_carnot, r.eta_ii, r.eta_carnot - r.eta_th))


def describe_violations(violated: "tuple[Margin, ...]") -> str:
    """One text listing every violated limit (used by the warning and the error)."""
    parts = []
    for m in violated:
        rel = ">" if m.kind == "max" else "<"
        at = "" if m.where is None else f" at state {m.where}"
        parts.append(f"{m.key}: value {m.value:.6g}{at} {rel} limit {m.limit:.6g} "
                     f"(margin {m.margin:.4g}, {m.margin_pct:.1f} %)")
    return "limit(s) violated: " + "; ".join(parts) + "."
