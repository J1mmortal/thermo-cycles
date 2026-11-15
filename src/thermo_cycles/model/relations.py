"""Scalar relations with explicit inverses (spec section 9.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

_SCALES = {"T": 300.0, "P": 1e5, "v": 1.0, "u": 1e5, "h": 1e5, "q": 1e5, "w": 1e5, "s": 100.0}


def default_scale(var: str) -> float:
    """Typical magnitude from the variable name: T 300, P/u/h/q/w 1e5, s 100, else 1.

    Additions to the plan's rule: ``mep`` is a pressure (1e5) and ``dT...`` a temperature (300).
    """
    if var.startswith("mep"):
        return 1e5
    if var.startswith("dT"):
        return 300.0
    return _SCALES.get(var[:1], 1.0)


def _div(x: float, y: float, rel: str, var: str) -> float:
    if y == 0:
        raise ZeroDivisionError(f"relation {rel!r}: cannot solve for {var!r}, the divisor is zero")
    return x / y


@dataclass(frozen=True)
class Relation:
    name: str
    vars: "tuple[str, ...]"
    residual: Callable[..., float]
    solvers: Mapping[str, Callable[..., float]]
    scale: float = 1.0

    def __post_init__(self):
        if len(set(self.vars)) != len(self.vars):
            raise ValueError(f"relation {self.name!r}: duplicate variables in {self.vars}")
        bad = [k for k in self.solvers if k not in self.vars]
        if bad:
            raise ValueError(f"relation {self.name!r}: inverse for {bad} which are not in vars {self.vars}")
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"relation {self.name!r}: scale must be finite and positive, got {self.scale!r}")

    def solve_for(self, var: str, values: Mapping[str, float]) -> float:
        """Explicit inverse for ``var`` from the other variables in ``values`` (extra keys ignored)."""
        if var not in self.solvers:
            raise KeyError(f"relation {self.name!r} has no explicit inverse for {var!r}")
        others = [v for v in self.vars if v != var]
        missing = [v for v in others if v not in values]
        if missing:
            raise ValueError(f"relation {self.name!r}: solving for {var!r} needs {missing} as well")
        return float(self.solvers[var](**{v: values[v] for v in others}))


def _finite(x, what):
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {x!r}") from None
    if not math.isfinite(f):
        raise ValueError(f"{what} must be finite, got {x!r}")
    return f


def equal(a: str, b: str, name: str) -> Relation:
    """a - b = 0."""
    if a == b:
        raise ValueError(f"equal({a!r}, {b!r}): the two variables must differ")
    return Relation(name, (a, b), lambda x, y: x - y,
                    {a: lambda **k: k[b], b: lambda **k: k[a]}, default_scale(a))


def ratio(out: str, num: str, den: str, name: str | None = None) -> Relation:
    """out * den - num = 0. Scale follows ``num`` (the residual has the units of num)."""
    nm = name or f"{out} = {num}/{den}"
    return Relation(
        nm, (out, num, den), lambda o, n, d: o * d - n,
        {out: lambda **k: _div(k[num], k[den], nm, out),
         num: lambda **k: k[out] * k[den],
         den: lambda **k: _div(k[num], k[out], nm, den)},
        default_scale(num))


def linear(out: str, terms: "dict[str, float]", name: str | None = None) -> Relation:
    """out - sum(c * x) = 0."""
    if not terms:
        raise ValueError("linear(): needs at least one term")
    if out in terms:
        raise ValueError(f"linear(): output {out!r} also appears among the terms")
    coef = {x: _finite(c, f"coefficient of {x}") for x, c in terms.items()}
    if any(c == 0 for c in coef.values()):
        raise ValueError(f"linear(): zero coefficient in {coef}")
    nm = name or f"{out} = " + " + ".join(f"{c:g}*{x}" for x, c in coef.items())
    xs = tuple(coef)

    def residual(o, *vals):
        return o - sum(c * v for c, v in zip(coef.values(), vals))

    solvers = {out: lambda **k: sum(c * k[x] for x, c in coef.items())}
    for t in xs:
        solvers[t] = (lambda t: lambda **k: (k[out] - sum(c * k[x] for x, c in coef.items() if x != t)) / coef[t])(t)
    return Relation(nm, (out,) + xs, residual, solvers, default_scale(out))


def product(out: str, a: str, b: str, name: str | None = None) -> Relation:
    """out - a * b = 0."""
    nm = name or f"{out} = {a}*{b}"
    return Relation(
        nm, (out, a, b), lambda o, x, y: o - x * y,
        {out: lambda **k: k[a] * k[b],
         a: lambda **k: _div(k[out], k[b], nm, a),
         b: lambda **k: _div(k[out], k[a], nm, b)},
        default_scale(out))


def fixed(var: str, value: float) -> Relation:
    """var - value = 0 (inputs)."""
    val = _finite(value, f"fixed value of {var}")
    return Relation(f"{var} = {val:g}", (var,), lambda x: x - val, {var: lambda **k: val}, default_scale(var))
