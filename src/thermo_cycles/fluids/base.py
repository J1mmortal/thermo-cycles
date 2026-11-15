"""Fluid protocol and default implementation (spec section 6.1)."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from scipy.optimize import brentq

from ..errors import DegenerateStateError, FluidError, UnsupportedPairError
from ..model.relations import Relation
from ..state import PROPS


@runtime_checkable
class Fluid(Protocol):
    name: str
    props: "tuple[str, ...]"

    def state(self, **two: float) -> "dict[str, float]": ...

    def independent(self, a: str, b: str) -> bool: ...

    def relations(self, sid: str) -> "list[Relation] | None": ...


class FluidBase:
    """Defaults shared by fluids.

    Subclasses implement ``_evaluate(**pair)`` for the pairs in ``supported_pairs``
    (``None`` means every independent pair). ``state()`` falls back to a bracketed
    1-D root find (``scipy.optimize.brentq``) on a supported pair for the others;
    that needs ``bracket(prop)`` to return a ``(lo, hi)`` interval for the free property.
    """

    name: str = "fluid"
    props: "tuple[str, ...]" = PROPS
    supported_pairs: "tuple[frozenset, ...] | None" = None
    degenerate_pairs: "tuple[frozenset, ...]" = ()

    # -- hooks ------------------------------------------------------------
    def _evaluate(self, **pair: float) -> "dict[str, float]":
        raise UnsupportedPairError(f"{self.name}: no direct evaluation for {sorted(pair)}")

    def bracket(self, prop: str) -> "tuple[float, float]":
        raise UnsupportedPairError(f"{self.name}: no bracket for {prop!r}, cannot use the root-find fallback")

    # -- public API -------------------------------------------------------
    def independent(self, a: str, b: str) -> bool:
        a, b = self._canon(a), self._canon(b)
        return a != b and frozenset((a, b)) not in self.degenerate_pairs

    def relations(self, sid: str) -> "list[Relation] | None":
        return None

    def state(self, **two: float) -> "dict[str, float]":
        pair = self._clean_pair(two)
        (a, va), (b, vb) = pair.items()
        if not self.independent(a, b):
            raise DegenerateStateError(
                f"{self.name}: {a} and {b} are not independent, so they do not fix a state. "
                f"Give one of them together with a different property."
            )
        if self.supported_pairs is None or frozenset((a, b)) in self.supported_pairs:
            return self._evaluate(**pair)
        return self._fallback(a, va, b, vb)

    # -- internals --------------------------------------------------------
    def _canon(self, name: str) -> str:
        for p in self.props:
            if p.lower() == str(name).lower():
                return p
        raise FluidError(f"{self.name}: unknown property {name!r}; known: {self.props}")

    def _clean_pair(self, two: dict) -> dict:
        if len(two) != 2:
            raise FluidError(f"{self.name}: state() needs exactly two properties, got {sorted(two)}")
        out = {}
        for k, v in two.items():
            key = self._canon(k)
            if key in out:
                raise FluidError(f"{self.name}: state() got the same property twice: {sorted(two)}")
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise FluidError(f"{self.name}: {key} must be a number, got {v!r}") from None
            if not math.isfinite(fv):
                raise FluidError(f"{self.name}: {key} must be finite, got {v!r}")
            out[key] = fv
        return out

    def _fallback(self, a: str, va: float, b: str, vb: float) -> "dict[str, float]":
        """Solve for c in a supported pair (a, c) so that property b equals vb."""
        for c in self.props:
            if c in (a, b) or frozenset((a, c)) not in (self.supported_pairs or ()):
                continue
            try:
                lo, hi = self.bracket(c)
            except UnsupportedPairError:
                continue

            def f(x, a=a, va=va, c=c, b=b, vb=vb):
                return self._evaluate(**{a: va, c: x})[b] - vb

            if f(lo) * f(hi) > 0:
                raise FluidError(f"{self.name}: no state with {a}={va:g} and {b}={vb:g} inside the "
                                 f"validity range of {c} [{lo:g}, {hi:g}]")
            root = brentq(f, lo, hi, xtol=1e-14, rtol=1e-13)
            return self._evaluate(**{a: va, c: root})
        raise UnsupportedPairError(f"{self.name}: cannot evaluate the pair ({a}, {b})")
