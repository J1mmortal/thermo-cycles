"""Complete thermodynamic state (spec section 5.1)."""

from __future__ import annotations

from dataclasses import dataclass

PROPS = ("T", "P", "v", "u", "h", "s")
_UNITS = {"T": "K", "P": "Pa", "v": "m3/kg", "u": "J/kg", "h": "J/kg", "s": "J/(kg K)"}
_LOOKUP = {name.lower(): name for name in PROPS + ("x",)}


@dataclass(frozen=True)
class State:
    """A complete, immutable state. ``x`` is vapour quality (None for single phase)."""

    id: "int | str"
    T: float
    P: float
    v: float
    u: float
    h: float
    s: float
    x: "float | None" = None

    def get(self, name: str):
        """Property by name, case-insensitive (``"t"`` -> T)."""
        key = _LOOKUP.get(str(name).lower())
        if key is None:
            raise KeyError(f"unknown state property {name!r}; known: {PROPS + ('x',)}")
        return getattr(self, key)

    def has(self, *names: str) -> bool:
        """True if every name is a known property with a value; unknown names give False."""
        for name in names:
            key = _LOOKUP.get(str(name).lower())
            if key is None or getattr(self, key) is None:
                return False
        return True

    def as_dict(self) -> "dict[str, float]":
        return {p: getattr(self, p) for p in PROPS}

    def summary(self) -> str:
        parts = [f"{p}={getattr(self, p):.6g} {_UNITS[p]}" for p in PROPS]
        if self.x is not None:
            parts.append(f"x={self.x:.6g}")
        return f"State#{self.id}(" + ", ".join(parts) + ")"
