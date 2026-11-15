"""Process step kinds (spec section 7)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..state import State
from .relations import Relation, _div, default_scale, equal, fixed, linear

if TYPE_CHECKING:
    from ..fluids.base import Fluid

_SYSTEMS = ("closed", "flow")


def _check_system(system):
    if system not in _SYSTEMS:
        raise ValueError(f"system must be one of {_SYSTEMS}, got {system!r}")


def _prod_diff(out, f, x, y, name):
    """out - f*(x - y) = 0, explicit in every variable."""
    return Relation(
        name, (out, f, x, y), lambda o, f_, x_, y_: o - f_ * (x_ - y_),
        {out: lambda **k: k[f] * (k[x] - k[y]),
         f: lambda **k: _div(k[out], k[x] - k[y], name, f),
         x: lambda **k: k[y] + _div(k[out], k[f], name, x),
         y: lambda **k: k[x] - _div(k[out], k[f], name, y)},
        default_scale(out))


def _energy(system):
    """Property letter of the energy balance: u (closed) or h (flow)."""
    return "u" if system == "closed" else "h"


def _w_from_state_change(a, b, system):
    e = _energy(system)
    return linear(f"w_{a}{b}", {f"{e}{b}": -1.0, f"{e}{a}": 1.0}, f"w {a}->{b} = -d{e}")


class StepKind(ABC):
    name: str
    reversible: bool = True

    @abstractmethod
    def relations(self, a: str, b: str, *, system: str, role: str, efficiency: "str | None") -> "list[Relation]":
        ...

    def hidden_states(self, b: str) -> "tuple[str, ...]":
        return ()

    def path(self, fluid: "Fluid", sa: State, sb: State, n: int = 100) -> "list[State] | None":
        return None

    # -- shared helpers ---------------------------------------------------
    def _label(self, a, b):
        return f"{self.name} {a}->{b}"

    def _sample(self, fluid, sa, sb, n, fixed_prop, fixed_val, swept, bounds, spacing="linear"):
        if n < 2:
            raise ValueError(f"path needs n >= 2 points, got {n}")
        self._check_endpoints(sa, sb)
        lo, hi = bounds
        out = []
        for i in range(n):
            t = i / (n - 1)
            if spacing == "log":
                x = math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))
            else:
                x = lo + t * (hi - lo)
            d = fluid.state(**{fixed_prop: fixed_val, swept: x})
            out.append(State(id=f"{sa.id}->{sb.id}:{i}", **d))
        return out

    def _check_endpoints(self, sa, sb):
        pass

    def _require_equal(self, sa, sb, prop, tol):
        va, vb = getattr(sa, prop), getattr(sb, prop)
        if abs(va - vb) > tol:
            raise ValueError(f"{self.name} path from state {sa.id} to {sb.id}: {prop} differs "
                             f"({va:g} vs {vb:g}), so the end states do not lie on one {self.name} process")


class Isentropic(StepKind):
    name = "isentropic"

    def relations(self, a, b, *, system, role, efficiency):
        _check_system(system)
        return [equal(f"s{b}", f"s{a}", self._label(a, b)),
                fixed(f"q_{a}{b}", 0.0),
                _w_from_state_change(a, b, system)]

    def _check_endpoints(self, sa, sb):
        self._require_equal(sa, sb, "s", 1e-6 * 100.0)

    def path(self, fluid, sa, sb, n=100):
        return self._sample(fluid, sa, sb, n, "s", sa.s, "P", (sa.P, sb.P), "log")


class Isochoric(StepKind):
    name = "isochoric"

    def relations(self, a, b, *, system, role, efficiency):
        _check_system(system)
        rel = [equal(f"v{b}", f"v{a}", self._label(a, b))]
        q, w = f"q_{a}{b}", f"w_{a}{b}"
        if system == "closed":
            rel.append(linear(q, {f"u{b}": 1.0, f"u{a}": -1.0}, f"q {a}->{b} = du"))
            rel.append(fixed(w, 0.0))
        else:
            rel.append(_prod_diff(w, f"v{a}", f"P{a}", f"P{b}", f"w {a}->{b} = -v dP"))
            nq = f"q {a}->{b} = dh - v dP"
            va, pa, pb, hb, ha = f"v{a}", f"P{a}", f"P{b}", f"h{b}", f"h{a}"
            rel.append(Relation(
                nq, (q, hb, ha, va, pb, pa),
                lambda q_, hb_, ha_, v_, pb_, pa_: q_ - ((hb_ - ha_) - v_ * (pb_ - pa_)),
                {q: lambda **k: (k[hb] - k[ha]) - k[va] * (k[pb] - k[pa]),
                 hb: lambda **k: k[q] + k[ha] + k[va] * (k[pb] - k[pa]),
                 ha: lambda **k: k[hb] - k[q] - k[va] * (k[pb] - k[pa]),
                 pb: lambda **k: k[pa] + _div(k[hb] - k[ha] - k[q], k[va], nq, pb),
                 pa: lambda **k: k[pb] - _div(k[hb] - k[ha] - k[q], k[va], nq, pa),
                 va: lambda **k: _div(k[hb] - k[ha] - k[q], k[pb] - k[pa], nq, va)},
                default_scale(q)))
        return rel

    def _check_endpoints(self, sa, sb):
        self._require_equal(sa, sb, "v", 1e-6 * abs(sa.v))

    def path(self, fluid, sa, sb, n=100):
        return self._sample(fluid, sa, sb, n, "v", sa.v, "u", (sa.u, sb.u))


class Isobaric(StepKind):
    name = "isobaric"

    def relations(self, a, b, *, system, role, efficiency):
        _check_system(system)
        rel = [equal(f"P{b}", f"P{a}", self._label(a, b)),
               linear(f"q_{a}{b}", {f"h{b}": 1.0, f"h{a}": -1.0}, f"q {a}->{b} = dh")]
        if system == "closed":
            rel.append(_prod_diff(f"w_{a}{b}", f"P{a}", f"v{b}", f"v{a}", f"w {a}->{b} = P dv"))
        else:
            rel.append(fixed(f"w_{a}{b}", 0.0))
        return rel

    def _check_endpoints(self, sa, sb):
        self._require_equal(sa, sb, "P", 1e-6 * abs(sa.P))

    def path(self, fluid, sa, sb, n=100):
        return self._sample(fluid, sa, sb, n, "P", sa.P, "h", (sa.h, sb.h))


class Isothermal(StepKind):
    name = "isothermal"

    def relations(self, a, b, *, system, role, efficiency):
        _check_system(system)
        q, w, e = f"q_{a}{b}", f"w_{a}{b}", _energy(system)
        return [equal(f"T{b}", f"T{a}", self._label(a, b)),
                _prod_diff(q, f"T{a}", f"s{b}", f"s{a}", f"q {a}->{b} = T ds"),
                linear(w, {q: 1.0, f"{e}{b}": -1.0, f"{e}{a}": 1.0}, f"w {a}->{b} = q - d{e}")]

    def _check_endpoints(self, sa, sb):
        self._require_equal(sa, sb, "T", 1e-6 * abs(sa.T))

    def path(self, fluid, sa, sb, n=100):
        return self._sample(fluid, sa, sb, n, "T", sa.T, "s", (sa.s, sb.s))


class Adiabatic(StepKind):
    """Adiabatic with isentropic efficiency; hidden isentropic end state ``<b>s``."""

    name = "adiabatic"
    reversible = False

    def hidden_states(self, b):
        return (f"{b}s",)

    def relations(self, a, b, *, system, role, efficiency):
        _check_system(system)
        if role not in ("compression", "expansion"):
            raise ValueError(f"adiabatic step {a}->{b} needs role 'compression' or 'expansion', got {role!r}")
        bs = f"{b}s"
        lab = self._label(a, b)
        rel = [equal(f"s{bs}", f"s{a}", f"{lab} (isentropic end state)"),
               equal(f"P{bs}", f"P{b}", f"{lab} (same end pressure)")]
        ha, hb, hbs = f"h{a}", f"h{b}", f"h{bs}"
        nm = f"{lab} (efficiency)"
        if efficiency is None:
            rel.append(equal(hbs, hb, f"{lab} (eta = 1)"))
        else:
            eta = efficiency
            if role == "compression":   # (hbs - ha) = eta (hb - ha)
                rel.append(Relation(
                    nm, (hbs, ha, hb, eta), lambda s_, a_, b_, e_: (s_ - a_) - e_ * (b_ - a_),
                    {hbs: lambda **k: k[ha] + k[eta] * (k[hb] - k[ha]),
                     hb: lambda **k: k[ha] + _div(k[hbs] - k[ha], k[eta], nm, hb),
                     ha: lambda **k: _div(k[eta] * k[hb] - k[hbs], k[eta] - 1.0, nm, ha),
                     eta: lambda **k: _div(k[hbs] - k[ha], k[hb] - k[ha], nm, eta)},
                    1e5))
            else:                        # (ha - hb) = eta (ha - hbs)
                rel.append(Relation(
                    nm, (hbs, ha, hb, eta), lambda s_, a_, b_, e_: (a_ - b_) - e_ * (a_ - s_),
                    {hb: lambda **k: k[ha] - k[eta] * (k[ha] - k[hbs]),
                     hbs: lambda **k: k[ha] - _div(k[ha] - k[hb], k[eta], nm, hbs),
                     ha: lambda **k: _div(k[hb] - k[eta] * k[hbs], 1.0 - k[eta], nm, ha),
                     eta: lambda **k: _div(k[ha] - k[hb], k[ha] - k[hbs], nm, eta)},
                    1e5))
        rel.append(fixed(f"q_{a}{b}", 0.0))
        rel.append(_w_from_state_change(a, b, system))
        return rel


KINDS: "dict[str, StepKind]" = {k.name: k for k in (Isentropic(), Isochoric(), Isobaric(), Isothermal(), Adiabatic())}
