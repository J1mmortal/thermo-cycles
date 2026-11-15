"""Optimisation over cycle inputs with user limits as constraints (spec section 12).

``optimise`` replaces the old EFFICIENCY mode. It varies some inputs of a cycle between bounds, keeps the
others fixed, and maximises or minimises a result while every limit (spec 11 key grammar) holds.

- Before any search the input set ``fixed`` + ``vary`` must be well-posed (``requirements`` status ``ok``);
  otherwise the usual ``UnderdeterminedError`` / ``OverdeterminedError`` is raised.
- A point is infeasible when its solve raises (no solution, invalid cycle, ...) or a limit is violated
  (margin < 0). Feasibility is strict: a returned optimum never violates a limit, not even by rounding.
- 1 variable: ``grid`` points (log-spaced when hi/lo > 10), then a bounded refinement next to the best feasible
  point and, towards an infeasible neighbour, the boundary (``brentq`` on the binding limit's margin, or a
  bisection on solvability when the neighbour did not solve). The best candidate wins.
- 2 or more variables: a scrambled Sobol sample (64 d points, ``seed``), then SLSQP from the 3 best feasible
  samples with each limit as an inequality constraint margin / |limit| >= 0. The best result wins.
- No feasible point raises ``NoFeasiblePointError`` listing, per limit, the smallest violation found.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy import optimize as _opt
from scipy.stats import qmc

from .cycle import Cycle, requirements
from .errors import (InputError, InputPlausibilityWarning, NoFeasiblePointError, OverdeterminedError,
                     ThermoCyclesError, UnderdeterminedError, UnknownCycleError, UnknownInputError)
from .fluids import AIR
from .margins import MarginReport, check_limit_value, parse_limit_key
from .model.cycle_def import get_cycle
from .results import CycleResults
from .solver import raise_for, resolve_name, system_for

__all__ = ["optimise", "OptimisationResult", "ACTIVE_PCT"]

ACTIVE_PCT = 1e-3                 # a limit is active when |margin_pct| < ACTIVE_PCT at the optimum (spec 12)
_N_STARTS = 3                     # SLSQP starts from the best feasible Sobol samples (spec 12)
_SOBOL_PER_DIM = 64               # Sobol sample size per variable (spec 12)
_X_RTOL = 1e-12                   # relative x tolerance of brentq and the bisections
_RESULT_NAMES = tuple(f for f in CycleResults.__dataclass_fields__)
# errors that say "the inputs themselves are wrong": never turned into an infeasible point
_STRUCTURAL = (UnderdeterminedError, OverdeterminedError, UnknownInputError, UnknownCycleError)


@dataclass(frozen=True)
class OptimisationResult:
    """Outcome of ``optimise`` (spec 12).

    x: the varied inputs at the optimum (canonical names). value: the objective there (as computed, not
    negated). cycle: the full solved ``Cycle`` at the optimum, with the limits. feasible: True (an infeasible
    search raises). active: limit keys with |margin_pct| < 1e-3 at the optimum, in the order given.
    evaluations: number of distinct cycle evaluations. history: one ``(x, value or None, feasible)`` per
    evaluation, in order (value None when the solve failed). message: how the optimum was found.
    """

    x: Mapping[str, float]
    value: float
    cycle: Cycle
    feasible: bool
    active: "tuple[str, ...]"
    evaluations: int
    history: "tuple[tuple[Mapping[str, float], float | None, bool], ...]"
    message: str


@dataclass
class _Point:
    x: "tuple[float, ...]"
    cycle: "Cycle | None"          # None when the solve failed
    value: "float | None"          # objective value (None when the solve failed)
    margins: "MarginReport | None"
    feasible: bool
    error: "str | None"

    def violated(self) -> "tuple[str, ...]":
        return () if self.margins is None else tuple(m.key for m in self.margins.violated)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _check_bounds(cyc: str, vary, system) -> "dict[str, tuple[float, float]]":
    if not isinstance(vary, _MappingABC) or not vary:
        raise InputError(f"{cyc}: vary must be a non-empty mapping such as {{'r_c': (5, 20)}}, got {vary!r}.")
    out: "dict[str, tuple[float, float]]" = {}
    for key, bounds in vary.items():
        name = resolve_name(system, key)
        if name in out:
            raise InputError(f"{cyc}: vary names {name} twice; give it once.")
        try:
            lo, hi = bounds
        except (TypeError, ValueError):
            raise InputError(f"{cyc}: vary[{key!r}] must be a pair (lo, hi), got {bounds!r}.") from None
        for b in (lo, hi):
            if isinstance(b, bool) or not isinstance(b, numbers.Real) or not math.isfinite(float(b)):
                raise InputError(f"{cyc}: the bounds of {name} must be finite real numbers, got {bounds!r}.")
        lo, hi = float(lo), float(hi)
        if not lo < hi:
            raise InputError(f"{cyc}: the bounds of {name} must satisfy lo < hi, got ({_fmt(lo)}, {_fmt(hi)}). "
                             f"To keep {name} constant, pass it in fixed instead.")
        out[name] = (lo, hi)
    return out


def _objective(cyc: str, system, maximise, minimise) -> "tuple[Callable[[Cycle], object], str, float]":
    if (maximise is None) == (minimise is None):
        raise InputError(f"{cyc}: give exactly one of maximise= or minimise= (a result name such as 'eta_th' "
                         f"or a callable f(cycle) -> float).")
    obj, sign = (maximise, 1.0) if maximise is not None else (minimise, -1.0)
    if callable(obj):
        return obj, getattr(obj, "__name__", "objective"), sign
    if not isinstance(obj, str):
        raise InputError(f"{cyc}: the objective must be a result or variable name such as 'eta_th', or a "
                         f"callable f(cycle) -> float, got {obj!r}.")
    lower = {n.lower(): n for n in _RESULT_NAMES}
    if obj.lower() in lower:
        name = lower[obj.lower()]
    else:
        try:
            name = resolve_name(system, obj)
        except UnknownInputError as e:
            raise UnknownInputError(f"{cyc}: unknown objective {obj!r}: {str(e).split(': ', 1)[-1]} "
                                    f"Cycle results are also accepted: {', '.join(_RESULT_NAMES)}.",
                                    name=obj, suggestions=e.suggestions,
                                    accepted=tuple(e.accepted) + _RESULT_NAMES) from None
    return (lambda c, _n=name: c.results.as_dict()[_n] if _n in _RESULT_NAMES else c.value(_n)), name, sign


class _Search:
    """Evaluation bookkeeping shared by the 1-D and n-D searches."""

    def __init__(self, name, fluid, fixed, limits, names, bounds, objective, label, sign):
        self.name, self.fluid, self.fixed, self.limits = name, fluid, dict(fixed), dict(limits)
        self.names, self.bounds = names, bounds
        self.objective, self.label, self.sign = objective, label, sign
        self.cache: "dict[tuple[float, ...], _Point]" = {}
        self.history: "list[tuple[Mapping[str, float], float | None, bool]]" = []

    def xmap(self, x) -> "dict[str, float]":
        return dict(zip(self.names, (float(v) for v in x)))

    def where(self, x) -> str:
        return ", ".join(f"{k} = {_fmt(v)}" for k, v in self.xmap(x).items())

    def evaluate(self, x) -> _Point:
        x = tuple(min(max(float(v), lo), hi) for v, (lo, hi) in zip(x, self.bounds))
        hit = self.cache.get(x)
        if hit is not None:
            return hit
        cycle, error = None, None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InputPlausibilityWarning)   # already shown for the bounds
                cycle = Cycle(self.name, fluid=self.fluid, limits=self.limits or None, on_violation="ignore",
                              **self.fixed, **self.xmap(x))
        except _STRUCTURAL:
            raise
        except ThermoCyclesError as e:                 # no solution, invalid cycle, ...: an infeasible point
            error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        if cycle is None:
            p = _Point(x, None, None, None, False, error)
        else:
            raw = self.objective(cycle)
            if raw is None:
                why = {"mep": " (mep exists for closed cycles only)",
                       "bwr": " (bwr exists for flow cycles only)"}.get(self.label, "")
                raise InputError(f"{self.name}: the objective {self.label} is not defined for this cycle{why}; "
                                 f"it is None at {self.where(x)}. Choose another objective.")
            value = float(raw)
            if not math.isfinite(value):
                raise InputError(f"{self.name}: the objective {self.label} is {value} at {self.where(x)}; it "
                                 f"must be a finite number.")
            p = _Point(x, cycle, value, cycle.margins, cycle.margins.ok, None)
        self.cache[x] = p
        self.history.append((MappingProxyType(self.xmap(x)), p.value, p.feasible))
        return p

    def goal(self, p: _Point) -> float:
        """Larger is better."""
        return self.sign * p.value

    def best(self, points) -> "_Point | None":
        feas = [p for p in points if p is not None and p.feasible]
        return max(feas, key=self.goal) if feas else None

    def margin(self, p: _Point, key: str) -> float:
        return p.margins[key].margin

    # 1-D helpers ------------------------------------------------------------------------------------

    def pull_back(self, p: _Point, toward: "tuple[float, ...]") -> "_Point | None":
        """``p`` if feasible, else the first feasible point a growing fraction of the way to ``toward``."""
        if p.feasible:
            return p
        a, b = np.asarray(p.x), np.asarray(toward)
        for s in 10.0 ** np.arange(-12, 0):
            q = self.evaluate(tuple(a + s * (b - a)))
            if q.feasible:
                return q
        return None

    def bisect(self, good: _Point, bad: _Point) -> _Point:
        """Feasibility boundary on the segment good -> bad (``good`` feasible): the last feasible point."""
        a, b = np.asarray(good.x), np.asarray(bad.x)
        lo, hi = 0.0, 1.0
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
        while (hi - lo) * float(np.max(np.abs(b - a))) > _X_RTOL * scale:
            mid = 0.5 * (lo + hi)
            if self.evaluate(tuple(a + mid * (b - a))).feasible:
                lo = mid
            else:
                hi = mid
        return self.evaluate(tuple(a + lo * (b - a)))

    def boundary(self, good: _Point, bad: _Point) -> "list[_Point]":
        """Boundary points between a feasible point and an infeasible 1-D neighbour."""
        if bad.cycle is None:                              # the neighbour did not solve: edge of solvability
            return [self.bisect(good, bad)]
        out = []
        for key in bad.violated():
            if self.margin(good, key) < 0:
                continue

            def g(x, _k=key):
                p = self.evaluate((x,))
                if p.cycle is None:
                    raise _NoMargin
                return self.margin(p, _k)

            xa, xb = good.x[0], bad.x[0]
            try:
                root = _opt.brentq(g, xa, xb, xtol=_X_RTOL * max(abs(xa), abs(xb)), rtol=4 * np.finfo(float).eps,
                                   maxiter=200)
            except (_NoMargin, ValueError, RuntimeError):
                out.append(self.bisect(good, bad))
                continue
            q = self.pull_back(self.evaluate((root,)), good.x)
            out.append(q if q is not None else self.bisect(good, bad))
        if not out:                                        # violated limits are violated at `good` too
            out.append(self.bisect(good, bad))
        return out


class _NoMargin(Exception):
    """A solve failed inside the brentq bracket."""


def _grid(lo: float, hi: float, n: int) -> np.ndarray:
    if lo > 0 and hi / lo > 10:
        g = np.geomspace(lo, hi, n)
    else:
        g = np.linspace(lo, hi, n)
    g[0], g[-1] = lo, hi
    return g


def _search_1d(s: _Search, grid: int) -> "tuple[_Point | None, str]":
    lo, hi = s.bounds[0]
    xs = _grid(lo, hi, grid)
    pts = [s.evaluate((x,)) for x in xs]
    best = s.best(pts)
    if best is None:
        return None, ""
    i = next(k for k, p in enumerate(pts) if p is best)
    nbrs = [pts[k] for k in (i - 1, i + 1) if 0 <= k < len(pts)]
    cands = [best]
    notes: "list[str]" = []
    # interior refinement over the feasible neighbours (both, or the one there is at the edge of the grid)
    a = pts[i - 1].x[0] if i > 0 and pts[i - 1].feasible else best.x[0]
    b = pts[i + 1].x[0] if i + 1 < len(pts) and pts[i + 1].feasible else best.x[0]
    if a < b:
        worst = min(s.goal(p) for p in pts if p.feasible)
        penalty = -worst + abs(worst) + 1.0

        def f(x):
            p = s.evaluate((x,))
            return -s.goal(p) if p.feasible else penalty

        r = _opt.minimize_scalar(f, bounds=(a, b), method="bounded",
                                 options={"xatol": _X_RTOL * 1e3 * max(abs(a), abs(b)), "maxiter": 500})
        p = s.evaluate((r.x,))
        if p.feasible:
            cands.append(p)
        notes.append("bounded refinement")
    for nb in nbrs:
        if not nb.feasible:
            cands.extend(s.boundary(best, nb))
            notes.append("boundary search")
    win = s.best(cands)
    kind = "log-spaced" if lo > 0 and hi / lo > 10 else "linear"
    how = f"1-D: {grid}-point {kind} grid" + "".join(f", {n}" for n in dict.fromkeys(notes))
    return win, how


def _search_nd(s: _Search, seed: int) -> "tuple[_Point | None, str]":
    d = len(s.bounds)
    n = _SOBOL_PER_DIM * d
    log = [lo > 0 and hi / lo > 10 for lo, hi in s.bounds]

    def to_x(u):
        x = []
        for ui, (lo, hi), lg in zip(np.clip(u, 0.0, 1.0), s.bounds, log):
            if ui >= 1.0:
                x.append(hi)
            elif lg:
                x.append(lo * (hi / lo) ** ui)
            else:
                x.append(lo + ui * (hi - lo))
        return tuple(x)

    sample = qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(math.ceil(math.log2(n))))[:n]
    pts = [(u, s.evaluate(to_x(u))) for u in sample]
    feas = sorted((up for up in pts if up[1].feasible), key=lambda up: s.goal(up[1]), reverse=True)
    if not feas:
        return None, ""
    scale = max(abs(s.goal(feas[0][1])), 1e-300)
    worst = min(s.goal(p) for _, p in feas)
    penalty = (-worst + abs(worst) + scale) / scale
    keys = list(s.limits)

    def fun(u):
        p = s.evaluate(to_x(u))
        return -s.goal(p) / scale if p.cycle is not None else penalty

    def con(u, key):
        p = s.evaluate(to_x(u))
        if p.cycle is None:
            return -1.0
        m = p.margins[key]
        return m.margin / (abs(m.limit) if m.limit != 0 else max(abs(m.value), 1.0))

    cons = [{"type": "ineq", "fun": con, "args": (k,)} for k in keys]
    cands = [feas[0][1]]
    statuses = []
    for u0, p0 in feas[:_N_STARTS]:
        r = _opt.minimize(fun, np.asarray(u0), method="SLSQP", bounds=[(0.0, 1.0)] * d, constraints=cons,
                          options={"maxiter": 200, "ftol": 1e-12})
        statuses.append(r.status)
        p = s.evaluate(to_x(r.x))
        if not p.feasible:                     # a hair outside a limit: back along the line to the start
            p = s.bisect(p0, p)
        cands.append(p)
    win = s.best(cands)
    ok = sum(st == 0 for st in statuses)
    how = (f"{d}-D: {n}-point Sobol sample (seed {seed}), SLSQP from the {len(statuses)} best feasible samples "
           f"({ok} converged)")
    return win, how


def _no_feasible(s: _Search) -> NoFeasiblePointError:
    pts = list(s.cache.values())
    solved = [p for p in pts if p.margins is not None]
    span = "; ".join(f"{k} in [{_fmt(lo)}, {_fmt(hi)}]" for k, (lo, hi) in zip(s.names, s.bounds))
    lines = [f"{s.name}: no feasible point found in {len(pts)} evaluations over {span}."]
    closest = None
    if solved:
        closest = {}
        for key in s.limits:
            p = max(solved, key=lambda q, _k=key: s.margin(q, _k))
            m = p.margins[key]
            closest[key] = m
            if m.ok:
                lines.append(f"  {key} = {_fmt(m.limit)}: met at some points (best margin {_fmt(m.margin)}).")
            else:
                at = "" if m.where is None else f" at state {m.where}"
                lines.append(f"  {key} = {_fmt(m.limit)}: smallest violation {_fmt(-m.margin)} "
                             f"({_fmt(-m.margin_pct)} %), value {_fmt(m.value)}{at} at {s.where(p.x)}.")
        lines.append("  Relax a limit or move the bounds.")
    failed = [p for p in pts if p.cycle is None]
    if failed:
        lines.append(f"  {len(failed)} of {len(pts)} points did not solve; first: {failed[0].error} "
                     f"(at {s.where(failed[0].x)}).")
    return NoFeasiblePointError("\n".join(lines), closest=closest)


def optimise(name: str, *, vary: Mapping[str, "tuple[float, float]"], maximise: "str | Callable | None" = None,
             minimise: "str | Callable | None" = None, fixed: Mapping[str, float] = {},
             limits: Mapping[str, float] = {}, fluid=None, grid: int = 81,
             seed: int = 0) -> OptimisationResult:
    """Maximise or minimise a cycle result over the inputs in ``vary`` (spec 12).

    vary: ``{input: (lo, hi)}``; fixed: the other inputs; together they must be well-posed. maximise /
    minimise (exactly one): a result or solved-variable name (``eta_th``, ``w_net``, ``mep``, ``eta_ii``,
    ``bwr``, ``T3``, ...) or a callable ``f(cycle) -> float``. limits: spec 11 keys (``T_max``, ``P2_max``,
    ``eta_th_min``, ...), used as constraints. grid: points of the 1-D grid. seed: the Sobol seed (n-D).
    """
    defn = get_cycle(name)
    fl = AIR if fluid is None else fluid
    system = system_for(defn, fl)
    cyc = defn.name
    if not isinstance(fixed, _MappingABC):
        raise InputError(f"{cyc}: fixed must be a mapping of inputs such as {{'T1': 300}}, got {fixed!r}.")
    if not isinstance(limits, _MappingABC):
        raise InputError(f"{cyc}: limits must be a mapping such as {{'T_max': 2500}}, got {limits!r}.")
    if isinstance(grid, bool) or not isinstance(grid, numbers.Integral) or grid < 3:
        raise InputError(f"{cyc}: grid must be an integer >= 3, got {grid!r}.")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise InputError(f"{cyc}: seed must be an integer, got {seed!r}.")
    bounds = _check_bounds(cyc, vary, system)
    fixed_names = {resolve_name(system, k): k for k in fixed}
    both = [n for n in bounds if n in fixed_names]
    if both:
        raise InputError(f"{cyc}: {', '.join(both)} is both fixed and varied; give it in one of the two.")
    for key, val in limits.items():
        parse_limit_key(key, system)
        check_limit_value(key, val)
    objective, label, sign = _objective(cyc, system, maximise, minimise)

    # well-posedness first, with the bounds checked against the solver's ranges (spec 12, 9.8)
    req = requirements(name, fluid=fluid, **dict(fixed), **{k: lo for k, (lo, _) in bounds.items()})
    raise_for(req, cyc)
    requirements(name, fluid=fluid, **dict(fixed), **{k: hi for k, (_, hi) in bounds.items()})

    names = tuple(bounds)
    s = _Search(name, fluid, fixed, limits, names, tuple(bounds.values()), objective, label, sign)
    if len(names) == 1:
        win, how = _search_1d(s, int(grid))
    else:
        win, how = _search_nd(s, int(seed))
    if win is None:
        raise _no_feasible(s)

    x = MappingProxyType(s.xmap(win.x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputPlausibilityWarning)
        cycle = Cycle(name, fluid=fluid, limits=dict(limits) or None, **dict(fixed), **x)
    active = tuple(k for k, m in cycle.margins.items()
                   if not math.isnan(m.margin_pct) and abs(m.margin_pct) < ACTIVE_PCT)
    sense = "maximum" if sign > 0 else "minimum"
    msg = (f"{sense} of {label} = {_fmt(win.value)} at {s.where(win.x)}; "
           f"{'active limits: ' + ', '.join(active) if active else 'no active limit'}. {how}; "
           f"{len(s.history)} evaluations.")
    return OptimisationResult(x=x, value=win.value, cycle=cycle, feasible=True, active=active,
                              evaluations=len(s.history), history=tuple(s.history), message=msg)
