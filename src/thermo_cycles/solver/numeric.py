"""Numeric block solver (spec section 9.5).

The system plus its inputs is ordered into blocks with :func:`blt_blocks`. Each block is solved in turn:

- ``explicit``: a 1x1 block whose relation has an explicit inverse for its variable;
- ``closure``: the 4x4 generic per-state closure of a fluid without sparse relations, solved by one call to
  ``fluid.state(pair)``;
- ``torn``: one tear variable t such that the other block relations evaluate one after another with explicit
  inverses, leaving one relation as the residual g(t). g is scanned over t's physical range (400 points) and
  every sign change is refined with ``brentq``;
- ``newton``: ``scipy.optimize.root(method="hybr")`` on the scaled residuals, positive variables in log space,
  from up to 8 seeds.

A block with several roots branches: every candidate is completed through the remaining blocks. With
``validate`` given, candidates with problems are dropped; if more than one remains,
:class:`AmbiguousSolutionError` names the user-facing variable that tells them apart.
"""

from __future__ import annotations

import math
from collections import ChainMap
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

from ..errors import (AmbiguousSolutionError, ConvergenceError, FluidError, NoSolutionError,
                      UnsupportedPairError)
from ..model.relations import default_scale
from ..state import PROPS
from .structure import blt_blocks, dm_partition

__all__ = ["SolveReport", "solve_numeric"]

N_SCAN = 400                 # torn scan points (spec 9.5)
XTOL_REL = 1e-12             # brentq xtol relative to |x|
DISTINCT_REL = 1e-6          # roots are distinct when their relative difference exceeds this
N_SEEDS = 8                  # Newton seeds
ROOT_ACCEPT = 1e-6           # scaled |g| at a refined sign change; larger means a pole, not a root
NEWTON_TOL = 1e-10           # scaled residual for a converged Newton solution
SOLVE_TOL = 1e-6             # final max scaled residual above this is a solver failure
MAX_SOLUTIONS = 64           # branching limit (candidate full solutions)
N_SWEEP = 60                 # points of the no-solution reach sweep (message only)

# inputs blamed first in a no-solution message: derived targets before design inputs such as q_in
_BLAME_ORDER = ("eta_th", "bwr", "w_net", "mep", "q_out", "q_in")
_EVAL_ERRORS = (ArithmeticError, ValueError, FluidError, KeyError, TypeError)


@dataclass(frozen=True)
class SolveReport:
    blocks: "tuple[tuple[str, ...], ...]"
    methods: "tuple[str, ...]"           # "explicit" | "torn" | "newton" | "closure"
    max_residual: float


# ---------------------------------------------------------------------------------------------------------
# formatting


def _fmt(var: str, x: float) -> str:
    """A value with its unit, short (spec 9.5 example: "w_net = 400 kJ/kg")."""

    def num(y):
        return f"{y:.3g}" if abs(y) < 1000 else f"{y:.0f}"

    if var.startswith(("mep", "P")):
        return f"{num(x / 1e3)} kPa"
    if var[:1] in ("q", "w", "u", "h"):
        return f"{num(x / 1e3)} kJ/kg"
    if var.startswith(("T", "dT")):
        return f"{num(x)} K"
    if var.startswith("s"):
        return f"{num(x)} J/(kg K)"
    if var.startswith("v"):
        return f"{x:.4g} m³/kg"
    return f"{x:.4g}"


def _distinct(a: float, b: float) -> bool:
    return abs(a - b) > DISTINCT_REL * max(abs(a), abs(b))


# ---------------------------------------------------------------------------------------------------------
# context


class _Ctx:
    def __init__(self, system, inputs, rels, blocks, cache, nested):
        self.system = system
        self.inputs = inputs
        self.rels = rels
        self.blocks = blocks                       # list of (tuple[Relation], tuple[str])
        self.cache = cache                         # tear plans by relation identity
        self.nested = nested
        self.var_block = {v: k for k, (_, vs) in enumerate(blocks) for v in vs}
        cyc = system.cycle
        self.params = {p.name for p in cyc.params} if cyc is not None else set()
        self.name = cyc.name if cyc is not None else "system"
        self.rank = {v: i for i, v in enumerate(system.candidates)}

    def user_facing(self, v: str) -> bool:
        return v in self.rank

    def ancestor_blocks(self, k: int) -> "set[int]":
        """Block k and every block whose values reach it."""
        seen, todo = {k}, [k]
        while todo:
            b = todo.pop()
            rels, vs = self.blocks[b]
            for r in rels:
                for v in r.vars:
                    j = self.var_block[v]
                    if j not in seen:
                        seen.add(j)
                        todo.append(j)
        return seen

    def ancestor_inputs(self, k: int) -> "list[str]":
        """Inputs (and defaults) whose values reach block k, in the order the user gave them."""
        found = {v for j in self.ancestor_blocks(k) for v in self.blocks[j][1]}
        given = list(self.inputs) + [d for d in self.system.defaults if d not in self.inputs]
        return [v for v in given if v in found]

    def determined_params(self, k: int) -> "list[str]":
        """Catalogue parameters (not given) that block k determines: in block k or downstream of it."""
        given = set(self.inputs) | set(self.system.defaults)
        out = [p for p in self.params if p not in given and k in self.ancestor_blocks(self.var_block[p])]
        return sorted(out, key=lambda v: self.rank.get(v, 1 << 30))


# ---------------------------------------------------------------------------------------------------------
# public entry


def solve_numeric(system, inputs: Mapping[str, float], *,
                  validate: "Callable[[Mapping[str, float]], list[str]] | None" = None
                  ) -> "tuple[dict[str, float], SolveReport]":
    """Solve ``system`` with ``inputs`` block by block (spec 9.5).

    The system plus inputs must be structurally well-determined (status ``ok``); the caller checks that
    first and phrases the fix (Task 7). Otherwise this raises ``ValueError``.

    Raises ``NoSolutionError`` (a block has no root in the physical range), ``AmbiguousSolutionError``
    (more than one solution survives ``validate``) or ``ConvergenceError`` (Newton found nothing).
    ``validate(values)`` returns a list of problems; it is only called when there is more than one candidate.
    """
    return _solve(system, inputs, validate, {}, False)


def _solve(system, inputs, validate, cache, nested):
    inputs = {str(k): float(v) for k, v in inputs.items()}
    rels = system.with_inputs(inputs)
    idx = {v: i for i, v in enumerate(system.variables)}
    eq_vars = [frozenset(idx[v] for v in r.vars) for r in rels]
    n = len(system.variables)
    dm = dm_partition(eq_vars, n)
    if not (len(rels) == n == dm.rank):
        free = sorted(system.variables[i] for i in dm.under_vars)
        surplus = sorted(rels[e].name for e in dm.over_eqs)
        raise ValueError(
            f"solve_numeric needs a structurally well-determined system (status ok): {len(rels)} relations "
            f"for {n} variables, structural rank {dm.rank}; free: {free}; surplus: {surplus}")
    blocks = [(tuple(rels[e] for e in b.eqs), tuple(system.variables[v] for v in b.vars))
              for b in blt_blocks(eq_vars, n)]
    ctx = _Ctx(system, inputs, rels, blocks, cache, nested)

    out: list = []
    try:
        _solve_from(ctx, 0, {}, [], [], out)
    except _TooMany:
        raise _ambiguous(ctx, out, none_valid=False, problems=[]) from None

    chosen = out
    rejected: "list[tuple[dict, list]]" = []
    if len(out) > 1 and validate is not None:
        chosen = []
        for sol in out:
            problems = list(validate(dict(sol[0])))
            if problems:
                rejected.append((sol, problems))
            else:
                chosen.append(sol)
        if not chosen:
            chosen = out
    if len(chosen) > 1:
        raise _ambiguous(ctx, chosen, none_valid=bool(rejected) and len(rejected) == len(out),
                         problems=[p for _, p in rejected])

    vals, methods, _ = chosen[0]
    res = max((abs(r.residual(*(vals[v] for v in r.vars))) / r.scale for r in rels), default=0.0)
    if not res <= SOLVE_TOL:
        raise ConvergenceError(
            f"{ctx.name}: the solution does not satisfy every relation (max scaled residual {res:.3g})",
            variables=tuple(system.variables), residual=res)
    report = SolveReport(tuple(vs for _, vs in blocks), tuple(methods), float(res))
    return {v: float(vals[v]) for v in system.variables}, report


# ---------------------------------------------------------------------------------------------------------
# branching over blocks


class _TooMany(Exception):
    pass


def _solve_from(ctx, k, vals, methods, branches, out):
    vals = dict(vals)
    methods = list(methods)
    while k < len(ctx.blocks):
        sols, method, tear = _solve_block(ctx, k, vals)
        if len(sols) == 1:
            vals.update(sols[0])
            methods.append(method)
            k += 1
            continue
        branch = branches + [(k, _report_var(ctx, k, tear, sols))]
        errors = []
        for s in sols:
            try:
                _solve_from(ctx, k + 1, {**vals, **s}, methods + [method], branch, out)
            except (NoSolutionError, ConvergenceError) as e:
                errors.append(e)
        if len(errors) == len(sols):
            raise errors[0]
        return
    out.append((vals, methods, branches))
    if len(out) > MAX_SOLUTIONS:
        raise _TooMany()


def _report_var(ctx, k, tear, sols) -> str:
    """The user-facing variable that tells the block's candidate solutions apart.

    A catalogue parameter of the block first (spec 9.5 example "r_c = 2.736 or 20.43", spec 15.3), then the
    torn variable, then (Newton) the first block variable in candidate order whose values differ.
    """
    _, vs = ctx.blocks[k]

    def differs(v):
        xs = [s[v] for s in sols]
        return any(_distinct(a, b) for i, a in enumerate(xs) for b in xs[i + 1:])

    for v in sorted((v for v in vs if v in ctx.params), key=lambda v: ctx.rank.get(v, 1 << 30)):
        if differs(v):
            return v
    if tear is not None:
        return tear
    for v in sorted(vs, key=lambda v: (not ctx.user_facing(v), ctx.rank.get(v, 1 << 30), vs.index(v))):
        if differs(v):
            return v
    return vs[0]


def _ambiguous(ctx, sols, none_valid, problems):
    branches = sols[0][2]
    var = None
    given = set(ctx.inputs) | set(ctx.system.defaults)
    params = sorted((p for p in ctx.params if p not in given), key=lambda v: ctx.rank.get(v, 1 << 30))
    for v in params + [v for _, v in branches]:
        xs = [s[0][v] for s in sols]
        if any(_distinct(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]):
            var = v
            break
    if var is None:
        var = next(v for v in ctx.system.variables
                   if any(_distinct(sols[0][0][v], s[0][v]) for s in sols[1:]))
    cands: "list[float]" = []
    for s in sols:
        x = s[0][var]
        if all(_distinct(x, c) for c in cands):
            cands.append(float(x))
    cands.sort()
    shown = " or ".join(f"{c:.4g}" for c in cands)
    msg = (f"{ctx.name}: the inputs allow {len(sols)} solutions: {var} = {shown}; pass {var} explicitly "
           f"(or another input that tells them apart).")
    if none_valid:
        first = "; ".join(p[0] for p in problems if p)
        msg += f" None of them passes the validity checks ({first})."
    return AmbiguousSolutionError(msg, variable=var, candidates=tuple(cands))


# ---------------------------------------------------------------------------------------------------------
# one block


def _solve_block(ctx, k, vals):
    """Solutions of block k given ``vals``: (list of {var: value}, method, tear variable or None)."""
    rels, vs = ctx.blocks[k]
    if len(vs) == 1 and vs[0] in rels[0].solvers:
        return [{vs[0]: _explicit(ctx, rels[0], vs[0], vals)}], "explicit", None
    sol = _closure(ctx, rels, vs, vals)
    if sol is not None:
        return [sol], "closure", None
    plan = _tear_plan(ctx, rels, vs)
    if plan is not None:
        return _torn(ctx, k, plan, vals), "torn", plan[0]
    return _newton(ctx, rels, vs, vals), "newton", None


def _explicit(ctx, rel, var, vals) -> float:
    try:
        x = rel.solve_for(var, vals)
    except _EVAL_ERRORS as e:
        raise NoSolutionError(
            f"{ctx.name}: relation {rel.name!r} cannot be solved for {var} with "
            + ", ".join(f"{v} = {vals[v]:.6g}" for v in rel.vars if v != var) + f": {e}",
            variables=(var,), detail=str(e)) from None
    if not math.isfinite(x):
        raise NoSolutionError(f"{ctx.name}: relation {rel.name!r} gives {var} = {x} (not finite)",
                              variables=(var,), detail=f"{var} = {x}")
    return x


def _closure(ctx, rels, vs, vals):
    """The generic 4x4 per-state closure (spec 9.5): solved by one ``fluid.state(pair)`` call."""
    fluid = getattr(ctx.system, "fluid", None)
    if fluid is None or len(rels) != 4 or not all(r.name.startswith("closure ") for r in rels):
        return None
    sids = {r.name.split()[1] for r in rels}
    if len(sids) != 1:
        return None
    sid = sids.pop()
    names = {p: f"{p}{sid}" for p in PROPS}
    known = {p: vals[n] for p, n in names.items() if n not in vs}
    if len(known) != 2 or set(vs) != {n for p, n in names.items() if p not in known}:
        return None
    try:
        st = fluid.state(**known)
        sol = {names[p]: float(st[p]) for p in PROPS if p not in known}
    except UnsupportedPairError:
        return None
    except _EVAL_ERRORS as e:
        raise NoSolutionError(f"{ctx.name}: state {sid} has no fluid state with "
                              + ", ".join(f"{names[p]} = {x:.6g}" for p, x in known.items()) + f": {e}",
                              variables=tuple(vs), detail=str(e)) from None
    if not all(math.isfinite(x) for x in sol.values()):
        return None
    return sol


# --- tearing -----------------------------------------------------------------------------------------


def _tear_order(ctx, vs):
    """Tear preference: catalogue parameters, positive kinds (by candidate rank), fractions, signed."""
    kind_rank = {"positive": 1, "fraction": 2, "signed": 3}

    def key(v):
        first = 0 if v in ctx.params else kind_rank.get(ctx.system.var_kind.get(v), 4)
        return (first, ctx.rank.get(v, 1 << 30), vs.index(v))

    return sorted(vs, key=key)


def _tear_plan(ctx, rels, vs):
    """(tear, residual relation, [(relation, var), ...]) or None if no single tear reduces the block."""
    key = (tuple(id(r) for r in rels), vs)
    if key in ctx.cache:
        return ctx.cache[key]
    plan = None
    for t in _tear_order(ctx, vs):
        for ri, r in enumerate(rels):
            order = _propagate(rels[:ri] + rels[ri + 1:], set(vs) - {t})
            if order is not None:
                plan = (t, r, order)
                break
        if plan is not None:
            break
    ctx.cache[key] = plan
    return plan


def _propagate(rels, unknown):
    unknown = set(unknown)
    remaining = list(rels)
    order = []
    while remaining:
        for i, r in enumerate(remaining):
            unk = [v for v in r.vars if v in unknown]
            if not unk:
                return None                     # a second residual: one tear is not enough
            if len(unk) == 1 and unk[0] in r.solvers:
                order.append((r, unk[0]))
                unknown.discard(unk[0])
                del remaining[i]
                break
        else:
            return None
    return order if not unknown else None


def _grid(lo, hi, spacing, n=N_SCAN):
    if spacing == "log":
        return np.geomspace(lo, hi, n)
    return np.sinh(np.linspace(math.asinh(lo), math.asinh(hi), n))


def _evaluator(plan, vals):
    t, r, order = plan

    def evaluate(tv):
        local = {t: float(tv)}
        cm = ChainMap(local, vals)
        for rel, var in order:
            x = rel.solve_for(var, cm)
            if not math.isfinite(x):
                raise ArithmeticError(f"{var} not finite")
            local[var] = x
        g = r.residual(*(cm[v] for v in r.vars)) / r.scale
        return g, local

    def g(tv):
        try:
            val = evaluate(tv)[0]
        except _EVAL_ERRORS:
            return math.nan
        return val if math.isfinite(val) else math.nan

    return evaluate, g


def _torn(ctx, k, plan, vals):
    t, r, _ = plan
    lo, hi, spacing = ctx.system.scan_range[t]
    evaluate, g = _evaluator(plan, vals)
    xs = _grid(lo, hi, spacing)
    gs = [g(x) for x in xs]
    roots: "list[float]" = []

    def add(x):
        if all(_distinct(x, y) for y in roots):
            roots.append(float(x))

    for i, (a, ga) in enumerate(zip(xs, gs)):
        if ga == 0.0:
            add(a)
        if i + 1 == len(xs):
            break
        b, gb = xs[i + 1], gs[i + 1]
        if not (math.isfinite(ga) and math.isfinite(gb)) or ga == 0.0 or gb == 0.0 or ga * gb > 0:
            continue
        xtol = max(XTOL_REL * max(abs(a), abs(b)), 1e-300)
        try:
            x = brentq(g, a, b, xtol=xtol, rtol=4 * np.finfo(float).eps)
        except (ValueError, RuntimeError):
            continue
        gx = g(x)
        if math.isfinite(gx) and (abs(gx) <= ROOT_ACCEPT or abs(gx) < 1e-3 * min(abs(ga), abs(gb))):
            add(x)
    if not roots:
        raise _no_solution(ctx, k, plan, xs, gs)
    sols = []
    for x in roots:
        _, local = evaluate(x)
        sols.append(local)
    return sols


def _no_solution(ctx, k, plan, xs, gs):
    t, r, _ = plan
    _, vs = ctx.blocks[k]
    lo, hi, _ = ctx.system.scan_range[t]
    detail = _reach_message(ctx, k, plan) if not ctx.nested else None
    if detail is None:
        finite = [(abs(gv), x) for x, gv in zip(xs, gs) if math.isfinite(gv)]
        if finite:
            gmin, xmin = min(finite)
            detail = (f"relation {r.name!r} has no root for {t} in [{lo:g}, {hi:g}]; closest approach: scaled "
                      f"residual {gmin:.3g} at {t} = {xmin:.4g}")
        else:
            detail = f"relation {r.name!r} cannot be evaluated for any {t} in [{lo:g}, {hi:g}]"
    unsolved = tuple(dict.fromkeys(tuple(vs) + tuple(ctx.determined_params(k))))
    return NoSolutionError(f"{ctx.name}: no solution. {detail}", variables=unsolved, detail=detail)


def _reach_message(ctx, k, plan):
    """Phrase a missing root as an input out of reach (spec 9.5: "w_net = 400 kJ/kg exceeds what T1, P1,
    T3 allow (max ≈ 329 kJ/kg near r_c = 7.48)").

    Free one input x that feeds the block, fix the block's report variable p instead, and sweep p over its
    scan range with nested solves. If the given x lies outside the swept range of x, say so.
    """
    t, _, _ = plan
    _, vs = ctx.blocks[k]
    p = next(iter(ctx.determined_params(k)), t)
    anc = ctx.ancestor_inputs(k)
    order = sorted(anc, key=lambda v: (_BLAME_ORDER.index(v) if v in _BLAME_ORDER else len(_BLAME_ORDER),
                                       ctx.rank.get(v, 1 << 30)))
    lo, hi, spacing = ctx.system.scan_range[p]
    for x in order:
        given = ctx.inputs.get(x, ctx.system.defaults.get(x))
        base = {kk: vv for kk, vv in ctx.inputs.items() if kk != x}

        def implied(pv, x=x, base=base):
            try:
                sol, _ = _solve(ctx.system, {**base, p: float(pv)}, None, ctx.cache, True)
            except (ValueError, NoSolutionError, ConvergenceError, AmbiguousSolutionError):
                return math.nan
            return sol[x]

        ps = _grid(lo, hi, spacing, N_SWEEP)
        ys = [implied(pv) for pv in ps]
        ok = [(y, pv, i) for i, (pv, y) in enumerate(zip(ps, ys)) if math.isfinite(y)]
        if len(ok) < 2:
            continue
        others = ", ".join(v for v in anc if v != x)
        ymax, ymin = max(ok), min(ok)
        def edge(i):
            return f", at the edge of the scanned range {p} = {lo:g}..{hi:g}" if i in (0, len(ps) - 1) else ""

        if given > ymax[0]:
            yb, pb = _refine(implied, ps, ymax[2], +1)
            return (f"{x} = {_fmt(x, given)} exceeds what {others} allow (max ≈ {_fmt(x, yb)} near "
                    f"{p} = {pb:.3g}{edge(ymax[2])}).")
        if given < ymin[0]:
            yb, pb = _refine(implied, ps, ymin[2], -1)
            return (f"{x} = {_fmt(x, given)} is below what {others} allow (min ≈ {_fmt(x, yb)} near "
                    f"{p} = {pb:.3g}{edge(ymin[2])}).")
    return None


def _refine(f, ps, i, sign):
    """Refine the extreme of f near sample i (sign +1 max, -1 min); falls back to the sample."""
    a, b = float(ps[max(i - 1, 0)]), float(ps[min(i + 1, len(ps) - 1)])
    best_y, best_p = f(ps[i]), float(ps[i])

    def obj(pv):
        y = f(pv)
        return -sign * y if math.isfinite(y) else math.inf

    try:
        m = minimize_scalar(obj, bounds=(a, b), method="bounded", options={"xatol": 1e-9 * abs(b)})
    except (ValueError, RuntimeError):                               # pragma: no cover - defensive
        return best_y, best_p
    if math.isfinite(m.fun) and sign * (-sign * m.fun) > sign * best_y:
        best_y, best_p = -sign * m.fun, float(m.x)
    return best_y, best_p


# --- Newton ------------------------------------------------------------------------------------------

_SEEDS = {
    "T": (300.0, 1500.0, 800.0, 3000.0, 500.0, 1000.0, 2000.0, 400.0),
    "P": (1e5, 1e7, 1e6, 3e6, 3e5, 3e7, 5e5, 2e6),
    "v": (1.0, 0.1, 0.5, 0.05, 2.0, 0.2, 0.02, 5.0),
}
_POS_GENERIC = (1.0, 3.0, 0.3, 10.0, 0.1, 30.0, 0.03, 100.0)
_SIGNED_FACTORS = (0.0, 1.0, -1.0, 5.0, -5.0, 10.0, 0.3, -0.3)


def _seed(ctx, v, i, j):
    kind = ctx.system.var_kind.get(v, "signed")
    if kind == "positive":
        table = _SEEDS.get(v[:1]) if v[:1] in _SEEDS and v not in ctx.params else None
        if table is None:     # mild symmetry break between variables, alternating direction per seed
            return _POS_GENERIC[i % N_SEEDS] * (1.5 if i % 2 == 0 else 1 / 1.5) ** j
        return table[i % N_SEEDS]
    if kind == "fraction":
        lo, hi, _ = ctx.system.scan_range.get(v, (0.0, 1.0, "asinh"))
        return lo + (hi - lo) * ((i + j) % N_SEEDS + 0.5) / N_SEEDS
    return default_scale(v) * _SIGNED_FACTORS[(i + j) % N_SEEDS]


def _newton(ctx, rels, vs, vals):
    pos = [ctx.system.var_kind.get(v) == "positive" for v in vs]
    sc = [1.0 if p else default_scale(v) for v, p in zip(vs, pos)]

    def to_x(y):
        return [math.exp(min(yi, 700.0)) if p else yi * s for yi, p, s in zip(y, pos, sc)]

    def F(y):
        local = dict(zip(vs, to_x(y)))
        cm = ChainMap(local, vals)
        try:
            out = [r.residual(*(cm[v] for v in r.vars)) / r.scale for r in rels]
        except _EVAL_ERRORS:
            return np.full(len(rels), 1e10)
        return np.array([o if math.isfinite(o) else 1e10 for o in out])

    sols: "list[dict[str, float]]" = []
    best = math.inf
    for i in range(N_SEEDS):
        x0 = [_seed(ctx, v, i, j) for j, v in enumerate(vs)]
        y0 = [math.log(x) if p else x / s for x, p, s in zip(x0, pos, sc)]
        try:
            res = root(F, np.array(y0, dtype=float), method="hybr", options={"xtol": 1e-14})
        except (ValueError, RuntimeError, FloatingPointError):     # pragma: no cover - defensive
            continue
        fv = F(res.x)
        norm = float(np.linalg.norm(fv))
        best = min(best, norm)
        if not np.all(np.abs(fv) <= NEWTON_TOL):
            continue
        sol = {v: float(x) for v, x in zip(vs, to_x(res.x))}
        if all(any(_distinct(sol[v], s[v]) for v in vs) for s in sols):
            sols.append(sol)
    if not sols:
        raise ConvergenceError(
            f"{ctx.name}: Newton did not converge for the block {list(vs)} (relations: "
            f"{', '.join(repr(r.name) for r in rels)}); best residual norm {best:.3g} after {N_SEEDS} seeds",
            variables=tuple(vs), residual=best)
    return sols
