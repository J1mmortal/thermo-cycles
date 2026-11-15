"""Which inputs a cycle still needs, or which ones conflict (spec 9.3, 9.4; plan Task 7).

``analyse(system, given)`` classifies an input set as ``ok``, ``under`` (missing inputs) or ``over``
(dependent inputs) and phrases the fix; ``raise_for`` turns a non-``ok`` result into
``UnderdeterminedError`` / ``OverdeterminedError`` with the message templates of spec 9.4.

Two layers decide whether a set of fixed variables is independent and complete:

1. **Structural** (spec 9.3): a maximum matching of the bipartite relation/variable graph. Exact for the
   structure, but blind to identities that hold for every numeric value. Two such identities matter for
   ideal-gas cycles: the Otto efficiency depends on r_c alone (so ``eta_th`` given with ``r_c`` is
   redundant, and the solver would return a zero-heat cycle), and temperature/ratio inputs never fix the
   pressure scale (so ``P1`` is not interchangeable with ``T1`` or ``r_c``).
2. **Numeric** (refinement, default): the rank of the system's Jacobian at two reference solutions of the
   cycle, plus one unit row per fixed variable. The reference solutions come from the catalogue's
   ``typical_inputs`` with generic values, and the Jacobian is built once per system by central
   differences. The rank is counted after row/column equilibration with a relative singular-value
   tolerance of ``RANK_RTOL``. Regular input sets show sigma_min/sigma_max >= 5e-4 and degenerate ones
   <= 5e-12 (Otto probe), so the tolerance sits far from both. The generic rank is the larger of the two
   reference ranks. When no reference solution can be found, the structural layer is used alone.

Both layers answer the same three questions, so the rest of the module does not care which one is used:
rank r of the rows, number of rows m, number of variables n. ``over`` if r < m (some fixed rows are
dependent), ``under`` if r < n, ``ok`` otherwise. Over-determination is reported first (spec 9.3).
"""

from __future__ import annotations

import itertools
import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..errors import OverdeterminedError, UnderdeterminedError
from ..model.cycle_def import state_ids, step_vars
from .structure import max_matching
from .system import EquationSystem, resolve_name

__all__ = ["Requirements", "analyse", "raise_for", "RANK_RTOL"]

RANK_RTOL = 1e-8          # relative singular-value tolerance of the numeric rank
MAX_EXAMPLES = 6          # example_sets kept for k >= 2 (spec 9.4)
MAX_LISTED = 10           # names shown per list in messages (spec 9.4: add_one_of[:10])


@dataclass(frozen=True)
class Requirements:
    """Result of ``analyse`` (spec 9.4). The first seven fields are the plan's; the rest feed the
    over-determined message.

    status: "ok", "under" or "over" ("over" wins when both apply, spec 9.3).
    given: the given inputs, canonical names, in the given order without duplicates.
    missing: number of inputs still missing (variables minus rank).
    add_one_of: candidates of which adding any one brings the deficit down (k = 1: completes the cycle).
    example_sets: completing sets of additional inputs (k >= 2).
    drop_one_of: given inputs whose removal takes away one dependency.
    determined: candidates the given inputs already fix; adding one would over-determine.
    conflicting: fixed variables (given or defaulted) that are not independent, in ranking order.
    conflict_groups: ``conflicting`` split into independent dependency groups (one per surplus row).
    n_dof: degrees of freedom of the conflicting part (len(conflicting) minus the surplus).
    method: "numeric" or "structural": which layer answered.
    """

    status: str
    given: "tuple[str, ...]"
    missing: int
    add_one_of: "tuple[str, ...]" = ()
    example_sets: "tuple[tuple[str, ...], ...]" = ()
    drop_one_of: "tuple[str, ...]" = ()
    determined: "tuple[str, ...]" = ()
    conflicting: "tuple[str, ...]" = ()
    conflict_groups: "tuple[tuple[str, ...], ...]" = ()
    n_dof: int = 0
    method: str = "structural"


# ---------------------------------------------------------------------------------------------------------
# Reference solutions for the numeric layer

# Generic, deliberately "uneven" values (two variants), away from the special points of the v1 cycles
# (r_c = 1, eta = 1, the w_net maximum, ...). Only the base relations have to hold at a reference point;
# the values need not be physical.
_REF_VALUES = {
    "T1": (300.0, 285.0), "P1": (1.0e5, 0.9e5), "v1": (0.86, 0.95),
    "r_c": (9.3, 7.1), "r_e": (14.7, 11.3), "r_p": (7.3, 11.6), "cutoff": (1.9, 2.4), "alpha": (1.45, 1.27),
    "eps": (0.77, 0.62), "dT_regen": (23.0, 41.0), "eta_c": (0.87, 0.81), "eta_t": (0.91, 0.86),
    "q_in": (1.1e6, 0.83e6), "q_out": (5.0e5, 4.1e5), "w_net": (4.1e5, 3.3e5), "eta_th": (0.45, 0.38),
    "mep": (6.0e5, 4.5e5), "bwr": (0.41, 0.47),
}
_REF_STATE = {"T": (2600.0, 3900.0), "P": (6.0e6, 7.5e6), "v": (0.05, 0.08)}


def _ref_value(system: EquationSystem, name: str, variant: int) -> float:
    if name in _REF_VALUES:
        return _REF_VALUES[name][variant]
    if name[:1] in _REF_STATE and name[1:] in state_ids(system.cycle):
        return _REF_STATE[name[:1]][variant]
    kind = system.var_kind.get(name, "signed")
    lo, hi, _ = system.scan_range[name]
    if kind == "positive":
        g = (lo * hi) ** 0.5 if lo > 0 else 2.0
        return g * (1.0 + 0.23 * variant)
    if kind == "fraction":
        return lo + (hi - lo) * (0.6 + 0.13 * variant)
    return 3.0e5 * (1.0 + 0.2 * variant)


def _fd_jacobian(system: EquationSystem, vals: "dict[str, float]") -> np.ndarray:
    """Central-difference Jacobian of the base relations in relative variables (column j times
    max(|x_j|, 1)), each row scaled to max |entry| = 1."""
    idx = {v: i for i, v in enumerate(system.variables)}
    J = np.zeros((len(system.relations), len(system.variables)))
    for i, rel in enumerate(system.relations):
        x = [vals[v] for v in rel.vars]
        for j, v in enumerate(rel.vars):
            h = 1e-6 * max(abs(x[j]), 1.0)
            xp, xm = list(x), list(x)
            xp[j] += h
            xm[j] -= h
            # column scale = magnitude of the variable: derivatives w.r.t. relative changes, so pressures
            # (1e5..1e7 Pa) and volumes (0.1 m3/kg) weigh alike; a unit input row stays a unit row
            J[i, idx[v]] += (float(rel.residual(*xp)) - float(rel.residual(*xm))) / (2.0 * h) \
                * max(abs(x[j]), 1.0)
    if not np.all(np.isfinite(J)):
        raise ValueError("non-finite Jacobian")
    norm = np.abs(J).max(axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return J / norm


def _numeric_rank(J: np.ndarray, cols: "Iterable[int]") -> int:
    cols = list(cols)
    E = np.zeros((len(cols), J.shape[1]))
    E[np.arange(len(cols)), cols] = 1.0
    M = np.vstack([J, E])
    cmax = np.abs(M).max(axis=0)
    cmax[cmax == 0.0] = 1.0
    s = np.linalg.svd(M / cmax, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(np.count_nonzero(s > RANK_RTOL * s[0]))


def _reference_sets(system: EquationSystem, structural: "_Engine") -> "list[tuple[str, ...]]":
    sets = []
    defn = system.cycle
    for t in (defn.typical_inputs if defn is not None else ()):
        if all(n in structural.idx for n in t):
            sets.append(tuple(t))
    greedy = structural.greedy(())                  # fallback for cycles without typical inputs
    if greedy is not None:
        sets.append(greedy)
    return list(dict.fromkeys(sets))


def _reference_jacobians(system: EquationSystem, structural: "_Engine") -> "list[np.ndarray]":
    from .numeric import solve_numeric             # local: numeric imports the system module too

    jacs = []
    n = len(system.variables)
    idx = {v: i for i, v in enumerate(system.variables)}
    for variant in (0, 1):
        for inputs in _reference_sets(system, structural):
            vals_in = {nm: _ref_value(system, nm, variant) for nm in inputs}
            for d in system.defaults:              # generic values for defaulted parameters too (eta = 1 is
                vals_in.setdefault(d, _ref_value(system, d, variant))   # a special point)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vals, _ = solve_numeric(system, vals_in)
                J = _fd_jacobian(system, vals)
            except Exception:                       # any failure: try the next set / variant
                continue
            # accept only a regular point: its own inputs must give full rank
            if _numeric_rank(J, [idx[k] for k in vals_in]) == n:
                jacs.append(J)
                break
    return jacs


# ---------------------------------------------------------------------------------------------------------
# Rank engine


class _Engine:
    """Rank of (base relations + one row per fixed variable), structural or numeric, with a cache."""

    def __init__(self, system: EquationSystem, numeric: bool):
        self.system = system
        self.n = len(system.variables)
        self.idx = {v: i for i, v in enumerate(system.variables)}
        self.base = [frozenset(self.idx[v] for v in r.vars) for r in system.relations]
        self.jacs: "list[np.ndarray]" = []
        self._cache: "dict[frozenset, int]" = {}
        pos = {c: i for i, c in enumerate(system.candidates)}
        nc = len(system.candidates)
        self.pos = {v: pos.get(v, nc + i) for i, v in enumerate(system.variables)}
        sids = state_ids(system.cycle) if system.cycle is not None else ()
        secondary = {f"{p}{s}" for p in "uhs" for s in sids}
        if system.cycle is not None:
            secondary |= {v for st in system.cycle.steps for v in step_vars(st)}
        secondary |= {"w_compression", "w_expansion"}
        self.secondary = secondary
        if numeric:
            self.jacs = _reference_jacobians(system, _Engine(system, numeric=False))
        self.method = "numeric" if self.jacs else "structural"

    # -- rank ------------------------------------------------------------------------------------------
    def fixed(self, given: "Iterable[str]") -> "tuple[str, ...]":
        g = tuple(given)
        return g + tuple(d for d in self.system.defaults if d not in g)

    def rank(self, fixed: "Iterable[str]") -> int:
        key = frozenset(fixed)
        r = self._cache.get(key)
        if r is None:
            cols = [self.idx[f] for f in key]
            if self.jacs:
                r = max(_numeric_rank(J, cols) for J in self.jacs)
            else:
                eq = self.base + [frozenset({c}) for c in cols]
                r = sum(v is not None for v in max_matching(eq, self.n).eq_to_var)
            self._cache[key] = r
        return r

    def rows(self, fixed) -> int:
        return len(self.base) + len(frozenset(fixed))

    def status(self, given: "Iterable[str]") -> str:
        f = self.fixed(given)
        r = self.rank(f)
        if r < self.rows(f):
            return "over"
        return "under" if r < self.n else "ok"

    def gain(self, given: "Iterable[str]", c: str) -> bool:
        """True if fixing ``c`` as well adds one independent row (it is neither fixed nor determined)."""
        f = self.fixed(given)
        return self.rank(f + (c,)) == self.rank(f) + 1

    # -- helpers ---------------------------------------------------------------------------------------
    def ranked(self, names: "Iterable[str]") -> "tuple[str, ...]":
        return tuple(sorted(dict.fromkeys(names), key=self.pos.__getitem__))

    def greedy(self, given: "tuple[str, ...]") -> "tuple[str, ...] | None":
        """Additional inputs that complete ``given``, picked in ranking order (primary candidates first)."""
        cur = tuple(given)
        if self.status(cur) == "over":
            return None
        cands = self.system.candidates
        for pool in ([c for c in cands if c not in self.secondary], [c for c in cands if c in self.secondary]):
            for c in pool:
                if self.status(cur) == "ok":
                    break
                if c not in cur and self.gain(cur, c):
                    cur += (c,)
        if self.status(cur) != "ok":
            return None
        return cur[len(given):]


_ENGINES: "dict[tuple[int, bool], tuple[weakref.ref, _Engine]]" = {}


def _engine(system: EquationSystem, numeric: bool) -> _Engine:
    key = (id(system), numeric)
    hit = _ENGINES.get(key)
    if hit is not None and hit[0]() is system:
        return hit[1]
    eng = _Engine(system, numeric)
    try:
        # the dict is bound as a default: at interpreter exit the module global may already be None
        ref = weakref.ref(system, lambda _r, key=key, engines=_ENGINES: engines.pop(key, None))
    except TypeError:                               # not weak-referenceable: do not cache
        return eng
    _ENGINES[key] = (ref, eng)
    return eng


# ---------------------------------------------------------------------------------------------------------
# analyse


def analyse(system: EquationSystem, given: "Iterable[str]", *, numeric: bool = True) -> Requirements:
    """Classify ``given`` and phrase the fix (spec 9.4).

    Names are resolved like inputs (exact, case-insensitive, alias; ``UnknownInputError`` otherwise).
    ``numeric=False`` uses the structural layer only (spec 9.3 as written); see the module docstring for
    why the numeric refinement is the default.
    """
    names: "list[str]" = []
    for g in given:
        c = resolve_name(system, g)
        if c not in names:
            names.append(c)
    G = tuple(names)
    eng = _engine(system, numeric)
    F = eng.fixed(G)
    r, m, n = eng.rank(F), eng.rows(F), eng.n
    missing = n - r
    gset = set(G)
    determined = eng.ranked(c for c in system.candidates if c not in gset and not eng.gain(G, c))

    if r < m:                                       # over-determined: report first
        surplus = m - r
        conflicting = eng.ranked(f for f in F if eng.rank(tuple(x for x in F if x != f)) == r)
        drop = tuple(c for c in conflicting if c in gset)
        groups = _conflict_groups(eng, F, conflicting, r) if surplus > 1 else (conflicting,)
        return Requirements("over", G, missing, (), (), drop, determined, conflicting, groups,
                            len(conflicting) - surplus, eng.method)

    if missing == 0:
        return Requirements("ok", G, 0, determined=determined, method=eng.method)

    open_ = [c for c in system.candidates if c not in gset and c not in determined]
    primary = [c for c in open_ if c not in eng.secondary]
    secondary = [c for c in open_ if c in eng.secondary]
    add: "tuple[str, ...]" = ()
    examples: "list[tuple[str, ...]]" = []

    if missing == 1:
        for pool in (primary, primary + secondary):
            add = tuple(c for c in pool if eng.status(G + (c,)) == "ok")
            if add:
                break
    elif missing == 2:
        for pool in (primary, primary + secondary):
            pool = list(eng.ranked(pool))
            pairs = [(a, b) for a, b in itertools.combinations(pool, 2) if eng.status(G + (a, b)) == "ok"]
            if pairs:
                pairs.sort(key=lambda p: (eng.pos[p[0]] + eng.pos[p[1]], eng.pos[p[0]]))
                add = eng.ranked(c for p in pairs for c in p)
                examples = pairs[:MAX_EXAMPLES]
                break
    else:
        for pool in (primary, secondary):
            add = tuple(c for c in pool if eng.gain(G, c))
            if add:
                break
        greedy = eng.greedy(G)
        if greedy is not None:
            examples.append(greedy)
        for t in (system.cycle.typical_inputs if system.cycle is not None else ()):
            if gset <= set(t) and all(x in eng.idx for x in t) and eng.status(t) == "ok":
                ex = tuple(x for x in t if x not in gset)
                if ex not in examples:
                    examples.append(ex)
        examples = examples[:MAX_EXAMPLES]

    return Requirements("under", G, missing, add, tuple(examples), (), determined, method=eng.method)


def _conflict_groups(eng: _Engine, F, conflicting, r) -> "tuple[tuple[str, ...], ...]":
    """Split the conflicting rows into dependency groups: a and b belong together when removing a makes b
    independent (union-find over that relation)."""
    parent = {c: c for c in conflicting}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for a in conflicting:
        Fa = tuple(x for x in F if x != a)
        ra = eng.rank(Fa)
        for b in conflicting:
            if b != a and eng.rank(tuple(x for x in Fa if x != b)) < ra:
                parent[find(b)] = find(a)
    groups: "dict[str, list[str]]" = {}
    for c in conflicting:
        groups.setdefault(find(c), []).append(c)
    return tuple(tuple(g) for g in groups.values())


# ---------------------------------------------------------------------------------------------------------
# raise_for


def _join(names, limit=None) -> str:
    names = list(names)
    if limit is not None and len(names) > limit:
        return ", ".join(names[:limit]) + f", ... ({len(names) - limit} more)"
    return ", ".join(names)


def raise_for(req: Requirements, cycle_name: str) -> None:
    """Raise ``UnderdeterminedError`` / ``OverdeterminedError`` (spec 9.4 templates) unless ``req`` is ok."""
    if req.status == "ok":
        return None
    if req.status == "over":
        nc = len(req.conflicting)
        groups = [tuple(c for c in g if c in req.drop_one_of) for g in req.conflict_groups]
        groups = [g for g in groups if g]
        if len(groups) > 1:
            remove = "Remove one of each group: " + ", ".join(f"({_join(g)})" for g in groups)
        else:
            remove = f"Remove one of: {_join(req.drop_one_of)}"
        msg = (f"{cycle_name} is over-determined: {_join(req.conflicting)} are not independent\n"
               f"  ({nc} inputs for {req.n_dof} degrees of freedom in that part of the cycle).\n"
               f"  {remove}  -- or pass allow_redundant=True if the values are consistent.")
        raise OverdeterminedError(msg, conflicting=req.conflicting, drop_one_of=req.drop_one_of)
    if req.status != "under":
        raise ValueError(f"unknown Requirements.status {req.status!r}")

    k = req.missing
    sec = _engine_secondary(req)
    short = [d for d in req.determined if d not in sec]
    given = _join(req.given) if req.given else "(none)"
    fix = f"  (these already fix: {_join(short, MAX_LISTED)})" if short else ""
    add = _join(req.add_one_of, MAX_LISTED) if req.add_one_of else "(no single input; see the full list)"
    lines = [f"{cycle_name} needs {k} more input{'s' if k != 1 else ''}.",
             f"  Given: {given}{fix}",
             f"  Add one of: {add}"]
    if k >= 2 and req.example_sets:
        ex = " or ".join(f"({_join(s)})" for s in req.example_sets[:2])
        lines.append(f"  For example: {ex}{' ...' if len(req.example_sets) > 2 else ''}")
    lines.append(f"  See thermo_cycles.requirements('{cycle_name}', ...) for the full list.")
    raise UnderdeterminedError("\n".join(lines), missing=k, given=req.given, add_one_of=req.add_one_of,
                               example_sets=req.example_sets, determined=req.determined)


def _engine_secondary(req: Requirements) -> "set[str]":
    """Names only suggested as a last resort (u, h, s of a state; step q and w): left out of the short
    "these already fix" list. Recognised by pattern, since ``raise_for`` gets no system."""
    out = set()
    for d in req.determined:
        if (d[:1] in "uhs" and d[1:2].isdigit()) or (d[:2] in ("q_", "w_") and d[2:3].isdigit()) \
                or d in ("w_compression", "w_expansion"):
            out.add(d)
    return out
