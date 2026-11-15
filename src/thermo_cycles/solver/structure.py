"""Structural analysis of an equation system (pure graph code, spec §9.3 and §9.5).

The system is a bipartite graph: equation ``e`` is linked to every variable in
``eq_vars[e]``. Equations and variables are plain integers ``0..n-1``; nothing
here knows about thermodynamics.

- :func:`max_matching`: maximum matching (Hopcroft & Karp 1973), giving the
  structural rank.
- :func:`dm_partition`: coarse Dulmage-Mendelsohn partition (Dulmage &
  Mendelsohn 1958) into the under-determined part (reachable by alternating
  paths from unmatched variables), the over-determined part (reachable from
  unmatched equations), and implicitly the well-determined rest.
- :func:`blt_blocks`: block lower-triangular ordering of a structurally
  non-singular square system: orient each equation towards its matched
  variable and take the strongly connected components (Tarjan 1972) in solve
  order.

All three are iterative (no recursion), so graph size is not limited by
Python's recursion limit.
"""

from __future__ import annotations

import operator
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "Matching",
    "max_matching",
    "DMPartition",
    "dm_partition",
    "Block",
    "blt_blocks",
]


@dataclass(frozen=True)
class Matching:
    """A matching between equations and variables.

    ``eq_to_var[e]`` is the variable matched to equation ``e`` (or None);
    ``var_to_eq[v]`` is the equation matched to variable ``v`` (or None).
    """

    eq_to_var: tuple[Optional[int], ...]
    var_to_eq: tuple[Optional[int], ...]


@dataclass(frozen=True)
class DMPartition:
    """Coarse Dulmage-Mendelsohn partition (spec §9.3).

    under_vars / under_eqs: variables reachable by alternating paths from
        unmatched variables, and the equations on those paths. These are the
        free unknowns and the equations that cannot pin them down.
    over_vars / over_eqs: equations reachable by alternating paths from
        unmatched equations (the surplus equations), and the variables on
        those paths.
    rank: structural rank = size of a maximum matching.

    Everything not in either part is the well-determined (square) part. The
    sets do not depend on which maximum matching is used.
    """

    under_vars: frozenset[int]
    under_eqs: frozenset[int]
    over_vars: frozenset[int]
    over_eqs: frozenset[int]
    rank: int


@dataclass(frozen=True)
class Block:
    """One diagonal block of the BLT form.

    ``eqs`` is sorted ascending; ``vars[k]`` is the variable matched to
    ``eqs[k]`` (so the pairing equation -> output variable is kept).
    """

    eqs: tuple[int, ...]
    vars: tuple[int, ...]


def _adjacency(eq_vars: Sequence[frozenset[int]], n_vars: int) -> list[tuple[int, ...]]:
    """Validate the graph and return each equation's variables, sorted (for determinism)."""
    n_vars = operator.index(n_vars)
    if n_vars < 0:
        raise ValueError(f"n_vars must be >= 0, got {n_vars}")
    adj = []
    for e, vs in enumerate(eq_vars):
        row = sorted({operator.index(v) for v in vs})
        if row and (row[0] < 0 or row[-1] >= n_vars):
            bad = row[0] if row[0] < 0 else row[-1]
            raise ValueError(
                f"equation {e} uses variable {bad}, outside the range 0..{n_vars - 1}"
            )
        adj.append(tuple(row))
    return adj


def _hopcroft_karp(adj: list[tuple[int, ...]], n_vars: int) -> tuple[list, list]:
    n_eqs = len(adj)
    eq_to: list[Optional[int]] = [None] * n_eqs
    var_to: list[Optional[int]] = [None] * n_vars
    inf = n_eqs + 1

    while True:
        # BFS: layer the equations by alternating distance from the free equations.
        dist = [inf] * n_eqs
        queue: deque[int] = deque()
        for e in range(n_eqs):
            if eq_to[e] is None:
                dist[e] = 0
                queue.append(e)
        found = False
        while queue:
            e = queue.popleft()
            for v in adj[e]:
                e2 = var_to[v]
                if e2 is None:
                    found = True
                elif dist[e2] == inf:
                    dist[e2] = dist[e] + 1
                    queue.append(e2)
        if not found:
            break

        # DFS (iterative): vertex-disjoint shortest augmenting paths along the layers.
        nxt = [0] * n_eqs  # per-equation edge pointer, persists through the phase
        for root in range(n_eqs):
            if eq_to[root] is not None or dist[root] != 0:
                continue
            stack = [root]
            while stack:
                e = stack[-1]
                if nxt[e] < len(adj[e]):
                    v = adj[e][nxt[e]]
                    nxt[e] += 1
                    e2 = var_to[v]
                    if e2 is None:
                        # Augment: every equation on the stack takes the variable
                        # it last stepped through.
                        for eq in stack:
                            vv = adj[eq][nxt[eq] - 1]
                            eq_to[eq] = vv
                            var_to[vv] = eq
                        break
                    if dist[e2] == dist[e] + 1:
                        stack.append(e2)
                else:
                    dist[e] = inf  # dead end for the rest of this phase
                    stack.pop()
    return eq_to, var_to


def max_matching(eq_vars: Sequence[frozenset[int]], n_vars: int) -> Matching:
    """Maximum bipartite matching (Hopcroft-Karp, O(E*sqrt(V)))."""
    adj = _adjacency(eq_vars, n_vars)
    eq_to, var_to = _hopcroft_karp(adj, operator.index(n_vars))
    return Matching(tuple(eq_to), tuple(var_to))


def dm_partition(eq_vars: Sequence[frozenset[int]], n_vars: int) -> DMPartition:
    """Coarse Dulmage-Mendelsohn partition from a maximum matching (spec §9.3)."""
    adj = _adjacency(eq_vars, n_vars)
    n_vars = operator.index(n_vars)
    eq_to, var_to = _hopcroft_karp(adj, n_vars)
    var_eqs: list[list[int]] = [[] for _ in range(n_vars)]
    for e, row in enumerate(adj):
        for v in row:
            var_eqs[v].append(e)

    # Under-determined: from unmatched variables, var -(unmatched edge)-> eq -(matched edge)-> var.
    under_vars = {v for v in range(n_vars) if var_to[v] is None}
    under_eqs: set[int] = set()
    queue = deque(under_vars)
    while queue:
        v = queue.popleft()
        for e in var_eqs[v]:
            if e in under_eqs:
                continue
            under_eqs.add(e)
            w = eq_to[e]  # never None for a maximum matching (else an augmenting path exists)
            if w not in under_vars:
                under_vars.add(w)
                queue.append(w)

    # Over-determined: from unmatched equations, eq -(unmatched edge)-> var -(matched edge)-> eq.
    over_eqs = {e for e in range(len(adj)) if eq_to[e] is None}
    over_vars: set[int] = set()
    queue = deque(over_eqs)
    while queue:
        e = queue.popleft()
        for v in adj[e]:
            if v in over_vars:
                continue
            over_vars.add(v)
            f = var_to[v]  # never None for a maximum matching
            if f not in over_eqs:
                over_eqs.add(f)
                queue.append(f)

    rank = sum(v is not None for v in eq_to)
    return DMPartition(
        frozenset(under_vars), frozenset(under_eqs),
        frozenset(over_vars), frozenset(over_eqs), rank,
    )


def blt_blocks(eq_vars: Sequence[frozenset[int]], n_vars: int) -> list[Block]:
    """Block lower-triangular ordering of a square, structurally non-singular system.

    Returns the blocks in solve order: the variables of a block are computed
    from its own equations given the variables of earlier blocks only.
    Raises ValueError if there is no perfect matching.
    """
    adj = _adjacency(eq_vars, n_vars)
    n_vars = operator.index(n_vars)
    n = len(adj)
    eq_to, var_to = _hopcroft_karp(adj, n_vars)
    rank = sum(v is not None for v in eq_to)
    if n != n_vars or rank != n:
        raise ValueError(
            f"BLT ordering needs a perfect matching: {n} equations, {n_vars} variables, "
            f"structural rank {rank}"
        )

    # Equation e depends on the equation that computes each other variable it uses.
    deps = [tuple(var_to[v] for v in adj[e] if v != eq_to[e]) for e in range(n)]

    # Iterative Tarjan. An SCC is emitted only after every SCC it depends on,
    # so the emission order is the solve order.
    index: list[Optional[int]] = [None] * n
    low = [0] * n
    on_stack = [False] * n
    scc_stack: list[int] = []
    counter = 0
    blocks: list[Block] = []
    for root in range(n):
        if index[root] is not None:
            continue
        index[root] = low[root] = counter
        counter += 1
        scc_stack.append(root)
        on_stack[root] = True
        work = [(root, iter(deps[root]))]
        while work:
            node, it = work[-1]
            descended = False
            for w in it:
                if index[w] is None:
                    index[w] = low[w] = counter
                    counter += 1
                    scc_stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(deps[w])))
                    descended = True
                    break
                if on_stack[w]:
                    low[node] = min(low[node], index[w])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                members = []
                while True:
                    w = scc_stack.pop()
                    on_stack[w] = False
                    members.append(w)
                    if w == node:
                        break
                eqs = tuple(sorted(members))
                blocks.append(Block(eqs, tuple(eq_to[e] for e in eqs)))
    return blocks
