# tests/test_structure.py
import pytest
from thermo_cycles.solver.structure import max_matching, dm_partition, blt_blocks

F = frozenset

def test_square_chain_is_perfect_and_ordered():
    # e0: x0 ; e1: x0,x1 ; e2: x1,x2
    eqs = [F({0}), F({0, 1}), F({1, 2})]
    m = max_matching(eqs, 3)
    assert None not in m.eq_to_var and None not in m.var_to_eq
    blocks = blt_blocks(eqs, 3)
    assert [b.vars for b in blocks] == [(0,), (1,), (2,)]

def test_algebraic_loop_is_one_block():
    eqs = [F({0}), F({1, 2}), F({1, 2})]          # x1,x2 coupled
    blocks = blt_blocks(eqs, 3)
    assert [len(b.vars) for b in blocks] == [1, 2]

def test_under_determined_part():
    eqs = [F({0}), F({1, 2})]                      # 3 vars, 2 eqs
    dm = dm_partition(eqs, 3)
    assert dm.rank == 2 and dm.under_vars == F({1, 2}) and dm.over_eqs == F()

def test_over_determined_part_names_the_conflicting_equations():
    eqs = [F({0}), F({0}), F({1})]                 # e0, e1 both fix x0
    dm = dm_partition(eqs, 2)
    assert dm.over_eqs == F({0, 1}) and dm.over_vars == F({0}) and dm.under_vars == F()

def test_blt_requires_perfect_matching():
    with pytest.raises(ValueError):
        blt_blocks([F({0, 1})], 2)


# ---------------------------------------------------------------------------
# Additional tests (beyond the plan): small hand-checkable graphs, edge cases
# and brute-force property checks. Expected sets are derived by hand in the
# comment next to each case.
# ---------------------------------------------------------------------------
import itertools
import random

from thermo_cycles.solver.structure import Block, DMPartition, Matching


def _assert_valid_matching(eqs, n_vars, m):
    assert len(m.eq_to_var) == len(eqs) and len(m.var_to_eq) == n_vars
    for e, v in enumerate(m.eq_to_var):
        if v is not None:
            assert v in eqs[e] and m.var_to_eq[v] == e
    for v, e in enumerate(m.var_to_eq):
        if e is not None:
            assert m.eq_to_var[e] == v


def _brute_dm(eqs, n_vars):
    """Matching-independent characterisation of the DM sets, by enumerating
    every matching of the graph:
      rank       = size of the largest matching,
      under_vars = variables left unmatched by SOME maximum matching,
      over_eqs   = equations left unmatched by SOME maximum matching.
    (Standard result: a vertex is reachable by an alternating path from an
    unmatched vertex of its own side iff some maximum matching leaves it
    unmatched; Lovász & Plummer, Matching Theory, Gallai-Edmonds / DM.)"""
    n_eqs = len(eqs)
    edges = [(e, v) for e in range(n_eqs) for v in eqs[e]]
    rank = 0
    best = []
    for k in range(min(n_eqs, n_vars), -1, -1):
        for combo in itertools.combinations(edges, k):
            if len({e for e, _ in combo}) == k and len({v for _, v in combo}) == k:
                best.append(combo)
        if best:
            rank = k
            break
    under_vars, over_eqs = set(), set()
    for combo in best:
        under_vars |= set(range(n_vars)) - {v for _, v in combo}
        over_eqs |= set(range(n_eqs)) - {e for e, _ in combo}
    return rank, F(under_vars), F(over_eqs)


def test_matching_is_valid_and_maximum_on_hand_graph():
    # e0:{0,1} e1:{0}: greedy e0-x0 would block e1; the maximum matching is e0-x1, e1-x0
    eqs = [F({0, 1}), F({0})]
    m = max_matching(eqs, 2)
    assert isinstance(m, Matching)
    assert m.eq_to_var == (1, 0) and m.var_to_eq == (1, 0)


def test_well_determined_square_system_has_empty_dm_parts():
    eqs = [F({0}), F({0, 1}), F({1, 2})]
    dm = dm_partition(eqs, 3)
    assert isinstance(dm, DMPartition)
    assert dm == DMPartition(F(), F(), F(), F(), 3)


def test_under_and_over_in_the_same_system():
    # x0 fixed twice (e0, e1: over); e2 couples x1, x2 (one eq, two vars: under);
    # x3 well-determined by e3
    eqs = [F({0}), F({0}), F({1, 2}), F({3})]
    dm = dm_partition(eqs, 4)
    assert dm.rank == 3
    assert dm.over_eqs == F({0, 1}) and dm.over_vars == F({0})
    assert dm.under_vars == F({1, 2}) and dm.under_eqs == F({2})


def test_under_determined_chain_reaches_through_matched_edges():
    # e0:{0,1} e1:{1,2}: each of x0, x1, x2 is free in some maximum matching
    # (e0-x1,e1-x2 frees x0; e0-x0,e1-x2 frees x1; e0-x0,e1-x1 frees x2) -> all under
    eqs = [F({0, 1}), F({1, 2})]
    dm = dm_partition(eqs, 3)
    assert dm.rank == 2 and dm.under_vars == F({0, 1, 2}) and dm.under_eqs == F({0, 1})


def test_under_equation_may_use_well_determined_variables():
    # x0 fixed by e0 (well-determined); e1 links x0 with x1, x2 (under).
    eqs = [F({0}), F({0, 1, 2})]
    dm = dm_partition(eqs, 3)
    assert dm.under_vars == F({1, 2}) and dm.under_eqs == F({1})
    assert dm.over_eqs == F() and dm.over_vars == F()


def test_isolated_variable_is_under_determined():
    eqs = [F({0})]
    dm = dm_partition(eqs, 2)
    assert dm.under_vars == F({1}) and dm.under_eqs == F() and dm.rank == 1


def test_equation_without_variables_is_over_determined():
    eqs = [F({0}), F()]
    dm = dm_partition(eqs, 1)
    assert dm.over_eqs == F({1}) and dm.over_vars == F() and dm.rank == 1


def test_ideal_gas_degeneracy_is_over_determined():
    # Spec §9.3: T1 (x0) and u1 (x1) both given, plus u1 = cv*T1.
    # e0: fixed T1, e1: fixed u1, e2: u1 - cv*T1 = 0 -> 3 eqs on 2 vars, all over.
    eqs = [F({0}), F({1}), F({0, 1})]
    dm = dm_partition(eqs, 2)
    assert dm.rank == 2 and dm.over_eqs == F({0, 1, 2}) and dm.over_vars == F({0, 1})
    assert dm.under_vars == F() and dm.under_eqs == F()


def test_empty_system():
    assert max_matching([], 0) == Matching((), ())
    assert dm_partition([], 0) == DMPartition(F(), F(), F(), F(), 0)
    assert blt_blocks([], 0) == []


def test_out_of_range_variable_raises():
    with pytest.raises(ValueError):
        max_matching([F({2})], 2)
    with pytest.raises(ValueError):
        dm_partition([F({-1})], 2)
    with pytest.raises(ValueError):
        max_matching([F({0})], -1)


def test_blt_rejects_square_but_singular_system():
    # 2x2 but both equations only touch x0 -> no perfect matching
    with pytest.raises(ValueError):
        blt_blocks([F({0}), F({0})], 2)


def test_blt_rejects_non_square_with_full_row_rank():
    with pytest.raises(ValueError):
        blt_blocks([F({0}), F({1})], 3)


def test_blt_orders_reversed_chain():
    # e0 uses x0,x1 ; e1 uses x1,x2 ; e2 uses x2 -> e2 (x2), then e1 (x1), then e0 (x0)
    eqs = [F({0, 1}), F({1, 2}), F({2})]
    blocks = blt_blocks(eqs, 3)
    assert blocks == [Block((2,), (2,)), Block((1,), (1,)), Block((0,), (0,))]


def test_blt_loop_between_two_singletons():
    # x0 from e0, then x1,x2 coupled via e1,e2, then x3 from e3 (uses x2 and x0)
    eqs = [F({0}), F({0, 1, 2}), F({1, 2}), F({2, 3, 0})]
    blocks = blt_blocks(eqs, 4)
    assert [set(b.eqs) for b in blocks] == [{0}, {1, 2}, {3}]
    assert [set(b.vars) for b in blocks] == [{0}, {1, 2}, {3}]


def test_blt_whole_ring_is_one_block():
    # e0:{0,2} e1:{0,1} e2:{1,2}: every equation needs another's variable
    eqs = [F({0, 2}), F({0, 1}), F({1, 2})]
    blocks = blt_blocks(eqs, 3)
    assert len(blocks) == 1 and set(blocks[0].vars) == {0, 1, 2}


def test_blt_block_vars_are_paired_with_eqs():
    eqs = [F({0, 1}), F({1, 2}), F({2}), F({3, 4}), F({3, 4})]
    m = max_matching(eqs, 5)
    for b in blt_blocks(eqs, 5):
        assert b.eqs == tuple(sorted(b.eqs))
        assert tuple(m.eq_to_var[e] for e in b.eqs) == b.vars


def test_long_graphs_do_not_hit_recursion_limit():
    n = 3000
    chain = [F({0})] + [F({i - 1, i}) for i in range(1, n)]
    assert [b.vars for b in blt_blocks(chain, n)] == [(i,) for i in range(n)]
    ring = [F({i, (i + 1) % n}) for i in range(n)]
    assert len(blt_blocks(ring, n)) == 1
    # long alternating augmenting path: e_i:{i, i+1}, e_last:{0}; DM on n+1 vars
    path = [F({i, i + 1}) for i in range(n)]
    dm = dm_partition(path, n + 1)
    assert dm.rank == n and dm.under_vars == F(range(n + 1))


def _random_graph(rng, n_eqs, n_vars, p):
    return [F(v for v in range(n_vars) if rng.random() < p) for _ in range(n_eqs)]


@pytest.mark.parametrize("seed", range(300))
def test_matching_and_dm_against_brute_force(seed):
    rng = random.Random(seed)
    n_eqs, n_vars = rng.randint(0, 6), rng.randint(0, 6)
    eqs = _random_graph(rng, n_eqs, n_vars, rng.choice([0.2, 0.35, 0.5]))
    m = max_matching(eqs, n_vars)
    _assert_valid_matching(eqs, n_vars, m)
    rank, under_vars, over_eqs = _brute_dm(eqs, n_vars)
    assert sum(v is not None for v in m.eq_to_var) == rank
    dm = dm_partition(eqs, n_vars)
    assert dm.rank == rank
    assert dm.under_vars == under_vars
    assert dm.over_eqs == over_eqs
    # Structural identities of the coarse DM decomposition
    assert not (dm.under_vars & dm.over_vars) and not (dm.under_eqs & dm.over_eqs)
    assert F().union(*(eqs[e] for e in dm.over_eqs)) == dm.over_vars   # over eqs touch only over vars
    for v in dm.under_vars:                                             # under vars appear only in under eqs
        assert all(e in dm.under_eqs for e in range(n_eqs) if v in eqs[e])
    assert len(dm.over_eqs) - len(dm.over_vars) == n_eqs - rank
    assert len(dm.under_vars) - len(dm.under_eqs) == n_vars - rank
    # The remaining (well-determined) part is square and perfectly matchable
    sq_eqs = [e for e in range(n_eqs) if e not in dm.under_eqs | dm.over_eqs]
    sq_vars = sorted(set(range(n_vars)) - dm.under_vars - dm.over_vars)
    idx = {v: i for i, v in enumerate(sq_vars)}
    sub = [F(idx[v] for v in eqs[e] if v in idx) for e in sq_eqs]
    assert len(sq_eqs) == len(sq_vars)
    assert dm_partition(sub, len(sq_vars)).rank == len(sq_vars)


@pytest.mark.parametrize("seed", range(300))
def test_blt_against_definition(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 9)
    # guarantee a perfect matching via a random permutation, then add extra edges
    perm = list(range(n))
    rng.shuffle(perm)
    eqs = [F({perm[i]} | {v for v in range(n) if rng.random() < 0.25}) for i in range(n)]
    blocks = blt_blocks(eqs, n)
    assert sorted(e for b in blocks for e in b.eqs) == list(range(n))
    assert sorted(v for b in blocks for v in b.vars) == list(range(n))
    known = set()
    for b in blocks:
        assert len(b.eqs) == len(b.vars)
        for e in b.eqs:                       # a block uses only its own and earlier variables
            assert eqs[e] <= known | set(b.vars)
        known |= set(b.vars)
    # blocks are irreducible: inside a block every equation reaches every other
    # through the matched dependency graph (eq -> eq that owns a variable it uses)
    m = max_matching(eqs, n)
    for b in blocks:
        inside = set(b.eqs)
        for start in b.eqs:
            seen, stack = {start}, [start]
            while stack:
                e = stack.pop()
                for v in eqs[e]:
                    nxt = m.var_to_eq[v]
                    if nxt in inside and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            assert seen == inside
