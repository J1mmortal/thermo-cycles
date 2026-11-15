# thermo-cycles v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `J1mmortal/thermo-cycles` into the importable library `thermo_cycles`. A cycle is picked by name from
a data catalogue, a general structural solver accepts any well-posed input set, and the library reports states,
steps, results, margins, optimisation and T-s / P-v plots.

**Architecture:** Each cycle definition (data) is compiled into a system of scalar relations: fluid equations per
state, one relation per process step, parameter definitions, and one per input. Bipartite matching
(Dulmage-Mendelsohn) decides whether the inputs are under-, over- or well-determined and phrases the fix. BLT
ordering (Tarjan) then solves block by block: explicit inverse, a torn 1-D root scan, or Newton. Results, margins,
optimisation and plotting all work from the solved `CycleState`.

**Tech Stack:** Python ≥3.10, numpy, scipy (core); matplotlib (optional `[plot]`); pytest (`[dev]`); setuptools
`src/` layout, in the same style as rocket-analysis-toolkit.

**Spec:** `thermo-cycles-spec.md` (same folder; section numbers below, like §9.4, refer to it). Executors read both.

## Global Constraints

- Workspace: `git clone https://github.com/J1mmortal/thermo-cycles` into the executing task folder, then
  `git checkout -b feat/library-v1 main`. Commit locally after every task. **Never push**, open PRs or touch
  GitHub.
- Environment: `python -m venv .venv`, activate it, run `pip install -e ".[plot,dev]"`. Run tests with
  `python -m pytest`.
- Distribution `thermo-cycles`, import `thermo_cycles`, version `0.1.0`, `requires-python = ">=3.10"`, licence MIT
  (keep the existing LICENSE).
- Core dependencies are exactly `numpy>=1.23` and `scipy>=1.10`. matplotlib is imported only inside
  `plotting.py` functions. Do not declare a `[coolprop]` extra.
- SI units, per unit mass, everywhere (§3). Signs: q > 0 into the fluid, w > 0 done by the fluid; closed steps
  q − w = Δu, flow steps q − w = Δh.
- Ideal-gas reference: u = cv·T, h = cp·T, s = cp·ln(T/273.15) − R·ln(P/1e5) (§3, kept from old code).
- `AIR = IdealGas(R=287.0, gamma=1.4)`.
- Names: `r_c` compression ratio, `r_e` expansion ratio, `cutoff`, `alpha` (dual constant-volume pressure ratio),
  `r_p`, `eta_c`, `eta_t`, `eps`, `dT_regen`; state props `T1`, `P2`, `v3`, `u4`, `h2`, `s1`; hidden `T2s`;
  step `q_23`, `w_12`; cycle `q_in`, `q_out`, `w_net`, `eta_th`, `mep`, `bwr`.
- Error and warning class names and hierarchy exactly as spec §14. Message templates exactly as spec §9.4.
- Physical validity checks per §9.7; input validation per §9.8.
- Reference-case tolerances: rtol 3e-3 unless the test states otherwise. Closed-form identities: rtol 1e-9.

## Review Focus

1. **Non-SI magnitudes** (`P1=100` meaning kPa, `T1=25` meaning °C): the user expects a warning, not a silent
   near-vacuum cycle → `InputPlausibilityWarning` (test in Task 7).
2. **NaN, inf, negative or non-numeric inputs, numpy scalars and ints**: bad values raise `InputError` naming the
   input; `np.float64(300)` and `300` work (test in Task 7).
3. **Heat addition to a temperature below the compression end** (T3 < T2): the user expects an error naming
   step 2→3, not negative efficiencies (test in Task 8).
4. **Old-style names** (`R_C`, `t1`, `r`): resolved through case-insensitive and alias lookup; `rc` gets a
   "did you mean r_c / cutoff" suggestion (test in Task 7).
5. **Regenerator that would heat the hot stream** (high r_p, so T5 < T2): the user expects `InvalidCycleError`
   naming the regenerator, not a negative effectiveness (test in Task 10).

## Task order, models, dependencies

| # | task | model | depends on |
|---|---|---|---|
| 1 | Package skeleton, migration, errors, units | sonnet | – |
| 2 | State and IdealGas | sonnet | 1 |
| 3 | Relations and step kinds | sonnet | 2 |
| 4 | Structural graph algorithms | opus | 1 |
| 5 | Catalogue format, registry, Otto, build_system | opus | 3 |
| 6 | Numeric block solver | opus | 4, 5 |
| 7 | Input resolution and advice (requirements, errors) | opus | 6 |
| 8 | Results, CycleState, solve_cycle, validity, Cycle front | opus | 7 |
| 9 | Diesel, Dual, Atkinson + references + closed forms + custom cycle + regression | sonnet | 8 |
| 10 | Brayton and Brayton with regeneration | opus | 8 |
| 11 | Margins | sonnet | 8 |
| 12 | Optimisation | opus | 11 |
| 13 | Plotting | sonnet | 8 |
| 14 | describe(), README, examples, full suite | sonnet | 9-13 |

Tasks 9, 10, 11 and 13 are independent of each other once Task 8 is done.

---

### Task 1: Package skeleton, migration, errors, units

**Files:**
- Move: `src/thermo-cycles/` → `src/thermo_cycles/` (`git mv`)
- Delete: `src/thermo_cycles/core/` (all), `src/thermo_cycles/cycles/` (all), `src/thermo_cycles/plotting/`
- Create: `pyproject.toml`, `src/thermo_cycles/__init__.py`, `src/thermo_cycles/errors.py`, `src/thermo_cycles/units.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: `thermo_cycles.__version__ == "0.1.0"`; all classes of spec §14 in `thermo_cycles.errors`;
  `thermo_cycles.units`: `K=1.0, Pa=1.0, kPa=1e3, bar=1e5, MPa=1e6, atm=101325.0, J=1.0, kJ=1e3, MJ=1e6, L=1e-3`,
  `degC(t: float) -> float` (t + 273.15).
- Error constructors take keyword attributes (e.g. `UnderdeterminedError(message, missing=1, given=(...),
  add_one_of=(...), example_sets=(...), determined=(...))`) and store them as attributes.

- [ ] **Step 1: Set up the workspace and environment** (Global Constraints). Run the old smoke check to record the
  starting point: `python -c "import sys; sys.path.insert(0,'src/thermo-cycles'); import cycles"`. It should
  import, which records that the old layout works only through path hacks.
- [ ] **Step 2: Write the failing test**

```python
# tests/test_package.py
import importlib, sys
import pytest

def test_import_and_version():
    tc = importlib.import_module("thermo_cycles")
    assert tc.__version__ == "0.1.0"

def test_error_hierarchy():
    from thermo_cycles import errors as E
    assert issubclass(E.UnderdeterminedError, E.InputError)
    assert issubclass(E.InputError, ValueError) and issubclass(E.InputError, E.ThermoCyclesError)
    assert issubclass(E.FluidError, ValueError)
    assert issubclass(E.MissingDependencyError, ImportError)
    for w in (E.LimitViolationWarning, E.CycleValidityWarning, E.InputPlausibilityWarning):
        assert issubclass(w, UserWarning)
    e = E.UnderdeterminedError("msg", missing=1, given=("T1",), add_one_of=("T3",), example_sets=(), determined=())
    assert e.missing == 1 and e.add_one_of == ("T3",) and str(e) == "msg"

def test_units():
    from thermo_cycles.units import kPa, bar, MPa, atm, kJ, L, degC
    assert (kPa, bar, MPa, atm, kJ, L) == (1e3, 1e5, 1e6, 101325.0, 1e3, 1e-3)
    assert degC(25) == pytest.approx(298.15)

def test_no_old_modules_and_no_matplotlib_import():
    import thermo_cycles  # noqa: F401
    assert "matplotlib" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("thermo_cycles.core")
```

- [ ] **Step 3: Run the test to verify it fails.** Run `python -m pytest tests/test_package.py -q`.
  Expected: FAIL (`ModuleNotFoundError: thermo_cycles`).
- [ ] **Step 4: Implement.**
  - Write `pyproject.toml`, copied from rocket-analysis-toolkit's structure: setuptools ≥61; `package-dir`
    `src`; `[project]` name `thermo-cycles`, version 0.1.0, description "Thermodynamic cycle calculator: pick a
    cycle, give what you know, get every state", readme, requires-python ≥3.10, licence MIT, the same authors
    entry as rocket-analysis-toolkit, dependencies numpy and scipy; `[project.optional-dependencies]`
    `plot = ["matplotlib>=3.7"]`, `dev = ["pytest>=7"]`; `[tool.pytest.ini_options] testpaths = ["tests"]`;
    Homepage URL.
  - Make `__init__.py` set only `__version__` for now.
  - `git rm` the old `core/`, `cycles/` and `plotting/` (their content lives on per spec §16).
- [ ] **Step 5: Run the test to verify it passes.** Run `python -m pytest -q`. Expected: 4 passed.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "build: importable thermo_cycles package, errors, units; remove old modules"`

---

### Task 2: State and IdealGas

**Files:**
- Create: `src/thermo_cycles/state.py`, `src/thermo_cycles/fluids/__init__.py`, `src/thermo_cycles/fluids/base.py`, `src/thermo_cycles/fluids/ideal_gas.py`
- Test: `tests/test_state.py`, `tests/test_ideal_gas.py`

**Interfaces:**
- Produces: `PROPS = ("T","P","v","u","h","s")`; `State` exactly as spec §5.1; `Fluid` Protocol and `FluidBase`
  (§6.1). `FluidBase.relations(sid) -> None` and `FluidBase.state()` with the unsupported-pair fallback via
  `scipy.optimize.brentq` (a hook for later fluids; IdealGas overrides `state`).
- `IdealGas(*, R=None, cp=None, cv=None, gamma=None, name="ideal gas", T_ref=273.15, P_ref=1e5)` with attributes
  `R, cp, cv, gamma, T_ref, P_ref, name, props`. `state(**two) -> dict[str, float]` with all 6 props;
  `independent(a, b) -> bool`; `relations(sid: str) -> list[Relation]` is added in Task 3 (it needs Relation).
- `AIR`, exported from `thermo_cycles.fluids` and `thermo_cycles`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ideal_gas.py
import itertools, math
import pytest
from thermo_cycles.fluids import IdealGas, AIR
from thermo_cycles.errors import FluidError, DegenerateStateError

def test_air_constants():
    assert (AIR.R, AIR.gamma) == (287.0, 1.4)
    assert AIR.cp == pytest.approx(1004.5) and AIR.cv == pytest.approx(717.5)

@pytest.mark.parametrize("kw,expect", [
    (dict(cp=920, gamma=1.22), dict(cv=920/1.22, R=920-920/1.22)),       # Rajput Ex 13.31
    (dict(R=300, cv=700), dict(cp=1000, gamma=1000/700)),                 # Cengel 9-24
    (dict(cp=1004.5, cv=717.5, R=287.0), dict(gamma=1.4)),                # 3 consistent values
])
def test_constructor_any_two(kw, expect):
    g = IdealGas(**kw)
    for k, v in expect.items():
        assert getattr(g, k) == pytest.approx(v, rel=1e-9)

def test_inconsistent_and_underspecified():
    with pytest.raises(FluidError):
        IdealGas(cp=1005, cv=718, R=280)
    with pytest.raises(FluidError):
        IdealGas(R=287)

def test_old_code_state1_values():
    st = AIR.state(T=298.0, P=1e5)        # old code: V=0.85526, S=87.464217
    assert st["v"] == pytest.approx(0.85526, rel=1e-6)
    assert st["s"] == pytest.approx(87.464217, rel=1e-7)
    assert st["u"] == pytest.approx(717.5 * 298) and st["h"] == pytest.approx(1004.5 * 298)

INDEPENDENT = [p for p in itertools.combinations(("T","P","v","u","h","s"), 2)
               if set(p) not in ({"T","u"}, {"T","h"}, {"u","h"})]

@pytest.mark.parametrize("a,b", INDEPENDENT)
def test_round_trip_all_pairs(a, b):
    ref = AIR.state(T=1234.5, P=3.21e6)
    got = AIR.state(**{a: ref[a], b: ref[b]})
    for k in ref:
        assert got[k] == pytest.approx(ref[k], rel=1e-10, abs=1e-9)

@pytest.mark.parametrize("a,b", [("T","u"), ("T","h"), ("u","h")])
def test_degenerate_pairs(a, b):
    ref = AIR.state(T=500, P=1e5)
    assert AIR.independent(a, b) is False
    with pytest.raises(DegenerateStateError):
        AIR.state(**{a: ref[a], b: ref[b]})
```

```python
# tests/test_state.py
import dataclasses, pytest
from thermo_cycles.state import State, PROPS

def test_state_frozen_and_accessors():
    s = State(id=3, T=2400.0, P=6.5e6, v=0.1, u=1.7e6, h=2.4e6, s=1000.0)
    assert PROPS == ("T", "P", "v", "u", "h", "s")
    assert s.get("t") == 2400.0 and s.has("T", "P") and not s.has("x_quality")
    assert s.as_dict()["P"] == 6.5e6 and s.x is None
    assert "State#3" in s.summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.T = 1.0
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_state.py tests/test_ideal_gas.py -q`. Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `State` (frozen dataclass; `get` resolves names case-insensitively against `PROPS`
  plus `x`; `has` returns False for unknown names). Implement `IdealGas`:
  - The two given constants determine the rest. With 3 or 4 given, check consistency to rtol 1e-6. Fewer than 2
    given (none given means AIR's values) raises `FluidError`.
  - `state()` uses closed forms for all 12 independent pairs. Compute T and P first from the pair, then
    everything else. For example (v, s) gives T = T_ref·exp((s + R·ln(R·T_ref/(P_ref·v)))/cv), and (u, *) or
    (h, *) give T directly.
  - Pairs from `independent()` == False raise `DegenerateStateError`. Fewer or more than 2 kwargs, or an unknown
    name, raise `FluidError`.
  - Use `math`, not numpy.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: all passed.
- [ ] **Step 5: Commit.** `git commit -am "feat: State and IdealGas fluid (any two of R, cp, cv, gamma; all property pairs)"` (after `git add` of new files)

---

### Task 3: Relations and step kinds

**Files:**
- Create: `src/thermo_cycles/model/__init__.py`, `src/thermo_cycles/model/relations.py`, `src/thermo_cycles/model/steps.py`
- Modify: `src/thermo_cycles/fluids/ideal_gas.py` (add `relations(sid)`)
- Test: `tests/test_relations.py`, `tests/test_steps.py`

**Interfaces:**
- Consumes: `IdealGas`, `State`, `PROPS`.
- Produces: `Relation(name, vars, residual, solvers, scale=1.0)` and the helpers `equal(a, b, name)`,
  `ratio(out, num, den, name=None)` (out·den − num = 0), `linear(out, terms, name=None)`,
  `product(out, a, b, name=None)` and `fixed(var, value)`, exactly as spec §9.2. `Relation.solve_for(var, values:
  Mapping[str, float]) -> float` calls `solvers[var]` and raises `KeyError` if there is no inverse.
- `IdealGas.relations(sid: str) -> list[Relation]`: 4 relations over `T{sid} P{sid} v{sid} u{sid} h{sid} s{sid}`,
  named `"eos {sid}"`, `"u(T) {sid}"`, `"h(T) {sid}"`, `"s(T,P) {sid}"`, each with explicit inverses for every
  variable it uses.
- `StepKind` with `name`, `reversible`, `relations(a, b, *, system, role, efficiency) -> list[Relation]`,
  `hidden_states(b) -> tuple[str, ...]` and `path(fluid, sa, sb, n=100) -> list[State] | None`.
  `KINDS: dict[str, StepKind]` with keys `isentropic, isochoric, isobaric, isothermal, adiabatic`. Relations per
  spec §7.1 table. q and w relations define variables `q_{a}{b}` and `w_{a}{b}`. `adiabatic` hidden state id is
  `f"{b}s"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_relations.py
import pytest
from thermo_cycles.model.relations import ratio, linear, equal, product, fixed
from thermo_cycles.fluids import AIR

def test_ratio_all_inverses():
    r = ratio("r_c", "v1", "v2")
    vals = {"r_c": 10.0, "v1": 0.8, "v2": 0.08}
    assert r.residual(*(vals[v] for v in r.vars)) == pytest.approx(0, abs=1e-12)
    for var in r.vars:
        others = {k: v for k, v in vals.items() if k != var}
        assert r.solve_for(var, others) == pytest.approx(vals[var])

def test_linear_and_fixed():
    l = linear("q_in", {"q_23": 1.0, "q_34": 1.0})
    assert l.solve_for("q_34", {"q_in": 5.0, "q_23": 2.0}) == pytest.approx(3.0)
    f = fixed("T1", 300.0)
    assert f.vars == ("T1",) and f.solve_for("T1", {}) == 300.0

def test_ideal_gas_relations_consistent_with_state():
    st = AIR.state(T=900.0, P=2e6)
    vals = {f"{k}7": v for k, v in st.items()}
    rels = AIR.relations("7")
    assert [r.name for r in rels] == ["eos 7", "u(T) 7", "h(T) 7", "s(T,P) 7"]
    for r in rels:
        assert abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-12
        for var in r.vars:
            assert r.solve_for(var, {k: vals[k] for k in r.vars if k != var}) == pytest.approx(vals[var], rel=1e-12)
```

```python
# tests/test_steps.py
import math, pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.steps import KINDS
from thermo_cycles.state import State

def mk(id, **two):
    return State(id=id, **AIR.state(**two))

def test_kinds_present():
    assert set(KINDS) == {"isentropic", "isochoric", "isobaric", "isothermal", "adiabatic"}
    assert KINDS["adiabatic"].hidden_states("2") == ("2s",) and KINDS["isobaric"].hidden_states("2") == ()

def values_of(*states):
    return {f"{p}{s.id}": getattr(s, p) for s in states for p in ("T", "P", "v", "u", "h", "s")}

@pytest.mark.parametrize("system", ["closed", "flow"])
def test_isentropic_relations_and_work(system):
    s1 = mk(1, T=300.0, P=1e5); s2 = mk(2, v=s1.v / 8, s=s1.s)
    rels = KINDS["isentropic"].relations("1", "2", system=system, role="compression", efficiency=None)
    vals = values_of(s1, s2)
    du = s2.u - s1.u; dh = s2.h - s1.h
    vals["q_12"] = 0.0; vals["w_12"] = -(du if system == "closed" else dh)
    for r in rels:
        assert abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10

def test_isobaric_closed_heat_is_dh_and_work_is_Pdv():
    s2 = mk(2, T=900.0, P=5e6); s3 = mk(3, T=1800.0, P=5e6)
    rels = KINDS["isobaric"].relations("2", "3", system="closed", role="heat_in", efficiency=None)
    vals = values_of(s2, s3); vals["q_23"] = s3.h - s2.h; vals["w_23"] = 5e6 * (s3.v - s2.v)
    assert all(abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10 for r in rels)

def test_isothermal_heat_is_T_ds():
    a = mk(1, T=500.0, P=1e6); b = mk(2, T=500.0, P=1e5)
    rels = KINDS["isothermal"].relations("1", "2", system="closed", role="expansion", efficiency=None)
    vals = values_of(a, b); vals["q_12"] = 500.0 * (b.s - a.s); vals["w_12"] = vals["q_12"]  # du = 0 ideal gas
    assert vals["q_12"] == pytest.approx(287.0 * 500.0 * math.log(10))
    assert all(abs(r.residual(*(vals[v] for v in r.vars))) / r.scale < 1e-10 for r in rels)

def test_adiabatic_compression_efficiency():  # Cengel 9-110 numbers: T1 293, rp 8, eta_c 0.87 -> T2 566.3
    rels = KINDS["adiabatic"].relations("1", "2", system="flow", role="compression", efficiency="eta_c")
    names = {v for r in rels for v in r.vars}
    assert {"s2s", "P2s", "h2s", "h1", "h2", "eta_c", "q_12", "w_12"} <= names
    s1 = mk(1, T=293.0, P=1e5); s2s = mk("2s", P=8e5, s=s1.s)
    T2 = 293.0 + (s2s.T - 293.0) / 0.87
    assert T2 == pytest.approx(566.3, rel=3e-4)

@pytest.mark.parametrize("kind,end", [
    ("isentropic", dict(P=2e6)), ("isochoric", dict(T=1500.0)), ("isobaric", dict(T=1500.0)), ("isothermal", dict(P=3e5))])
def test_paths_endpoints_and_monotonic_T(kind, end):
    a = mk(1, T=600.0, P=1e6)
    if kind == "isentropic": b = mk(2, P=end["P"], s=a.s)
    elif kind == "isochoric": b = mk(2, T=end["T"], v=a.v)
    elif kind == "isobaric": b = mk(2, T=end["T"], P=a.P)
    else: b = mk(2, T=a.T, P=end["P"])
    path = KINDS[kind].path(AIR, a, b, n=50)
    assert len(path) == 50
    for p in ("T", "P", "v", "s"):
        assert getattr(path[0], p) == pytest.approx(getattr(a, p), rel=1e-9)
        assert getattr(path[-1], p) == pytest.approx(getattr(b, p), rel=1e-9)
    Ts = [s.T for s in path]
    assert Ts == sorted(Ts) or Ts == sorted(Ts, reverse=True)
    if kind == "isentropic":
        assert max(s.s for s in path) - min(s.s for s in path) < 1e-9

def test_adiabatic_path_is_none():
    a = mk(1, T=300.0, P=1e5); b = mk(2, T=600.0, P=8e5)
    assert KINDS["adiabatic"].path(AIR, a, b) is None
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_relations.py tests/test_steps.py -q`. Expected: FAIL (import errors).
- [ ] **Step 3: Implement** the relations and helpers.
  - `scale` per relation: for relations in J/kg use 1e5, in Pa use 1e5, in K use 300, ratios 1, s 100. Pick it
    from the first variable's letter: `T`→300, `P`→1e5, `v`→1, `u`/`h`/`q`/`w`→1e5, `s`→100, otherwise 1.
  - Step kinds per spec §7.1. Each also adds `q_ab` and `w_ab` relations, as `linear` relations in endpoint props
    where possible. Isothermal q = T_a·(s_b − s_a) uses `product`-style explicit inverses.
  - Paths per spec §7.1 "path sampled along" column, built with `fluid.state()`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: relation helpers, ideal-gas state relations, process step kinds with q/w and paths"`

---

### Task 4: Structural graph algorithms

**Files:**
- Create: `src/thermo_cycles/solver/__init__.py` (empty for now), `src/thermo_cycles/solver/structure.py`
- Test: `tests/test_structure.py`

**Interfaces:**
- Produces (pure functions on integers; no thermodynamics):
  - `Matching(eq_to_var: tuple[int | None, ...], var_to_eq: tuple[int | None, ...])` and
    `max_matching(eq_vars: Sequence[frozenset[int]], n_vars: int) -> Matching` (Hopcroft-Karp).
  - `DMPartition(under_vars, under_eqs, over_vars, over_eqs: frozenset[int], rank: int)` and
    `dm_partition(eq_vars, n_vars) -> DMPartition`. "under" = reachable by alternating paths from unmatched
    variables; "over" = reachable from unmatched equations (spec §9.3).
  - `Block(eqs: tuple[int, ...], vars: tuple[int, ...])` and `blt_blocks(eq_vars, n_vars) -> list[Block]` in solve
    order (Tarjan SCC on the matched dependency graph). Raises `ValueError` if the matching is not perfect.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_structure.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** Hopcroft-Karp (BFS layering + DFS augment), the DM partition by alternating BFS, and
  iterative Tarjan (no recursion; graphs stay below 300 nodes but avoid recursion limits). Blocks come out in
  dependency order: a block's variables depend only on earlier blocks.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: 5 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: structural analysis (matching, Dulmage-Mendelsohn, BLT)"`

---

### Task 5: Catalogue format, registry, Otto, build_system

**Files:**
- Create: `src/thermo_cycles/model/cycle_def.py`, `src/thermo_cycles/catalogue/__init__.py`, `src/thermo_cycles/catalogue/piston.py` (Otto only for now), `src/thermo_cycles/solver/system.py`
- Test: `tests/test_cycle_def.py`, `tests/test_system.py`

**Interfaces:**
- Consumes: `Relation` helpers, `KINDS`, `IdealGas.relations`.
- Produces:
  - `StepDef`, `ParamDef` and `CycleDef` exactly as spec §8.1.
  - `register_cycle(defn, *, replace=False)` validates per §8.1 and raises `ValueError` on bad definitions.
    `get_cycle(name) -> CycleDef` resolves name, alias or case and raises `UnknownCycleError` with difflib
    suggestions. `list_cycles() -> list[str]` is sorted.
  - `OTTO` registered in `catalogue/piston.py`; `thermo_cycles/__init__.py` imports `catalogue` so the registry
    is populated on import.
  - `EquationSystem` (frozen dataclass): `variables: tuple[str, ...]`, `relations: tuple[Relation, ...]`,
    `var_kind: Mapping[str, str]` ("positive" | "signed" | "fraction"), `scan_range: Mapping[str, tuple[float,
    float, str]]` (lo, hi, "log" | "asinh"), `candidates: tuple[str, ...]` (ranked per §9.4),
    `defaults: Mapping[str, float]`, `aliases: Mapping[str, str]`, `cycle: CycleDef`,
    `equivalent: Mapping[str, str]` (u_k, h_k → T_k for ideal gas).
  - `build_system(defn, fluid) -> EquationSystem`, per spec §9.1 (states incl. hidden, fluid relations or
    generic closure, step relations, parameter definitions, cycle-sum relations, extra relations; no input
    relations).
  - `EquationSystem.with_inputs(inputs: Mapping[str, float]) -> tuple[Relation, ...]` appends `fixed` relations
    for the inputs plus the defaults not overridden.
  - Scan ranges per §9.5: T (1, 1e5, log), P (1, 1e10, log), v (1e-6, 1e4, log), ratios (1.0, 1e4, log),
    efficiencies (1e-6, 1.0, log), eps (0.0, 1.0, linear treated as asinh), signed (−1e9, 1e9, asinh).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cycle_def.py
import pytest
import thermo_cycles
from thermo_cycles.model.cycle_def import get_cycle, list_cycles, register_cycle, CycleDef, StepDef
from thermo_cycles.errors import UnknownCycleError

def test_otto_registered():
    d = get_cycle("OTTO")
    assert d.name == "otto" and d.system == "closed" and d.states == (1, 2, 3, 4)
    assert [s.kind for s in d.steps] == ["isentropic", "isochoric", "isentropic", "isochoric"]
    assert [s.role for s in d.steps] == ["compression", "heat_in", "expansion", "heat_out"]
    assert [p.name for p in d.params] == ["r_c"] and (d.v_max_state, d.v_min_state) == (1, 2)
    assert "otto" in list_cycles()

def test_unknown_cycle_suggests():
    with pytest.raises(UnknownCycleError) as ei:
        get_cycle("oto")
    assert "otto" in ei.value.suggestions

def test_register_rejects_open_loop():
    bad = CycleDef(name="bad", title="", aliases=(), system="closed", states=(1, 2, 3),
                   steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in")))
    with pytest.raises(ValueError):
        register_cycle(bad)
```

```python
# tests/test_system.py
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system

def test_otto_counts():
    sys_ = build_system(get_cycle("otto"), AIR)
    assert len(sys_.variables) == 38            # spec §9.1 worked count
    assert len(sys_.relations) == 34            # -> 4 inputs needed
    assert {"T1", "q_23", "w_41", "q_in", "q_out", "w_net", "eta_th", "mep", "r_c"} <= set(sys_.variables)
    assert "bwr" not in sys_.variables
    assert sys_.var_kind["T3"] == "positive" and sys_.var_kind["q_23"] == "signed"
    assert sys_.candidates[0] == "r_c" and "u3" not in sys_.candidates[:12]
    assert sys_.equivalent["u3"] == "T3" and sys_.equivalent["h3"] == "T3"

def test_with_inputs_appends_fixed():
    sys_ = build_system(get_cycle("otto"), AIR)
    rels = sys_.with_inputs({"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    assert len(rels) == 38
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_cycle_def.py tests/test_system.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Write `cycle_def.py` and the registry.
  - Add `OTTO` per spec §8.2: `suggest=("r_c","T3","q_in","P3","T4")`; typical inputs
    `(("T1","P1","r_c","T3"), ("T1","P1","r_c","q_in"))`; description; references
    `("Waterloo ME354 Tutorial wk8", "Cengel & Boles 7e SM 9-39")`.
  - Write `build_system`. Cycle-sum relations:
    - `q_in` = Σ q of heat_in steps;
    - `q_out` = −Σ q of heat_out steps;
    - `w_net` = Σ w of all steps;
    - `eta_th`·`q_in` = `w_net`;
    - closed: `mep`·(v_max − v_min) = `w_net`;
    - flow: `w_compression` = Σ w(compression), `w_expansion` = Σ w(expansion) (as variables), and
      `bwr`·`w_expansion` = −`w_compression`.
  - Candidate ranking per §9.4. Defaults come from `ParamDef.default`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: cycle definitions as data, registry, Otto, equation-system builder"`

---

### Task 6: Numeric block solver

**Files:**
- Create: `src/thermo_cycles/solver/numeric.py`
- Test: `tests/test_numeric.py`

**Interfaces:**
- Consumes: `build_system`, `EquationSystem.with_inputs`, `blt_blocks`, `Relation.solve_for`.
- Produces:
  - `SolveReport(blocks: tuple[tuple[str, ...], ...], methods: tuple[str, ...], max_residual: float)`, where each
    method is "explicit" | "torn" | "newton" | "closure".
  - `solve_numeric(system, inputs: Mapping[str, float], *, validate: Callable[[Mapping[str, float]], list[str]] |
    None = None) -> tuple[dict[str, float], SolveReport]`. This requires a structurally `ok` system (the caller
    checks in Task 7/8); otherwise it raises `ValueError`.
  - Raises `NoSolutionError`, `AmbiguousSolutionError(variable=..., candidates=...)` or `ConvergenceError` per
    spec §9.5.
  - Torn scan: 400 points, `brentq(xtol=1e-12·|x|)`. Roots are distinct when their relative difference is
    > 1e-6. With `validate` given, candidate full solutions with a non-empty problem list are discarded before the
    ambiguity decision.
  - Newton: `scipy.optimize.root(method="hybr")` on scaled residuals with log-space positive variables and up to
    8 seeds.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_numeric.py
import pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system
from thermo_cycles.solver.numeric import solve_numeric
from thermo_cycles.errors import AmbiguousSolutionError, NoSolutionError

OTTO = build_system(get_cycle("otto"), AIR)

def test_forward_otto_is_all_explicit():          # Waterloo ME354 wk8 (O1)
    vals, rep = solve_numeric(OTTO, {"T1": 288.0, "P1": 100e3, "r_c": 10.0, "q_in": 1800e3})
    assert set(rep.methods) == {"explicit"}
    assert vals["T2"] == pytest.approx(723.4, rel=3e-3) and vals["P3"] == pytest.approx(11216.6e3, rel=3e-3)
    assert vals["w_net"] == pytest.approx(1083.4e3, rel=3e-3) and rep.max_residual < 1e-9

def test_backward_otto_T4_given():                # Cengel 7e SM 9-39 (O2)
    vals, _ = solve_numeric(OTTO, {"T1": 308.0, "P1": 100e3, "r_c": 9.5, "T4": 800.0})
    assert vals["T3"] == pytest.approx(1969, rel=3e-3) and vals["P3"] == pytest.approx(6072e3, rel=3e-3)

def test_algebraic_loop_P1_solved():               # O3: P1 unknown, P3 given
    vals, rep = solve_numeric(OTTO, {"T1": 288.0, "r_c": 10.0, "q_in": 1800e3, "P3": 11216.6e3})
    assert vals["P1"] == pytest.approx(1e5, rel=1e-3)
    assert set(rep.methods) <= {"explicit", "torn", "newton"} and rep.max_residual < 1e-9

def test_two_roots_is_ambiguous():
    with pytest.raises(AmbiguousSolutionError) as ei:
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 250e3})
    roots = sorted(ei.value.candidates)
    assert ei.value.variable in {"r_c", "v2", "T2", "P2"}
    if ei.value.variable == "r_c":
        assert roots == pytest.approx([2.7362, 20.4306], rel=1e-4)

def test_no_root_is_no_solution():                 # max w_net at T3 = 1500 is 328.87 kJ/kg
    with pytest.raises(NoSolutionError):
        solve_numeric(OTTO, {"T1": 300.0, "P1": 1e5, "T3": 1500.0, "w_net": 400e3})
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_numeric.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `solve_numeric`:
  - Build `eq_vars` from `system.with_inputs(inputs)` and get the blocks from `blt_blocks`. For each block, use
    explicit if 1×1 with an inverse, else the torn scan, else Newton (spec §9.5).
  - Ambiguity reporting: if the torn variable is not a user-facing candidate, map it back to the user-facing
    candidate it determines, in this order: a catalogue parameter in the same block (e.g. `r_c`), otherwise the
    torn variable itself.
  - `max_residual` = max over all relations of |residual|/scale.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: 5 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: block solver (explicit, torn 1-D scan with multi-root detection, Newton)"`

---

### Task 7: Input resolution and advice

**Files:**
- Create: `src/thermo_cycles/solver/advice.py`
- Modify: `src/thermo_cycles/solver/system.py` (add `resolve_inputs`)
- Test: `tests/test_advice.py`, `tests/test_inputs.py`

**Interfaces:**
- Consumes: `EquationSystem`, `dm_partition`, `max_matching`.
- Produces:
  - `resolve_inputs(system, raw: Mapping[str, object]) -> dict[str, float]` (canonical names, floats; spec §3
    lookup order and §9.8 validation; emits `InputPlausibilityWarning`).
  - `Requirements(status, given, missing, add_one_of, example_sets, drop_one_of, determined)` (frozen dataclass;
    tuples of str).
  - `analyse(system, given: Iterable[str]) -> Requirements` (spec §9.4).
  - `raise_for(req, cycle_name) -> None` raises `UnderdeterminedError` / `OverdeterminedError` with the exact
    templates of spec §9.4 when the status is not `ok`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_advice.py
import pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system
from thermo_cycles.solver.advice import analyse, raise_for
from thermo_cycles.errors import UnderdeterminedError, OverdeterminedError

OTTO = build_system(get_cycle("otto"), AIR)

def test_ok():
    assert analyse(OTTO, ["T1", "P1", "r_c", "q_in"]).status == "ok"

def test_one_missing_options_exclude_determined():
    req = analyse(OTTO, ["T1", "P1", "r_c"])
    assert req.status == "under" and req.missing == 1
    assert {"T3", "q_in", "P3", "T4"} <= set(req.add_one_of)
    assert "T2" not in req.add_one_of and "T2" in req.determined and "u3" not in req.add_one_of
    assert req.add_one_of[0] == "T3"            # suggest order after r_c (already given)

def test_two_missing_examples():
    req = analyse(OTTO, ["T1", "P1"])
    assert req.missing == 2 and ("r_c", "T3") in req.example_sets

def test_over_determined_names_conflict():
    req = analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"])
    assert req.status == "over" and {"q_in", "T3"} <= set(req.drop_one_of)
    req2 = analyse(OTTO, ["T1", "u1", "P1", "r_c", "q_in"])
    assert {"T1", "u1"} <= set(req2.drop_one_of)

def test_message_template():
    with pytest.raises(UnderdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "P1", "r_c"]), "otto")
    msg = str(ei.value)
    assert msg.startswith("otto needs 1 more input.")
    assert "Given: T1, P1, r_c" in msg and "Add one of: T3, " in msg
    with pytest.raises(OverdeterminedError) as ei:
        raise_for(analyse(OTTO, ["T1", "P1", "r_c", "q_in", "T3"]), "otto")
    assert "is over-determined" in str(ei.value) and "allow_redundant=True" in str(ei.value)
```

```python
# tests/test_inputs.py  (Review Focus 1, 2, 4)
import math, warnings
import numpy as np, pytest
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system, resolve_inputs
from thermo_cycles.errors import InputError, UnknownInputError, InputPlausibilityWarning

OTTO = build_system(get_cycle("otto"), AIR)

def test_names_case_and_alias():
    got = resolve_inputs(OTTO, {"t1": 300, "P1": np.float64(1e5), "R_C": 8, "Q_IN": 1e6})
    assert got == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6}
    assert resolve_inputs(OTTO, {"r": 8, "T1": 300})["r_c"] == 8.0

def test_unknown_name_suggests():
    with pytest.raises(UnknownInputError) as ei:
        resolve_inputs(OTTO, {"rc": 8})
    assert "r_c" in ei.value.suggestions

@pytest.mark.parametrize("bad", [math.nan, math.inf, -300.0, "300", 1j])
def test_bad_values(bad):
    with pytest.raises(InputError, match="T1"):
        resolve_inputs(OTTO, {"T1": bad})

def test_plausibility_warning_for_kpa_and_celsius():
    with pytest.warns(InputPlausibilityWarning, match="P1"):
        resolve_inputs(OTTO, {"P1": 100})
    with pytest.warns(InputPlausibilityWarning, match="T1"):
        resolve_inputs(OTTO, {"T1": 25})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_inputs(OTTO, {"T1": 300, "P1": 1e5})
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_advice.py tests/test_inputs.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `resolve_inputs` and `analyse` as in spec §9.4: k = 1 by candidate scan, k = 2 by
  pair scan, k ≥ 3 greedy; fold the ideal-gas equivalents. Then implement `raise_for` with the exact templates.
  The plausibility check applies to names starting with `P` (state props) < 1000 and `T` < 100.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: input resolution, structural advice and under/over-determination errors"`

---

### Task 8: Results, CycleState, solve_cycle, validity checks, Cycle front

**Files:**
- Create: `src/thermo_cycles/results.py`, `src/thermo_cycles/cycle_state.py`, `src/thermo_cycles/cycle.py`
- Modify: `src/thermo_cycles/solver/__init__.py` (add `solve_cycle`), `src/thermo_cycles/__init__.py` (exports)
- Test: `tests/test_results.py`, `tests/test_cycle_api.py`, `tests/test_reference_otto.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `StepResult`, `CycleResults` (spec §5.3).
  - `compute_results(defn, fluid, values) -> tuple[dict[int, State], dict[str, State], tuple[StepResult, ...],
    CycleResults]`.
  - `validity_problems(defn, values, steps) -> list[str]` (spec §9.7 rules and message style).
  - `CycleState` (spec §5.2).
  - `solve_cycle(defn, fluid, inputs, *, allow_redundant=False, check="raise") -> CycleState` (resolve → analyse
    → raise_for, or the redundant handling of §9.6 → solve_numeric(validate=...) → results → validity).
  - `Cycle` (spec §4 signature; `margins` returns an empty report until Task 11, `plot` raises
    `NotImplementedError` until Task 13).
  - `requirements(name, *, fluid=None, **inputs) -> Requirements`.
  - Package exports: `Cycle, IdealGas, AIR, requirements, list_cycles, register_cycle, errors`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reference_otto.py  (spec §15.1 O1, O2)
import pytest
from thermo_cycles import Cycle

def test_O1_waterloo_me354():
    c = Cycle("otto", T1=288, P1=100e3, r_c=10, q_in=1800e3)
    exp = {"T2": 723.4, "P2": 2511.9e3, "T3": 3230.4, "P3": 11216.6e3, "T4": 1286, "P4": 445.6e3}
    for k, v in exp.items():
        assert c.value(k) == pytest.approx(v, rel=3e-3), k
    r = c.results
    assert r.q_out == pytest.approx(716.6e3, rel=3e-3) and r.w_net == pytest.approx(1083.4e3, rel=3e-3)
    assert r.eta_th == pytest.approx(0.602, rel=3e-3) and r.mep == pytest.approx(1456.4e3, rel=3e-3)
    assert r.bwr is None and r.eta_carnot == pytest.approx(1 - 288 / c.states[3].T)
    assert r.eta_ii == pytest.approx(r.eta_th / r.eta_carnot)

def test_O2_cengel_9_39_backward():
    c = Cycle("otto", T1=308, P1=100e3, r_c=9.5, T4=800)
    assert c.states[2].T == pytest.approx(757.9, rel=3e-3) and c.states[2].P == pytest.approx(2338e3, rel=3e-3)
    assert c.states[3].T == pytest.approx(1969, rel=3e-3) and c.states[3].P == pytest.approx(6072e3, rel=3e-3)
    assert c.results.eta_th == pytest.approx(0.594, rel=3e-3) and c.results.mep == pytest.approx(652e3, rel=3e-3)
```

```python
# tests/test_results.py
import pytest
from thermo_cycles import Cycle

def test_energy_and_entropy_closure():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    assert c.results.w_net == pytest.approx(c.results.q_in - c.results.q_out, rel=1e-12)
    assert sum(s.ds for s in c.steps) == pytest.approx(0, abs=1e-9)
    assert [ (s.a, s.b) for s in c.steps ] == [(1, 2), (2, 3), (3, 4), (4, 1)]
    assert c.steps[0].w < 0 and c.steps[2].w > 0 and c.steps[1].q > 0 and c.steps[3].q < 0
    assert c.cycle_state.report.max_residual < 1e-9
```

```python
# tests/test_cycle_api.py  (incl. Review Focus 3)
import dataclasses, pytest
from thermo_cycles import Cycle, requirements, AIR
from thermo_cycles.errors import (InvalidCycleError, UnderdeterminedError, InconsistentInputError,
                                  OverdeterminedError, CycleValidityWarning)

def test_states_params_value():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    assert c.states[1].T == 300 and c.params["r_c"] == 8 and c.value("P1") == 1e5
    assert c.inputs == {"T1": 300.0, "P1": 1e5, "r_c": 8.0, "q_in": 1e6} and c.fluid is AIR

def test_with_inputs_does_not_mutate():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    c2 = c.with_inputs(r_c=10)
    assert c.params["r_c"] == 8 and c2.params["r_c"] == 10
    c3 = c.with_inputs(q_in=None, T3=2000)
    assert "q_in" not in c3.inputs and c3.states[3].T == pytest.approx(2000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.states[1].T = 1

def test_heat_addition_below_T2_names_step():          # Review Focus 3
    with pytest.raises(InvalidCycleError, match=r"2.3"):
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600)
    with pytest.warns(CycleValidityWarning):
        Cycle("otto", T1=300, P1=1e5, r_c=10, T3=600, check="warn")

def test_underdetermined_from_front():
    with pytest.raises(UnderdeterminedError, match="Add one of: T3"):
        Cycle("otto", T1=300, P1=1e5, r_c=10)
    assert requirements("otto", T1=300, P1=1e5, r_c=10).missing == 1

def test_redundant_consistent_and_inconsistent():
    base = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6)
    with pytest.raises(OverdeterminedError):
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=base.states[3].T)
    ok = Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=base.states[3].T, allow_redundant=True)
    assert ok.results.eta_th == pytest.approx(base.results.eta_th)
    with pytest.raises(InconsistentInputError) as ei:
        Cycle("otto", T1=300, P1=1e5, r_c=8, q_in=1e6, T3=2000, allow_redundant=True)
    assert ei.value.name in {"T3", "q_in"} and ei.value.given != pytest.approx(ei.value.implied)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_reference_otto.py tests/test_results.py tests/test_cycle_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the results (spec §10), `validity_problems` (§9.7), `CycleState`, `solve_cycle` (with
  the §9.6 redundant handling) and `Cycle` (eager solve, immutable; `with_inputs` merges and drops `None`;
  `summary()` and `to_dict()`).
  Validity message format: `f"step {a}→{b} ({kind}, {role}) has q = {q/1e3:.1f} kJ/kg < 0: ..."`.
  Heat-step sign checks use the tolerance in spec §9.7 (q = 0 is valid on a heat_in step, which the Task 9 Dual
  limit tests with alpha = 1 and cutoff = 1 rely on).
- [ ] **Step 4: Run the tests to verify they pass.** Run `python -m pytest -q`. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: results, CycleState, validity checks and Cycle front class"`

---

### Task 9: Diesel, Dual, Atkinson + references, closed forms, custom cycle, old-code regression

**Files:**
- Modify: `src/thermo_cycles/catalogue/piston.py` (add DIESEL, DUAL, ATKINSON)
- Test: `tests/test_reference_piston.py`, `tests/test_closed_form_piston.py`, `tests/test_custom_cycle.py`, `tests/test_regression_old.py`, `tests/test_atkinson_advice.py`

**Interfaces:**
- Consumes: `CycleDef`, `StepDef`, `ParamDef`, `ratio`, `register_cycle`, `Cycle`, `IdealGas`.
- Produces: cycles `diesel`, `dual` and `atkinson` exactly per spec §8.2 rows (steps, roles, parameter
  definitions, `v_max_state`/`v_min_state`, bounds cutoff ≥ 1, alpha ≥ 1, r_c > 1, r_e > 1). Suggest orders:
  diesel `("r_c","cutoff","T3","q_in")`, dual `("r_c","alpha","cutoff","T4","q_in","P3")`, atkinson
  `("r_c","T3","P3","q_in","r_e")`. References per spec §15.1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reference_piston.py  (spec §15.1 D1, D2, U1, A1)
import pytest
from thermo_cycles import Cycle, IdealGas

def test_D1_cengel8e_9_57_states():
    c = Cycle("diesel", T1=343, P1=97e3, r_c=22, cutoff=1.8)
    assert [c.states[i].T for i in (2, 3, 4)] == pytest.approx([1181, 2126, 781], rel=3e-3)
    assert c.results.eta_th == pytest.approx(0.66884, rel=1e-4)

def test_D2_cengel7e_9_58_states_1_to_3():
    c = Cycle("diesel", T1=293, P1=95e3, r_c=20, T3=2200)
    assert c.states[2].T == pytest.approx(971.1, rel=3e-3)
    assert c.params["cutoff"] == pytest.approx(2.265, rel=3e-3) and c.results.q_in == pytest.approx(1235e3, rel=3e-3)

def test_U1_cengel7e_9_54_dual():                 # Cengel x->3, 3->4, 4->5
    c = Cycle("dual", T1=253, P1=80e3, r_c=14, alpha=1.5, cutoff=1.2)
    assert c.states[1].v == pytest.approx(0.9076, rel=3e-3)
    assert c.states[2].P == pytest.approx(3219e3, rel=3e-3) and c.states[3].P == pytest.approx(4829e3, rel=3e-3)
    assert c.states[4].P == pytest.approx(4829e3, rel=3e-3)
    assert [c.states[i].T for i in (2, 3, 4, 5)] == pytest.approx([727.1, 1091, 1309, 490.0], rel=3e-3)
    r = c.results
    assert (r.q_in, r.q_out, r.eta_th) == pytest.approx((480.4e3, 170.2e3, 0.646), rel=3e-3)

def test_A1_rajput_ex_13_31_pressure_inputs():
    c = Cycle("atkinson", fluid=IdealGas(cp=920, gamma=1.22), T1=300, P1=1e5, P2=4e5, P3=16e5)
    assert [c.states[i].T for i in (2, 3, 4)] == pytest.approx([385.2, 1540.8, 934.9], rel=1e-3)
    assert c.results.q_out == pytest.approx(584.1e3, rel=3e-3)
    assert c.results.q_in == pytest.approx(866.7e3, rel=1e-2)     # book's k=1.22 vs cp/cv=1.2267
    assert c.results.w_net == pytest.approx(282.6e3, rel=2e-2)
    assert c.results.eta_th == pytest.approx(0.326, rel=1.5e-2)
    assert c.states[4].P == pytest.approx(1e5) and c.params["r_e"] > c.params["r_c"]

def test_atkinson_probe_case():
    c = Cycle("atkinson", T1=300, P1=1e5, r_c=8, T3=1800)
    assert c.params["r_e"] == pytest.approx(15.8813, rel=1e-5) and c.states[4].T == pytest.approx(595.550, rel=1e-5)
    assert c.results.eta_th == pytest.approx(0.627497, rel=1e-5) and c.results.mep == pytest.approx(312.25e3, rel=1e-4)
```

```python
# tests/test_closed_form_piston.py  (spec §15.2)
import itertools, pytest
from thermo_cycles import Cycle, IdealGas

G = [1.3, 1.4, 1.67]

@pytest.mark.parametrize("g,r,q", itertools.product(G, [4, 8, 12, 20], [0.5e6, 1.8e6]))
def test_otto(g, r, q):
    c = Cycle("otto", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, q_in=q)
    assert c.results.eta_th == pytest.approx(1 - r ** (1 - g), rel=1e-9)

@pytest.mark.parametrize("g,r,co", itertools.product(G, [12, 18, 22], [1.5, 2, 3]))
def test_diesel(g, r, co):
    c = Cycle("diesel", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, cutoff=co)
    assert c.results.eta_th == pytest.approx(1 - r ** (1 - g) * (co ** g - 1) / (g * (co - 1)), rel=1e-9)

@pytest.mark.parametrize("g,r,a,co", itertools.product(G, [10, 14], [1.2, 1.5], [1.2, 1.8]))
def test_dual(g, r, a, co):
    c = Cycle("dual", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, alpha=a, cutoff=co)
    exp = 1 - r ** (1 - g) * (a * co ** g - 1) / ((a - 1) + g * a * (co - 1))
    assert c.results.eta_th == pytest.approx(exp, rel=1e-9)

def test_dual_limits():
    d = Cycle("dual", T1=300, P1=1e5, r_c=14, alpha=1.0, cutoff=2.0)
    di = Cycle("diesel", T1=300, P1=1e5, r_c=14, cutoff=2.0)
    assert d.results.eta_th == pytest.approx(di.results.eta_th, rel=1e-9)
    o = Cycle("dual", T1=300, P1=1e5, r_c=8, alpha=3.0, cutoff=1.0)
    ot = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=o.states[3].T)
    assert o.results.eta_th == pytest.approx(ot.results.eta_th, rel=1e-9)

@pytest.mark.parametrize("g,r,T3", itertools.product(G, [6, 8, 10], [1500, 2400]))
def test_atkinson(g, r, T3):
    c = Cycle("atkinson", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_c=r, T3=T3)
    re = c.params["r_e"]
    assert c.results.eta_th == pytest.approx(1 - g * (re - r) / (re ** g - r ** g), rel=1e-9)
    assert 0 < c.results.eta_th < c.results.eta_carnot
```

```python
# tests/test_custom_cycle.py  (spec §15.1 C1, Cengel 7e SM 9-24)
import pytest
from thermo_cycles import Cycle, IdealGas, register_cycle
from thermo_cycles.model.cycle_def import CycleDef, StepDef, ParamDef
from thermo_cycles.model.relations import ratio

def test_C1_three_step_cycle():
    register_cycle(CycleDef(
        name="test_9_24", title="Cengel 9-24", aliases=(), system="closed", states=(1, 2, 3),
        steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
               StepDef("isochoric", 3, 1, "heat_out")),
        params=(ParamDef("r_c", "v1/v2", ratio("r_c", "v1", "v2")),), v_max_state=1, v_min_state=2), replace=True)
    c = Cycle("test_9_24", fluid=IdealGas(R=300, cv=700), T1=293, P1=1e5, r_c=5)
    assert (c.states[2].T, c.states[3].T) == pytest.approx((584.4, 2922), rel=3e-3)
    w = {(s.a, s.b): s.w for s in c.steps}; q = {(s.a, s.b): s.q for s in c.steps}
    assert w[(1, 2)] == pytest.approx(-204.0e3, rel=3e-3) and w[(2, 3)] == pytest.approx(701.3e3, rel=3e-3)
    assert q[(2, 3)] == pytest.approx(2338e3, rel=3e-3) and q[(3, 1)] == pytest.approx(-1840.3e3, rel=3e-3)
    k = 1000 / 700
    assert c.results.eta_th == pytest.approx((1 - 1 / k) - (1 - 5 ** (1 - k)) / (k * 4), rel=1e-9)
```

```python
# tests/test_regression_old.py  (spec §15.4; values from commit 9c1505b AtkinsonSolver DESIGN mode)
import pytest
from thermo_cycles import Cycle

def test_old_atkinson_design_mode_is_diesel():
    c = Cycle("diesel", T1=298.0, P1=1e5, r_c=20.0, T3=2480.988)
    assert c.states[2].T == pytest.approx(987.707297, rel=1e-6) and c.states[2].P == pytest.approx(6628908.03468, rel=1e-6)
    assert c.states[3].v == pytest.approx(0.107415, rel=1e-5) and c.states[4].T == pytest.approx(1081.960052, rel=1e-6)
    assert c.states[4].P == pytest.approx(363073.84284, rel=1e-6) and c.states[1].s == pytest.approx(87.464217, rel=1e-6)
    r = c.results
    assert (r.eta_th, r.w_net, r.mep) == pytest.approx((0.6250058918, 937509.1289, 1153861.6499), rel=1e-6)
```

```python
# tests/test_atkinson_advice.py  (the brief's example message)
import pytest
from thermo_cycles import Cycle
from thermo_cycles.errors import UnderdeterminedError, OverdeterminedError

def test_brief_example():
    with pytest.raises(UnderdeterminedError) as ei:
        Cycle("atkinson", T1=298, P1=1e5, r_c=10)
    assert {"T3", "q_in", "P3", "r_e"} <= set(ei.value.add_one_of)
    assert "T2" not in ei.value.add_one_of and "P4" not in ei.value.add_one_of

def test_r_e_plus_T3_over_determines():
    with pytest.raises(OverdeterminedError) as ei:
        Cycle("atkinson", T1=298, P1=1e5, r_c=10, r_e=14, T3=2000)
    assert {"r_c", "r_e", "T3"} <= set(ei.value.drop_one_of)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_reference_piston.py tests/test_closed_form_piston.py tests/test_custom_cycle.py tests/test_regression_old.py tests/test_atkinson_advice.py -q`. Expected: FAIL (`UnknownCycleError: diesel`).
- [ ] **Step 3: Implement** the three `CycleDef`s per spec §8.2. Nothing else should change. If a test needs
  solver changes, stop and report it: that is a solver bug, not a catalogue issue.
- [ ] **Step 4: Run the tests to verify they pass.** Run `python -m pytest -q`. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: Diesel, Dual and true Atkinson cycles with textbook and closed-form tests"`

---

### Task 10: Brayton and Brayton with regeneration

**Files:**
- Create: `src/thermo_cycles/catalogue/gas_turbine.py`
- Modify: `src/thermo_cycles/catalogue/__init__.py`
- Test: `tests/test_reference_brayton.py`, `tests/test_closed_form_brayton.py`

**Interfaces:**
- Consumes: `KINDS["adiabatic"]` (hidden states), `CycleDef.extra_relations`, `ParamDef.default`, `linear`,
  `ratio`.
- Produces: `brayton` and `brayton_regen` per spec §8.2:
  - `eta_c`, `eta_t` default 1.0 with bounds (0, 1];
  - `eps` = (h3 − h2)/(h5 − h2) with bounds [0, 1];
  - `dT_regen` = T5 − T3;
  - extra relation `h3 − h2 − (h5 − h6) = 0`;
  - suggest orders: brayton `("r_p","T3","q_in","eta_c","eta_t")`, regen
    `("r_p","T4","eps","dT_regen","eta_c","eta_t")`.
  - Validity rule (spec §9.7): T5 ≥ T3 and T6 ≥ T2, with the message naming "regenerator".

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reference_brayton.py  (spec §15.1 B1, B2; incl. Review Focus 5)
import pytest
from thermo_cycles import Cycle, IdealGas
from thermo_cycles.errors import InvalidCycleError

def test_B1_thermostate_cold_air():
    c = Cycle("brayton", fluid=IdealGas(R=287, gamma=1.401662), T1=300, P1=1e5, r_p=10, T3=1400)
    assert c.states[2].T == pytest.approx(580.34, rel=1e-4) and c.states[4].T == pytest.approx(723.71, rel=1e-4)
    assert c.results.bwr == pytest.approx(0.4145, rel=5e-4) and c.results.eta_th == pytest.approx(0.4831, rel=5e-4)
    assert c.results.mep is None

def test_B2_cengel7e_9_110_regen_with_efficiencies():
    c = Cycle("brayton_regen", T1=293, P1=1e5, r_p=8, T4=1073, eta_c=0.87, eta_t=0.93, dT_regen=10)
    h = c.hidden_states
    assert h["2s"].T == pytest.approx(530.8, rel=3e-3) and h["5s"].T == pytest.approx(592.3, rel=3e-3)
    exp = {2: 566.3, 5: 625.9, 3: 615.9, 6: 576.3}
    for i, T in exp.items():
        assert c.states[i].T == pytest.approx(T, rel=3e-3), i
    assert c.results.w_net == pytest.approx(174.7e3, rel=3e-3)
    assert c.results.q_in == pytest.approx(394.4 / 0.8586 * 1e3, rel=3e-3)
    assert c.params["eps"] == pytest.approx(0.83252, rel=1e-4)
    c2 = Cycle("brayton_regen", T1=293, P1=1e5, r_p=8, T4=1073, eta_c=0.87, eta_t=0.93, eps=c.params["eps"])
    assert c2.states[3].T == pytest.approx(c.states[3].T, rel=1e-9)

def test_regenerator_cannot_heat_hot_stream():          # Review Focus 5: r_p 40 -> T5 < T2
    with pytest.raises(InvalidCycleError, match="regenerator"):
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=40, T4=1200, eps=0.8)
```

```python
# tests/test_closed_form_brayton.py
import itertools, pytest
from thermo_cycles import Cycle, IdealGas

@pytest.mark.parametrize("g,rp,T3", itertools.product([1.3, 1.4, 1.67], [4, 8, 16], [1200, 1600]))
def test_brayton(g, rp, T3):
    c = Cycle("brayton", fluid=IdealGas(R=287, gamma=g), T1=300, P1=1e5, r_p=rp, T3=T3)
    x = rp ** ((g - 1) / g)
    assert c.results.eta_th == pytest.approx(1 - 1 / x, rel=1e-9)
    assert c.results.bwr == pytest.approx(300 / T3 * x, rel=1e-9)
    assert c.results.w_net == pytest.approx(c.results.q_in - c.results.q_out, rel=1e-12)

@pytest.mark.parametrize("rp", [3, 5, 8])
def test_ideal_regeneration(rp):
    c = Cycle("brayton_regen", T1=300, P1=1e5, r_p=rp, T4=1400, eps=1.0)
    assert c.results.eta_th == pytest.approx(1 - 300 / 1400 * rp ** (0.4 / 1.4), rel=1e-9)
    assert sum(s.q for s in c.steps if s.role == "internal") == pytest.approx(0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail.** Run `python -m pytest tests/test_reference_brayton.py tests/test_closed_form_brayton.py -q`. Expected: FAIL (`UnknownCycleError`).
- [ ] **Step 3: Implement** the two definitions and the regenerator validity rule in `validity_problems`. The
  rule is keyed on `role == "internal"` step pairs declared in the definition. Use
  `CycleDef.extra_relations` for the energy balance.
- [ ] **Step 4: Run the tests to verify they pass.** Run `python -m pytest -q`. Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: Brayton (with component efficiencies) and regenerative Brayton"`

---

### Task 11: Margins

**Files:**
- Create: `src/thermo_cycles/margins.py`
- Modify: `src/thermo_cycles/cycle.py` (wire `limits`, `on_violation`, `margins`)
- Test: `tests/test_margins.py`

**Interfaces:**
- Consumes: `CycleState`.
- Produces: `Margin`, `SecondLaw(eta_th, eta_carnot, eta_ii, gap)`, `MarginReport` (spec §11),
  `parse_limit_key(key, system) -> tuple[str, str]` (name, "max" | "min"; raises `UnknownInputError`) and
  `evaluate_limits(cs: CycleState, limits: Mapping[str, float]) -> MarginReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_margins.py
import warnings, pytest
from thermo_cycles import Cycle
from thermo_cycles.errors import LimitViolationWarning, LimitViolationError, UnknownInputError

KW = dict(T1=300, P1=1e5, r_c=8, T3=1800)

def test_margin_values_and_location():
    c = Cycle("atkinson", limits={"T_max": 2000, "P_max": 8e6, "eta_th_min": 0.5}, **KW)
    m = c.margins["T_max"]
    assert (m.kind, m.limit, m.value, m.where) == ("max", 2000, pytest.approx(1800), 3)
    assert m.margin == pytest.approx(200) and m.margin_pct == pytest.approx(10.0) and m.ok
    assert c.margins["eta_th_min"].margin == pytest.approx(c.results.eta_th - 0.5)
    assert c.margins.ok and c.margins.violated == ()
    sl = c.margins.second_law
    assert sl.eta_ii == pytest.approx(c.results.eta_th / c.results.eta_carnot)
    assert sl.gap == pytest.approx(c.results.eta_carnot - c.results.eta_th)

def test_violation_policies():
    with pytest.warns(LimitViolationWarning, match="P_max"):
        c = Cycle("atkinson", limits={"P_max": 1e6}, **KW)
    assert not c.margins["P_max"].ok and c.margins.worst.key == "P_max"
    with pytest.raises(LimitViolationError):
        Cycle("atkinson", limits={"P_max": 1e6}, on_violation="raise", **KW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Cycle("atkinson", limits={"P_max": 1e6}, on_violation="ignore", **KW)

def test_state_specific_and_unknown_keys():
    c = Cycle("atkinson", limits={"P2_max": 2e6, "r_e_max": 20}, **KW)
    assert c.margins["P2_max"].where == 2 and c.margins["r_e_max"].ok
    with pytest.raises(UnknownInputError):
        Cycle("atkinson", limits={"Tmax": 2000}, **KW)

def test_no_limits_still_has_second_law():
    c = Cycle("atkinson", **KW)
    assert len(c.margins) == 0 and c.margins.second_law.eta_carnot == pytest.approx(1 - 300 / 1800)
```

- [ ] **Step 2: Run the test to verify it fails.** Run `python -m pytest tests/test_margins.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per spec §11. Property-letter keys take the extreme over main and hidden states and
  record `where`. `table()` gives a fixed-width text table.
- [ ] **Step 4: Run the test to verify it passes.** Same command. Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: margins to user limits and second-law efficiency report"`

---

### Task 12: Optimisation

**Files:**
- Create: `src/thermo_cycles/optimise.py`
- Modify: `src/thermo_cycles/__init__.py` (export `optimise`)
- Test: `tests/test_optimise.py`

**Interfaces:**
- Consumes: `Cycle`, `requirements`, `MarginReport`.
- Produces: `optimise(...)` and `OptimisationResult` exactly as spec §4 / §12.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimise.py
import pytest
from thermo_cycles import optimise
from thermo_cycles.errors import NoFeasiblePointError, UnderdeterminedError

def test_old_efficiency_mode_replacement_boundary_optimum():
    # old EFFICIENCY mode: T1 298, T3_MAX 2700, r_c 5..60, P2/P1 <= 70 (P1 was hard-coded 101325)
    res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
                   fixed=dict(T1=298, P1=101325, T3=2700), limits={"P2_max": 70 * 101325})
    assert res.x["r_c"] == pytest.approx(70 ** (1 / 1.4), rel=1e-5)       # 20.7935
    assert res.value == pytest.approx(0.623769, rel=1e-5)
    assert res.active == ("P2_max",) and res.feasible
    assert res.cycle.states[2].P == pytest.approx(70 * 101325, rel=1e-5)   # states are the optimum's

def test_interior_optimum_max_work():
    res = optimise("otto", vary={"r_c": (1.5, 30)}, maximise="w_net", fixed=dict(T1=300, P1=1e5, T3=1500))
    assert res.x["r_c"] == pytest.approx(5 ** 1.25, rel=1e-4)              # 7.4767
    assert res.value == pytest.approx(328.87e3, rel=1e-4) and res.active == ()

def test_two_variables():
    # Brayton max w_net: T3 pinned by T_max, then r_p* = (T3/T1)^(gamma/(2(gamma-1))) = 18.7175,
    # w* = cp (T3 - 2 sqrt(T1 T3) + T1) = 516.674 kJ/kg
    res = optimise("brayton", vary={"r_p": (2, 40), "T3": (800, 2000)}, maximise="w_net",
                   fixed=dict(T1=300, P1=1e5), limits={"T_max": 1600})
    assert res.feasible and res.active == ("T_max",)
    assert res.x["T3"] == pytest.approx(1600, rel=1e-4) and res.x["r_p"] == pytest.approx(18.7175, rel=1e-3)
    assert res.value == pytest.approx(516.674e3, rel=1e-4)

def test_callable_objective_and_errors():
    res = optimise("otto", vary={"r_c": (2, 20)}, minimise=lambda c: -c.results.eta_th,
                   fixed=dict(T1=300, P1=1e5, q_in=1e6), limits={"T_max": 2500})
    assert res.cycle.results.T_max <= 2500 * (1 + 1e-6)
    with pytest.raises(NoFeasiblePointError):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5, q_in=1e6),
                 limits={"P_max": 1e5})
    with pytest.raises(UnderdeterminedError):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5))
```

- [ ] **Step 2: Run the test to verify it fails.** Run `python -m pytest tests/test_optimise.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per spec §12: well-posedness check first; 1-D grid → bounded refine or boundary
  `brentq` on the binding limit's margin; ≥ 2-D Sobol + SLSQP. `active` means |margin_pct| < 1e-3. Record the
  history.
- [ ] **Step 4: Run the test to verify it passes.** Same command. Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: optimise() over cycle inputs with limits as constraints"`

---

### Task 13: Plotting

**Files:**
- Create: `src/thermo_cycles/plotting.py`
- Modify: `src/thermo_cycles/cycle.py` (`Cycle.plot`), `src/thermo_cycles/__init__.py` (export `plot_cycles`, lazy)
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `StepKind.path`, `Cycle`.
- Produces: `plot_cycle(...)` and `plot_cycles(...)` (spec §13). The x/y data of each step line is retrievable as
  `ax.lines` with `gid=f"step {a}-{b}"`. Irreversible steps use linestyle `"--"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plotting.py
import numpy as np, pytest
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from thermo_cycles import Cycle, plot_cycles

def lines(ax):
    return {l.get_gid(): l for l in ax.lines if l.get_gid() and l.get_gid().startswith("step")}

def test_Ts_true_curves():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    L = lines(c.plot("Ts"))
    s12 = L["step 1-2"].get_xdata()
    assert np.ptp(s12) < 1e-6                                   # isentrope vertical in T-s
    T23 = L["step 2-3"].get_ydata(); s23 = L["step 2-3"].get_xdata()
    assert len(T23) > 10
    mid = len(T23) // 2                                         # isochore is curved, not a chord
    chord = T23[0] + (T23[-1] - T23[0]) * (s23[mid] - s23[0]) / (s23[-1] - s23[0])
    assert abs(T23[mid] - chord) > 1.0

def test_Pv_and_units_and_irreversible_dashed():
    c = Cycle("brayton", T1=300, P1=1e5, r_p=10, T3=1400, eta_c=0.85, eta_t=0.9)
    ax = c.plot("Pv", P_unit="kPa")
    L = lines(ax)
    assert L["step 1-2"].get_linestyle() == "--" and L["step 2-3"].get_linestyle() == "-"
    assert max(L["step 2-3"].get_ydata()) == pytest.approx(1000.0, rel=1e-9)     # 1e6 Pa in kPa
    assert "kPa" in ax.get_ylabel()

def test_overlay_and_both():
    a = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800); b = a.with_inputs(r_c=10)
    ax = plot_cycles([a, b], "Ts", labels=["r=8", "r=10"])
    assert len([t.get_text() for t in ax.get_legend().get_texts()]) == 2
    fig, (ax1, ax2) = a.plot("both")
    assert ax1 is not ax2
```

- [ ] **Step 2: Run the test to verify it fails.** Run `python -m pytest tests/test_plotting.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** per spec §13. matplotlib is imported inside the functions. Its absence raises
  `MissingDependencyError` with the exact message from §13. Add a test that monkeypatches
  `sys.modules["matplotlib"] = None` and asserts the error. Steps that have a hidden state get a dotted isentrope
  to the hidden state and an open marker.
- [ ] **Step 4: Run the test to verify it passes.** Run `python -m pytest -q` (the package test still asserts
  matplotlib is not imported by `import thermo_cycles`). Expected: all passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: T-s and P-v plots with true process curves (optional matplotlib)"`

---

### Task 14: describe(), README, examples, full suite

**Files:**
- Modify: `src/thermo_cycles/cycle.py` (`describe`), `src/thermo_cycles/__init__.py` (final `__all__`), `README.md`
- Create: `examples/quickstart.py`, `examples/old_modes_migrated.py`
- Test: `tests/test_describe_and_examples.py`

**Interfaces:**
- Produces: `describe(name) -> str` containing title, numbered steps "1→2 isentropic (compression)", parameters
  with descriptions and defaults, DOF ("needs 4 inputs"), typical input sets and references. Final `__all__ =
  ["Cycle", "IdealGas", "AIR", "requirements", "describe", "list_cycles", "register_cycle", "optimise",
  "plot_cycles", "errors", "units", "__version__"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_describe_and_examples.py
import runpy, pathlib, pytest
import thermo_cycles as tc

def test_describe_atkinson():
    text = tc.describe("atkinson")
    assert "1→2 isentropic (compression)" in text and "4→1 isobaric (heat_out)" in text
    assert "r_e" in text and "needs 4 inputs" in text and "Rajput" in text

def test_list_cycles_v1():
    assert tc.list_cycles() == ["atkinson", "brayton", "brayton_regen", "diesel", "dual", "otto"]

@pytest.mark.parametrize("script", ["quickstart.py", "old_modes_migrated.py"])
def test_examples_run(script, monkeypatch):
    pytest.importorskip("matplotlib").use("Agg")
    runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / script), run_name="__main__")
```

  The custom cycle registered in Task 9 tests must not leak into `list_cycles()`. Have `test_custom_cycle.py`
  unregister it in a fixture (`thermo_cycles.model.cycle_def._unregister(name)`, private helper), or make
  `list_cycles()` exclude names starting with `test_`. The fixture is the preferred option.
- [ ] **Step 2: Run the test to verify it fails.** Run `python -m pytest tests/test_describe_and_examples.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Write `describe`.
  - Write `examples/quickstart.py`, which shows the spec §4 snippets with Atkinson and Otto and saves
    `quickstart_Ts.png` to a temp directory.
  - Write `examples/old_modes_migrated.py`, which shows the old DESIGN mode as `Cycle("diesel", ...)` and the
    old EFFICIENCY mode as `optimise(...)`.
  - Rewrite the README: install (`pip install -e .[plot]`), quick start, the catalogue table from spec §8.2,
    conventions from §3, error example, references from §15.1, and the "Atkinson fix" note from §16.
- [ ] **Step 4: Run the full suite.** Run `python -m pytest -q`. Expected: all passed, 0 failures. Record the
  count in the commit message.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "docs: describe(), README, examples; full suite green (N tests)"`

---

## Self-review notes (done while writing)

- **Spec coverage:**
  - §2 layout: T1.
  - §3 conventions: T1, T2, T7.
  - §4 API: T8, T12, T13, T14.
  - §5 data model: T2, T8.
  - §6 fluid: T2, T3; CoolProp is design only.
  - §7 steps: T3.
  - §8 catalogue: T5, T9, T10.
  - §9 solver: T4–T8.
  - §10 results: T8.
  - §11 margins: T11.
  - §12 optimisation: T12.
  - §13 plotting: T13.
  - §14 errors: T1 (classes), T6–T8 (raised).
  - §15 validation: T6, T8, T9, T10.
  - §16 migration: T1, T9 (regression), T14 (example).
- **Type consistency:** `solve_numeric` returns `(values, SolveReport)` (T6) and is consumed by `solve_cycle` (T8).
  `Requirements` fields (T7) are used by `requirements()` (T8) and `optimise` (T12). `CycleDef` fields (T5) are
  used in T9, T10 and T14.
- **Review Focus** items each have a test: 1, 2 and 4 in T7; 3 in T8; 5 in T10.
