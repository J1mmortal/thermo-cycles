# thermo-cycles v0.1: design spec

Status: draft for Jesper's review, written 2026-11-14 (Trio task T0055). Not verified.
Repo: github.com/J1mmortal/thermo-cycles, reviewed at commit `9c1505b` ("added valueerrors").
Companion: `thermo-cycles-plan.md`, the test-first build plan.

## 0. Summary

`thermo-cycles` becomes an importable Python library (`import thermo_cycles`). You pick a cycle by name from a
built-in catalogue and pass whatever inputs you have. You get back every state, every process step, the standard
results, margins to your limits, and T-s / P-v plots with the real process curves:

```python
from thermo_cycles import Cycle
c = Cycle("atkinson", T1=298, P1=1e5, r_c=10, T3=2400, limits={"T_max": 2700, "P_max": 90e5})
c.states[3].T          # 2400.0
c.results.eta_th       # thermal efficiency
c.params["r_e"]        # expansion ratio that follows from the inputs
c.margins["P_max"]     # Margin(limit=9e6, value=8.054e6, margin=..., margin_pct=..., ok=True)
c.plot("Ts")
```

Four design decisions carry the rest:

1. **Cycles are data.** A cycle is an ordered list of process steps between numbered states, plus named parameters
   (r_c, cutoff, r_p, ...) that are defined by relations between state properties. Adding a cycle means writing
   about 20 lines of catalogue data and no solver code.
2. **One general solver replaces the per-cycle mode patterns.** Every cycle becomes a system of scalar equations:
   fluid equations per state, one relation per process step, the parameter definitions, and one equation per user
   input. A structural analysis (bipartite matching / Dulmage-Mendelsohn) runs before any number is computed. It
   says whether the inputs are enough, too many or contradictory, and it phrases the fix from the inputs you gave.
   A block-triangular ordering (BLT) then solves the system in closed form where it can. It falls back to a scanned
   1-D root find, which also catches multiple solutions, and to Newton only for genuine algebraic loops.
3. **The fluid is pluggable.** The solver only talks to a small `Fluid` protocol. The built-in `IdealGas`
   (constant cp, cv; cold-air standard by default) also exposes its equations in sparse form, so it gets closed
   forms. A CoolProp fluid later only has to implement `state(**two_props)`; the interface stays as it is.
4. **Results, margins, optimisation and plots sit on top of one solved `CycleState`.** Optimisation is a function
   that calls the solver repeatedly. It is not a per-cycle mode.

## 1. Goals and non-goals

Goals (v0.1):
- Python API only. Importable package with a `src/` layout and pyproject, in the same style as
  `rocket-analysis-toolkit`.
- Catalogue v1: Otto, Diesel, Dual, Atkinson, Brayton (simple, with optional isentropic efficiencies), and
  Brayton with regeneration (the stretch goal, included because the step and solver design make it cheap).
- Almost-free input with clear under- and over-determination errors.
- Results: q_in, q_out, w_net, eta_th, MEP (piston cycles), back-work ratio (flow cycles), eta_Carnot, eta_II.
- Margins to any user limit, plus second-law efficiency.
- Optimisation over one or more inputs with limits as constraints.
- T-s and P-v plots with true process curves. matplotlib is an optional dependency.
- Validation against published worked examples plus closed-form property tests.

Non-goals (v0.1): CLI or menu; CoolProp implementation; Stirling, Ericsson, Rankine, refrigeration (designed for
and listed in §8.3); sensitivity analysis; unit parsing (inputs are SI floats); absolute (per-cycle-mass)
quantities; variable specific heats (a later `Fluid`, see §6.4).

## 2. Package name, layout, dependencies

- Distribution name `thermo-cycles`, import name `thermo_cycles`. On 2026-11-14, PyPI returned 404 (name free) for
  `thermo-cycles`, `thermocycles` and `thermo-cycle`. PyPI treats `thermo-cycles` and `thermo_cycles` as the same
  name.
- Python `>=3.10`, the same floor as rocket-analysis-toolkit. Python 3.10.11 is what is installed locally.
- Core dependencies: `numpy>=1.23`, `scipy>=1.10` (root finding, `optimize.root`, `minimize_scalar`, SLSQP,
  `stats.qmc`). Optional: `[plot] = matplotlib>=3.7`, `[dev] = pytest>=7`. `[coolprop]` is reserved for later and
  not declared in v0.1 (see Needs Jesper).
- Build: setuptools, `package-dir = {"" = "src"}`, `packages.find where=["src"]`, licence MIT (kept). This copies
  rocket-analysis-toolkit's pyproject.

```
thermo-cycles/
  pyproject.toml  README.md  LICENSE  .gitignore
  src/thermo_cycles/
    __init__.py          public API re-exports, __version__
    errors.py            exception and warning classes (§14)
    units.py             SI helper constants: kPa, bar, MPa, atm, kJ, L, degC()
    state.py             State (frozen), PROPS
    fluids/__init__.py   IdealGas, AIR, Fluid
    fluids/base.py       Fluid protocol, FluidBase (generic numeric fallbacks)
    fluids/ideal_gas.py  IdealGas
    model/relations.py   Relation + helper constructors
    model/steps.py       process step kinds (isentropic, isochoric, isobaric, isothermal, adiabatic)
    model/cycle_def.py   StepDef, ParamDef, CycleDef, registry (register_cycle, get_cycle, list_cycles)
    catalogue/__init__.py   imports the modules below so the registry is populated
    catalogue/piston.py     otto, diesel, dual, atkinson
    catalogue/gas_turbine.py  brayton, brayton_regen
    solver/__init__.py   solve_cycle()
    solver/system.py     EquationSystem, build_system()
    solver/structure.py  matching, DM partition, BLT blocks (pure graph code)
    solver/numeric.py    block solving: explicit, torn 1-D scan, Newton
    solver/advice.py     Requirements, analyse(), error phrasing
    cycle_state.py       CycleState (the solved container)
    results.py           StepResult, CycleResults, compute_results()
    margins.py           Margin, MarginReport, evaluate_limits()
    cycle.py             Cycle front class, requirements(), describe()
    optimise.py          optimise(), OptimisationResult
    plotting.py          plot_cycle(), plot_cycles() (imports matplotlib lazily)
  tests/                 one test module per source module, plus reference-case modules
```

## 3. Units and conventions

- **SI everywhere, per unit mass.** T [K], P [Pa], v [m³/kg], u, h [J/kg], s [J/(kg·K)], q, w [J/kg], MEP [Pa].
  Ratios and efficiencies are dimensionless fractions (0.62, not 62 %). Margins in % are the one exception and are
  labelled `_pct`.
- `thermo_cycles.units` gives you constants so you can write `P1=1*bar` or `T1=degC(25)`. The library never
  converts silently. A given state pressure below 1 000 Pa or state temperature below 100 K raises `InputError`
  by default, because those usually mean kPa, bar or °C were passed (§9.8); the keyword
  `plausibility="warn"` turns this into `InputPlausibilityWarning` (the v0.1 behaviour) and `"off"` skips it
  (v0.1.1).
- **Signs:** q > 0 is heat into the working fluid; w > 0 is work done by the fluid. Closed (piston) steps follow
  q − w = Δu and flow (steady-flow) steps follow q − w = Δh. Compression work is therefore negative.
  `q_in` and `q_out` are reported as positive magnitudes: q_in = Σ q over heat_in steps and
  q_out = −Σ q over heat_out steps.
- **State numbering:** integers from 1, where state 1 is the start of compression, as in Cengel & Boles. Hidden
  states (the isentropic end points of non-ideal steps) get string ids such as `"2s"`, again following Cengel.
- **Reference state (kept from the current code):** u = cv·T and h = cp·T (zero at 0 K), and
  s = cp·ln(T/273.15 K) − R·ln(P/10⁵ Pa). Only differences matter in the results. Keeping the reference makes the
  old-code regression numbers carry over unchanged (§15.4).
- **Variable names** used for inputs, limits and lookups:
  - state properties: `T1`, `P3`, `v2`, `u4`, `h3`, `s1` (letter + state id; hidden states give `T2s`, ...).
  - step heat and work: `q_23`, `w_12`.
  - cycle results that live in the equation system: `q_in`, `q_out`, `w_net`, `eta_th`, `mep` (closed cycles),
    `bwr` (flow cycles).
  - catalogue parameters: `r_c`, `r_e`, `cutoff`, `alpha`, `r_p`, `eta_c`, `eta_t`, `eps`, `dT_regen`.
  - Look-up order: exact name first, then a unique case-insensitive match (so the old upper-case `R_C` and `T3`
    still work), then the alias table (`r` -> `r_c`). Anything else raises `UnknownInputError` with difflib
    suggestions.
- **Naming clash to note:** Cengel & Boles call the *cutoff* ratio r_c and the dual-cycle constant-volume
  pressure ratio r_p. This package keeps Jesper's `r_c` = compression ratio, which is also the brief's usage. It
  therefore uses `cutoff` and `alpha` for those two. The parameter descriptions say so.

## 4. Public API

```python
import thermo_cycles as tc
from thermo_cycles import Cycle, IdealGas, AIR, optimise, requirements, describe, list_cycles
from thermo_cycles.units import kPa, bar, degC

list_cycles()                       # ['atkinson', 'brayton', 'brayton_regen', 'diesel', 'dual', 'otto']
print(describe("atkinson"))         # steps, state roles, parameters, degrees of freedom, typical inputs, refs

c = Cycle("otto", T1=288, P1=100*kPa, r_c=10, q_in=1800e3)
c.states[3].T, c.states[3].P        # 3232.1 K, 1.1223e7 Pa
c.steps[1]                          # StepResult(a=2, b=3, kind='isochoric', role='heat_in', q=1.8e6, w=0.0, ...)
c.results.eta_th, c.results.mep     # 0.6019, 1.456e6
c.results.eta_carnot, c.results.eta_ii
c.value("P4")                       # any solved variable by name
c.summary()                         # text table of states, steps, results, margins

# any well-posed input set works, including "backwards" ones
Cycle("otto", T1=308, P1=100e3, r_c=9.5, T4=800)            # Cengel 9-39
Cycle("otto", T1=288, r_c=10, q_in=1800e3, P3=11.2166e6)     # P1 solved (algebraic loop)

# not enough / too much information
requirements("atkinson", T1=298, P1=1e5, r_c=10)
# Requirements(status='under', missing=1, add_one_of=('T3','P3','q_in','r_e','T4','v4','q_out','w_net','eta_th','mep'), ...)
Cycle("atkinson", T1=298, P1=1e5, r_c=10)
# UnderdeterminedError: atkinson needs 1 more input.
#   Given: T1, P1, r_c  (these already fix: v1, T2, P2, v2, v3, P4)
#   Add one of: T3, P3, q_in, r_e, T4, v4, q_out, w_net, eta_th, mep

# other fluid (any two of R, cp, cv, gamma)
Cycle("atkinson", fluid=IdealGas(cp=920, gamma=1.22), T1=300, P1=1e5, P2=4e5, P3=16e5)

# what-if without mutating
c2 = c.with_inputs(r_c=12)          # new Cycle; c unchanged. Passing None removes an input.

# optimisation (replaces the old EFFICIENCY mode)
res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
               fixed=dict(T1=298, P1=101325, T3=2700), limits={"P2_max": 70*101325})
res.x, res.value, res.active, res.cycle.states[2].P

# plots (needs the [plot] extra)
ax = c.plot("Ts"); c.plot("Pv", logscale_P=True); tc.plot_cycles([c, c2], "Pv")
```

Signatures (all keyword-only after the name):

```python
class Cycle:
    def __init__(self, name: str, *, fluid: Fluid | None = None, limits: Mapping[str, float] | None = None,
                 allow_redundant: bool = False, check: Literal["raise", "warn", "off"] = "raise",
                 on_violation: Literal["warn", "raise", "ignore"] = "warn",
                 plausibility: Literal["raise", "warn", "off"] = "raise", **inputs: float) -> None
    name: str; definition: CycleDef; fluid: Fluid; inputs: Mapping[str, float]   # canonical names
    states: Mapping[int, State]; hidden_states: Mapping[str, State]
    steps: tuple[StepResult, ...]; results: CycleResults; params: Mapping[str, float]
    margins: MarginReport; cycle_state: CycleState
    def value(self, name: str) -> float
    def with_inputs(self, **changes: float | None) -> "Cycle"      # keeps limits, check, on_violation, plausibility
    def plot(self, kind: Literal["Ts", "Pv", "both"] = "Ts", **kw) -> Any
    def summary(self) -> str
    def to_dict(self) -> dict

def requirements(name: str, *, fluid: Fluid | None = None, plausibility: str = "raise",
                 **inputs: float) -> Requirements
def describe(name: str) -> str
def list_cycles() -> list[str]
def register_cycle(defn: CycleDef, *, replace: bool = False) -> None      # advanced; how the catalogue is built
def optimise(name: str, *, vary: Mapping[str, tuple[float, float]], maximise: str | Callable | None = None,
             minimise: str | Callable | None = None, fixed: Mapping[str, float] = {},
             limits: Mapping[str, float] = {}, fluid: Fluid | None = None, grid: int = 81,
             seed: int = 0, plausibility: Literal["raise", "warn", "off"] = "raise") -> OptimisationResult
# solve_cycle(defn, fluid, inputs, *, allow_redundant=False, check="raise", plausibility="raise")  (internal)
```

`Cycle` solves eagerly in `__init__` and is immutable afterwards. Errors are raised at construction.

## 5. Data model

### 5.1 State (kept and tightened)

The current `State` is a mutable property bag. It keeps upper-case keys and carries gamma/R on every state. The new
`State` is a frozen dataclass holding a complete state. The partially-known state the solver needs is just the
solver's value dict, so the bag is no longer needed for that.

```python
PROPS = ("T", "P", "v", "u", "h", "s")
@dataclass(frozen=True)
class State:
    id: int | str
    T: float; P: float; v: float; u: float; h: float; s: float
    x: float | None = None                 # vapour quality; None for single-phase / ideal gas
    def get(self, name: str) -> float      # case-insensitive, kept from the old API ("t" -> T)
    def has(self, *names: str) -> bool     # kept from the old API
    def as_dict(self) -> dict[str, float]
    def summary(self) -> str               # "State#3(T=2400 K, P=6.53 MPa, v=0.105 m3/kg, ...)"
```

### 5.2 CycleState (kept as the solution container)

```python
@dataclass(frozen=True)
class CycleState:
    definition: CycleDef
    fluid: Fluid
    inputs: Mapping[str, float]            # canonical names, as given
    values: Mapping[str, float]            # every solved variable of the equation system
    states: Mapping[int, State]
    hidden_states: Mapping[str, State]
    steps: tuple[StepResult, ...]
    results: CycleResults
    report: SolveReport                    # blocks, method per block (explicit/torn/newton), redundant checks
```

`Cycle` is the user-facing wrapper around one `CycleState` and adds margins and plotting. The old fields `mode`,
`known` and `internal` disappear: there are no modes, and the inputs are the `known` set.

### 5.3 StepResult and CycleResults

```python
@dataclass(frozen=True)
class StepResult:
    a: int; b: int; kind: str; role: str   # role: compression | expansion | heat_in | heat_out | internal
    q: float; w: float; ds: float          # J/kg, J/kg, J/(kg K); signs per §3
    hidden: str | None = None              # e.g. "2s" for non-ideal compression

@dataclass(frozen=True)
class CycleResults:
    q_in: float; q_out: float; w_net: float; eta_th: float
    mep: float | None                      # closed cycles only
    bwr: float | None                      # flow cycles only
    w_compression: float; w_expansion: float   # sums over steps with those roles (signed)
    T_min: float; T_max: float; P_min: float; P_max: float; v_min: float; v_max: float
    eta_carnot: float                      # 1 - T_min/T_max over all main and hidden states
    eta_ii: float                          # eta_th / eta_carnot
    def as_dict(self) -> dict[str, float | None]
```

## 6. Fluid interface

### 6.1 Protocol

```python
class Fluid(Protocol):
    name: str
    props: tuple[str, ...]                          # ("T","P","v","u","h","s") (+ "x" for two-phase fluids)
    def state(self, **two: float) -> dict[str, float]
        # complete every prop from exactly two independent ones; raises UnsupportedPairError, DegenerateStateError
    def independent(self, a: str, b: str) -> bool   # structural independence (ideal gas: (T,u),(T,h),(u,h) -> False)
    def relations(self, sid: str) -> list[Relation] | None
        # optional sparse equations linking the props of state `sid`; None -> the solver uses the generic closure
```

`FluidBase` supplies the defaults:
- `relations()` returns `None`. The solver then models the state with the **generic closure**: len(props) − 2
  equations, each touching every prop of the state, so the structure is dense. They are solved together by
  calling `state()` with a known independent pair.
- `state()` with an unsupported pair falls back to a 1-D bracketed root find on a supported pair. Example from
  the probe: CoolProp does not accept (T, h). The fallback solves P such that h(T, P) = h_target using the
  supported (T, P) pair.

### 6.2 IdealGas (built in)

```python
class IdealGas(FluidBase):
    def __init__(self, *, R: float | None = None, cp: float | None = None, cv: float | None = None,
                 gamma: float | None = None, name: str = "ideal gas",
                 T_ref: float = 273.15, P_ref: float = 1e5) -> None
AIR = IdealGas(R=287.0, gamma=1.4, name="air (cold-air standard)")    # cp = 1004.5, cv = 717.5 J/(kg K)
```

- Give exactly two of R, cp, cv, gamma. With none given you get AIR's values. Three or four consistent values
  (within 1e-6 relative) are accepted; inconsistent ones raise `FluidError`. Cengel's room-temperature set
  (cp 1.005, cv 0.718, R 0.287 kJ/kg·K, k 1.4) is internally consistent to 0.03 %. The reference tests therefore
  use AIR, and their tolerances (§15) absorb the difference.
- Equations per state: P·v = R·T; u = cv·T; h = cp·T; s = cp·ln(T/T_ref) − R·ln(P/P_ref). Each one can be solved
  explicitly for each of its variables, which is what gives the solver its closed forms.
- `state()` supports all 12 independent pairs of (T, P, v, u, h, s) in closed form. (T, u), (T, h) and (u, h) are
  degenerate and raise `DegenerateStateError`.

### 6.3 CoolProp later (design check, nothing built)

`CoolPropFluid(name)` will implement `state()` through `PropsSI`, with `props` including `x`. `relations()`
returns `None`, so the solver uses the generic closure. `saturation_curve()` is an optional extra method that the
plotting code will use when present. It returns a mapping of equal-length arrays `T` [K], `P` [Pa], `s_f`, `s_g` [J/(kg·K)] and
`v_f`, `v_g` [m³/kg] (saturated liquid and vapour along the dome, in the same order as `T`). This format is the one `plotting.py` reads;
it was assumed by the plotting task, not given by this spec before. A probe against the installed CoolProp 6.6.0 (water, 600 K, 5 MPa) showed:
- These pairs work: (T,P), (P,h), (P,s), (ρ,T), (ρ,P), (ρ,u), (ρ,s), (ρ,h), (h,s), (T,s), (P,u).
- These are rejected: (T,h), (T,u), (s,u). (T,Q) and (P,Q) work for saturation.

So the v0.1 interface already covers what CoolProp needs: unsupported pairs go through the `FluidBase` fallback,
and two-phase degeneracy of (T,P) surfaces as `DegenerateStateError` at solve time. One more requirement carries
over to the step paths: they must sample along pairs that stay valid inside the vapour dome (§7.3). Nothing in the
solver, steps, results or plotting depends on the fluid being an ideal gas.

### 6.4 Later fluids

`IdealGasVariableCp` (NASA polynomials) needs no interface change. It returns `None` from `relations()`, or sparse
relations without explicit inverses, and the solver uses torn 1-D roots. It would make the variable-specific-heat
textbook answers testable, e.g. Cengel Otto 52.3 %.

## 7. Process steps

### 7.1 Step kinds (v0.1)

Each kind supplies the relations that define it, its heat and work as **algebraic expressions of the end states**
(valid for any fluid), and a path for plotting.

| kind | defining relation(s) | q | w (closed) | w (flow) | path sampled along |
|---|---|---|---|---|---|
| `isentropic` | s_b = s_a | 0 | −(u_b−u_a) | −(h_b−h_a) | (P, s), P log-spaced |
| `isochoric` | v_b = v_a | closed u_b−u_a; flow (h_b−h_a) − v(P_b−P_a) | 0 | −v(P_b−P_a) | (v, u), u linear |
| `isobaric` | P_b = P_a | h_b−h_a | P(v_b−v_a) | 0 | (P, h), h linear |
| `isothermal` | T_b = T_a | T(s_b−s_a) | q−(u_b−u_a) | q−(h_b−h_a) | (T, s), s linear |
| `adiabatic` (with efficiency) | see below; hidden state `<b>s` | 0 | −(u_b−u_a) | −(h_b−h_a) | none: dashed straight line |

`adiabatic` with role `compression` and efficiency parameter `eta_c` has three relations: s_bs = s_a,
P_bs = P_b, and (h_bs − h_a) = eta_c·(h_b − h_a). With role `expansion` and `eta_t` the third relation becomes
(h_a − h_b) = eta_t·(h_a − h_bs). The parameter defaults to 1.0, which makes the hidden state coincide with b.

Every kind also adds the two energy relations that define `q_ab` and `w_ab`. They add as many unknowns as
equations, so they do not change the degrees of freedom. As a result the user may give `q_23` or `w_12` as inputs.

The heat q of every v0.1 kind is exact from the end states for any fluid. Isobaric q = Δh and isochoric q = Δu
hold for any simple compressible substance, and isothermal q = T·Δs holds for a reversible process. So no
numerical integration is needed. The paths are used only for plotting and for the path-extreme checks.

### 7.2 Interface

```python
class StepKind(ABC):
    name: str; reversible: bool
    def relations(self, a: str, b: str, *, system: str, role: str, efficiency: str | None) -> list[Relation]
    def hidden_states(self, b: str) -> tuple[str, ...]          # () or (f"{b}s",)
    def path(self, fluid: Fluid, sa: State, sb: State, n: int = 100) -> list[State] | None   # None -> dashed
KINDS: dict[str, StepKind]   # "isentropic", "isochoric", "isobaric", "isothermal", "adiabatic"
```

### 7.3 Paths are two-phase-safe by construction

Each path samples along the pair listed in §7.1, never along (T, P). That is why the same code will draw a
Rankine boiler correctly once a CoolProp fluid exists. For the ideal gas the paths are exact and use the same
`fluid.state()` calls.

### 7.4 Later step kinds

- `isenthalpic`: throttle, h_b = h_a, irreversible, dashed path. Needed for refrigeration.
- `polytropic(n)`: P·vⁿ constant, with w = (P_b v_b − P_a v_a)/(1−n) and q from the energy balance. Cengel 9-58
  becomes a test then.
- Isobaric steps with a pressure-loss fraction.

## 8. Catalogue

### 8.1 Format

Cycles are Python data, so no YAML or TOML dependency and no expression parser is needed. Parameters are built
from small relation helpers (§9.2). One cycle definition:

```python
ATKINSON = CycleDef(
    name="atkinson", title="Ideal Atkinson cycle (air-standard)", aliases=("atk",), system="closed",
    states=(1, 2, 3, 4),
    steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isochoric", 2, 3, "heat_in"),
           StepDef("isentropic", 3, 4, "expansion"),   StepDef("isobaric", 4, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2", ratio("r_c", "v1", "v2")),
            ParamDef("r_e", "expansion ratio v4/v3 (> r_c)", ratio("r_e", "v4", "v3"))),
    v_max_state=4, v_min_state=2,
    suggest=("r_c", "T3", "P3", "q_in", "r_e"),
    typical_inputs=(("T1", "P1", "r_c", "T3"), ("T1", "P1", "r_c", "q_in"), ("T1", "P1", "r_c", "r_e")),
    description="Isentropic compression, constant-volume heat addition, isentropic expansion to P1, "
                "constant-pressure heat rejection.",
    references=("Rajput, Engineering Thermodynamics, sec. 13.8 and Example 13.31",),
)
```

```python
@dataclass(frozen=True)
class StepDef:  kind: str; a: int; b: int; role: str; efficiency: str | None = None
@dataclass(frozen=True)
class ParamDef: name: str; description: str; relation: Relation; default: float | None = None
                bounds: tuple[float, float] | None = None       # validity range, also the scan range
@dataclass(frozen=True)
class CycleDef: name; title; aliases; system: Literal["closed","flow"]; states: tuple[int, ...]
                steps: tuple[StepDef, ...]; params: tuple[ParamDef, ...] = ()
                extra_relations: tuple[Relation, ...] = ()       # couplings, e.g. regenerator energy balance
                kind: Literal["power"] = "power"; v_max_state: int | None = None; v_min_state: int | None = None
                suggest: tuple[str, ...] = (); typical_inputs: tuple[tuple[str, ...], ...] = ()
                description: str = ""; references: tuple[str, ...] = ()
                regenerators: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()   # ((a_c, b_c), (a_h, b_h))
```

`regenerators` declares the regenerator stream pairs, one entry per regenerator: `((a_c, b_c), (a_h, b_h))`. The
cold-side step comes first (the stream the regenerator heats), then the hot-side step (the stream it cools). Steps
are named by their state numbers, as in every message ("2->3"), not by position. A pair therefore cannot be off by
one, it stays right when a loop is renumbered or extended, and a typo is caught because (a, b) must be a step.
Example: `brayton_regen` has `regenerators=(((2, 3), (5, 6)),)`. The §9.7 regenerator rule uses these sides,
never the step order.

Validation in `register_cycle` (and `build_system`):
- the steps form one closed loop through all states in order, and every state appears exactly once as a step
  start;
- parameter names don't clash with property or result names;
- step kinds exist;
- every `regenerators` entry is a pair of two different steps of the cycle, both with role `internal`;
- every `internal` step is in exactly one pair. An internal step in no pair is a definition error, not a silent
  pass.

Degrees of freedom of a closed cycle with N states and N single-relation steps: 2N − N = N. Each non-ideal
adiabatic step adds a hidden state (+2) and two more relations (−2), so it is neutral apart from its efficiency
parameter, which has a default. Each `extra_relation` takes away one.

### 8.2 Catalogue v1

| name | system | steps (a->b: kind, role) | parameters (definition) | DOF | typical inputs |
|---|---|---|---|---|---|
| `otto` | closed | 1->2 isentropic comp; 2->3 isochoric heat_in; 3->4 isentropic exp; 4->1 isochoric heat_out | r_c = v1/v2 | 4 | T1,P1,r_c + one of T3, q_in, P3, T4 |
| `diesel` | closed | 1->2 isentropic; 2->3 **isobaric** heat_in; 3->4 isentropic; 4->1 isochoric heat_out | r_c = v1/v2; cutoff = v3/v2 | 4 | T1,P1,r_c + one of cutoff, T3, q_in |
| `dual` | closed | 1->2 isentropic; 2->3 isochoric heat_in; 3->4 isobaric heat_in; 4->5 isentropic; 5->1 isochoric heat_out | r_c = v1/v2; alpha = P3/P2; cutoff = v4/v3 | 5 | T1,P1,r_c,alpha,cutoff; or T1,P1,r_c,P3,T4; or ...,q_in |
| `atkinson` | closed | 1->2 isentropic; 2->3 isochoric heat_in; 3->4 isentropic exp **to P4 = P1**; 4->1 **isobaric** heat_out | r_c = v1/v2; r_e = v4/v3 | 4 | T1,P1,r_c + one of T3, q_in, P3, r_e |
| `brayton` | flow | 1->2 adiabatic comp (eta_c); 2->3 isobaric heat_in; 3->4 adiabatic exp (eta_t); 4->1 isobaric heat_out | r_p = P2/P1; eta_c = 1, eta_t = 1 (defaults) | 4 (+2 defaulted) | T1,P1,r_p,T3 |
| `brayton_regen` | flow | 1->2 adiabatic comp (eta_c); 2->3 isobaric internal (regen cold side); 3->4 isobaric heat_in; 4->5 adiabatic exp (eta_t); 5->6 isobaric internal (regen hot side); 6->1 isobaric heat_out | r_p = P2/P1; eta_c, eta_t (defaults 1); eps = (h3−h2)/(h5−h2); dT_regen = T5−T3; extra: h3−h2 = h5−h6; regenerators = ((2,3), (5,6)) | 5 (+2 defaulted) | T1,P1,r_p,T4,eps (or dT_regen) |

Notes:
- Dual numbering is 1..5. Cengel's state x is our 3, his 3 is our 4 and his 4 is our 5. The reference tests map
  this explicitly.
- Atkinson MEP uses v_max = v4 (not v1), because r_e > r_c.
- The brayton_regen numbering equals Cengel 9-110's, including the hidden states 2s and 5s. `eps` and `dT_regen`
  are both defined parameters, so the user gives either one.
- `brayton_regen` declares its regenerator as `regenerators=(((2, 3), (5, 6)),)`: cold side 2->3, hot side 5->6.
- Validity bounds: cutoff ≥ 1, alpha ≥ 1, r_c, r_e, r_p ≥ 1 at input but > 1 in effect (r = 1 is rejected as a
  zero-heat / zero-swept-volume / zero-expansion-work cycle, §9.7; the code keeps the closed bounds), 0 < eta_c,
  eta_t ≤ 1, 0 ≤ eps ≤ 1. They feed the post-solve checks (§9.7) and the scan ranges (§9.5).
- Atkinson's r_e > r_c is not a declared bound (bounds are per parameter). A violation is caught by the step 2->3
  validity check.

### 8.3 Later cycles: what each one needs

| cycle | missing pieces |
|---|---|
| Stirling (isothermal compression and expansion, isochoric regeneration) | catalogue entry only. The steps exist (`isothermal`, `isochoric` with role `internal`). Validation case ready: Cengel 7e SM 9-81 (T_H 1788 K, q_in 1275, q_out 212.5, w_net 1063 kJ/kg). |
| Ericsson (isothermal + isobaric regeneration) | catalogue entry only. |
| Miller / over-expanded Atkinson (expansion to P4 > P1, then isochoric + isobaric rejection) | catalogue entry with 5 states. |
| Rankine (pump, boiler, turbine, condenser) | `CoolPropFluid` with `x` as a property; inputs such as `x1=0` (saturated liquid at pump inlet) and `x3`; `saturation_curve()` for the dome on T-s; pump work on liquid. The step kinds are already two-phase-safe (§7.3). |
| Vapour-compression refrigeration / heat pump | CoolProp plus the `isenthalpic` step kind; `CycleDef.kind = "refrigeration"` -> results report COP_R / COP_HP instead of eta_th, and Carnot COP instead of eta_Carnot. |
| Brayton with intercooling / reheat | catalogue entries (more states); no new code. |

## 9. The general solver

### 9.1 The equation system

`build_system(defn, fluid)` creates scalar **variables** and **relations** (equations):

| source | variables | relations |
|---|---|---|
| each state k (main and hidden) | T_k, P_k, v_k, u_k, h_k, s_k | the fluid's 4 relations (ideal gas: sparse and explicit; generic: dense closure) |
| each step a->b | q_ab, w_ab | the kind's defining relation(s) + 2 energy relations (§7.1) |
| each parameter | the parameter | its definition (e.g. r_c·v2 − v1 = 0) |
| cycle | q_in, q_out, w_net, eta_th, mep or bwr | q_in = Σq(heat_in); q_out = −Σq(heat_out); w_net = Σw; eta_th·q_in = w_net; mep·(v_max−v_min) = w_net; bwr·w_expansion = −w_compression |
| catalogue extras | | e.g. h3 − h2 − (h5 − h6) = 0 |
| each input `name=value` | | name − value = 0 (also for defaulted parameters the user did not override) |

Worked count for Otto: 24 state props + 8 step q/w + 5 cycle results + 1 parameter = 38 variables. Without the
inputs there are 16 + 4 + 8 + 5 + 1 = 34 relations, so 4 inputs are needed.

`eta_carnot`, `eta_ii` and the T/P/v extremes involve max/min. They are computed after the solve and cannot be
inputs.

### 9.2 Relations

```python
@dataclass(frozen=True)
class Relation:
    name: str                                   # human-readable, used in messages ("isentropic 1->2")
    vars: tuple[str, ...]
    residual: Callable[..., float]              # residual(*values in vars order) -> float, 0 when satisfied
    solvers: Mapping[str, Callable[..., float]] # var -> explicit inverse taking the other vars as keywords
    scale: float = 1.0                          # typical magnitude, so residuals are dimensionless

# helpers (model/relations.py)
equal(a, b, name)                    # a - b = 0, explicit in both
ratio(out, num, den, name=None)      # out*den - num = 0, explicit in all three
linear(out, terms: dict[str, float], name=None)    # out - sum(c*x) = 0, explicit in all
product(out, a, b, name=None)        # out - a*b = 0, explicit in all (b != 0 / a != 0)
fixed(var, value)                    # var - value = 0 (inputs)
```

### 9.3 Structural analysis (before any numbers)

Variables and relations form a bipartite graph. A relation is linked to the variables it uses. Inputs are
`fixed` relations.

1. **Maximum matching** (Hopcroft-Karp; graphs have fewer than 200 nodes) gives the structural rank.
2. **Dulmage-Mendelsohn partition** from the matching:
   - *under-determined part*: variables reachable by alternating paths from unmatched variables. These are the
     free unknowns.
   - *over-determined part*: relations reachable by alternating paths from unmatched relations. These are the
     surplus equations. The `fixed` (input) relations in this part are the **conflicting inputs**.
   - the rest is the *well-determined* square part.
3. Status: `ok` (perfect matching), `under`, `over`, or both. Over-determination is reported first because it is
   definite.

**Numeric rank refinement (as built).** Structure alone is not physically sufficient: of the structurally "ok"
4-input Otto sets, 19 % are numerically singular (for example `eta_th` given with `r_c`: the Otto efficiency
depends on r_c alone, so the solve returns q_in ≈ 0). So `analyse` takes status, `missing`, `determined`,
`add_one_of` and `drop_one_of` from the generic numeric rank of the relation Jacobian at reference solutions.
The structural matching is the fallback (when no reference solution can be built) and the upper bound.
`Requirements.method` says which layer answered (`"numeric"` or `"structural"`). The structure-only behaviour
remains available as `analyse(..., numeric=False)`.

The structural analysis is exact for structure. It also catches ideal-gas degeneracy: T1 and u1 given together over-determine
state 1, because u1 = cv·T1 is one relation linking two fixed variables. Numerical singularity on a structurally
fine system is handled at solve time (§9.5).

### 9.4 Phrasing the fix (minimal additional inputs)

`analyse(system, given) -> Requirements`:

- **Candidates** are the user-facing names: T_k, P_k, v_k of main states, catalogue parameters, q_in, q_out,
  w_net, eta_th, mep / bwr. u_k, h_k, s_k, q_ab and w_ab are accepted as inputs but only suggested when no other
  candidate works. For the ideal gas, u_k and h_k are always equivalent to T_k, so they are folded into T_k.
- **determined**: candidates the given inputs already fix (they lie in the well-determined part). Adding one of
  them would over-determine. The message lists them as "these already fix: ...".
- **missing** k = number of unmatched variables after a maximum matching. That equals the DOF deficit, except
  when an over-determined part exists elsewhere.
- **k = 1**: `add_one_of` = every candidate c (not given, not determined) such that given ∪ {c} has status `ok`.
  That takes one matching per candidate, under 50 candidates.
- **k = 2**: enumerate candidate pairs (at most ~1 200 matchings, around 0.2 s) and keep the ones that give `ok`.
  `add_one_of` = candidates that appear in at least one completing pair. `example_sets` = up to 6 pairs, ranked.
- **k ≥ 3**: `add_one_of` = candidates that reduce the deficit by one (greedy). `example_sets` = one completing
  set built greedily in ranking order, plus the catalogue's `typical_inputs` that are supersets of the given set.
- **Ranking** follows the catalogue's `suggest` order, then T, P, v by state number, then results (q_in, w_net,
  eta_th, ...), then u/h/s.
- **Over-determined**: `drop_one_of` = conflicting inputs whose removal leaves no over-determined part.

Message templates (exact copy, implemented in `solver/advice.py`):

```
UnderdeterminedError: {cycle} needs {k} more input{s}.
  Given: {given}  (these already fix: {determined_short})
  Add one of: {add_one_of[:10]}{more}
  [k>=2] For example: {example_sets[0]} or {example_sets[1]} ...
  See thermo_cycles.requirements('{cycle}', ...) for the full list.

OverdeterminedError: {cycle} is over-determined: {conflicting} are not independent
  ({n_conflict} inputs for {n_dof} degrees of freedom in that part of the cycle).
  Remove one of: {drop_one_of}  -- or pass allow_redundant=True if the values are consistent.
```

The case from the brief, Atkinson with T1, P1, r_c: the message says "add one of: T3, P3, q_in, r_e, T4, v4,
q_out, w_net, eta_th, mep". T2 and P4 are not offered, because they are already fixed. That makes them "determined"
rather than options.

### 9.5 Numerical solution

1. **BLT ordering.** Take the perfect matching, orient each relation towards its matched variable, and find the
   strongly connected components (Tarjan). This gives blocks in solve order. Most blocks are 1×1.
2. **1×1 block with an explicit inverse**: closed form. With the ideal gas, every relation in the catalogue has
   inverses, but not every forward problem is all 1×1 blocks. Otto(T1, P1, r_c, q_in) has torn (s, v) 2×2 blocks
   at states 2 and 4 (the isentropic step ties s and v of both end states), which step 3 solves by tearing. The
   solver therefore reproduces the old hand-written forward chain, but not with 1×1 blocks alone.
3. **Tearing for blocks larger than 1×1 (algebraic loops).** Try each block variable as the tear t. Check whether
   the other block relations can then be evaluated one after another with explicit inverses, leaving one relation
   as the residual g(t). If so, scan g over the variable's physical range: 400 log-spaced points for positive
   kinds (T 1..1e5 K, P 1..1e10 Pa, v 1e-6..1e4 m³/kg, ratios 1..1e4, efficiencies 1e-6..1). For signed kinds
   (u, h, s, q, w) the scan is asinh-spaced over ±1e9. Every sign change is then refined with `brentq`
   (xtol 1e-12 relative).
   - 0 roots -> `NoSolutionError`. It names the inputs involved, e.g. "w_net = 400 kJ/kg exceeds what T1, P1, T3
     allow (max ≈ 329 kJ/kg near r_c = 7.48)". The maximum |g| sample is reported when it can be found.
     v0.1.1: `solve_cycle` then appends the given inputs at or one step away from the unsolved states (plus
     given parameters touching them and whole-cycle inputs) with the role of the state each sets, derived from
     the steps (`model.roles.state_roles`, also shown by `describe()`), e.g. "T3 = 1250 K is the regenerator
     cold-side outlet in brayton_regen (state 3, before 3->4 heat_in). ... Not solved: the turbine inlet is T4".
   - more than 1 root -> each candidate solution is completed and checked against §9.7. If exactly one is valid it
     is used. Otherwise `AmbiguousSolutionError(variable, candidates)` is raised, for example "r_c = 2.736 or
     20.43; pass r_c explicitly". This is the classic case of w_net given at fixed T_max.
4. **Newton fallback** for blocks that one tear cannot reduce: `scipy.optimize.root(method="hybr")` on the scaled
   residuals. Positive variables are solved in log space. It starts from up to 8 seeds (typical values
   T 300/1500 K, P 1e5/1e7 Pa, ...). Distinct converged solutions are handled as in step 3. If none converges, it
   raises `ConvergenceError` with the block's variables, relations and best residual norm.
5. **Solver guard (`solver/numeric.py`).** A solve gives up with `ConvergenceError` (suggesting `analyse`) when
   the relation evaluations over the whole solve exceed `MAX_WORK` (2e6), when a torn residual is flat (its
   rounding noise would give spurious roots), or when no candidate solution has a full-rank Jacobian (the inputs
   look degenerate). A regular solve of every catalogue cycle reports work far below the budget
   (`test_regular_solve_reports_work_well_below_budget`). If a new cycle comes near the budget, raise
   `MAX_WORK`; never weaken the degenerate checks.
6. **Accuracy target:** every relation's scaled residual < 1e-9 after the solve. This is asserted in tests through
   `CycleState.report.max_residual`.

For a CoolProp fluid the per-state generic closure is a 4×4 block once two props are known. The block is solved by
calling `fluid.state(pair)`. The block solver detects this "closure" pattern before trying tearing.

### 9.6 Redundant and inconsistent inputs

By default, over-determined inputs raise `OverdeterminedError`, which is a definite and clear failure. With
`allow_redundant=True`:
1. Drop inputs from `drop_one_of` in reverse given order until the status is `ok` (the input given last is
   checked against the others).
2. Solve.
3. Check each dropped input against the solution: |given − implied| ≤ 1e-6·max(|given|, 1).
4. Any mismatch raises `InconsistentInputError`, e.g. "r_e = 14 given, but T1, P1, r_c, T3 imply r_e = 15.88".
   This way a redundant-but-consistent input is accepted and a contradictory one is always named.

### 9.7 Physical validity checks (after the solve)

`check="raise"` (default) raises `InvalidCycleError` listing every problem. `"warn"` emits `CycleValidityWarning`
and `"off"` skips the checks, except the defined-results rule below, which raises in every mode. With
`check="off"` the validity problems still filter the candidate roots of an ambiguous solve (§9.5); `check` only
controls the final report.
- All T, P, v > 0 (main and hidden states).
- Role `compression`: P_b > P_a and w < 0. Role `expansion`: P_b < P_a and w > 0.
- Role `heat_in`: q ≥ −1e-9·q_in, so zero is allowed (Dual with alpha = 1 or cutoff = 1 is a valid limiting
  case), and q_in > 0 overall. Role `heat_out`: q ≤ 1e-9·q_in. With zero heat input (next bullet) the scale of
  both tolerances is the largest step |q| or |w|, not q_in.
- Defined results. A divisor counts as zero within 1e-9 of the cycle's scale. That scale is the largest |q| or
  |w| of any step, or the larger of the two volumes for the MEP. The rule has three cases:
  - Zero heat input: |q_in| ≤ 1e-9·scale, so eta_th = w_net/q_in is undefined. The message names each heat_in
    step with its q and the temperatures that coincide (e.g. "T3 = T2 = 689.22 K"), and gives the fix: "r_e must
    exceed r_c", "cutoff = 1 is the degenerate limit", "raise T3 / q_in".
  - Zero swept volume (closed cycles): v_max_state = v_min_state, so the MEP is undefined (e.g. r_c = 1).
  - Zero expansion work (flow cycles): |w_expansion| ≤ 1e-9·scale, so bwr is undefined (e.g. r_p = 1).

  These raise `InvalidCycleError` with every `check` mode, `"off"` included. An undefined result is never
  returned. A divide-by-zero in one of these quotients during the solve gives the same error, not
  `NoSolutionError`.
- Parameters inside their `bounds`.
- Regenerator, for each pair declared in `regenerators` (cold side a_c->b_c, hot side a_h->b_h): T(a_h) ≥ T(b_c),
  T(b_h) ≥ T(a_c) and T(a_h) ≥ T(a_c), each with a relative tolerance of 1e-9. For brayton_regen these are
  T5 ≥ T3, T6 ≥ T2 and T5 ≥ T2. The third condition is needed because at eps = 1 the first two hold with
  equality even when T5 < T2. The internal steps' q must sum to zero within 1e-6 × max(q_in, the largest
  internal |q|).
- Message example: "step 2->3 (isochoric, heat_in) has q = −120.4 kJ/kg < 0: T3 = 600 K is below T2 = 689.2 K.
  Raise T3 / q_in or lower r_c." v0.1.1: the advice is checked against the given inputs by a numeric
  sensitivity (each given input perturbed by 1e-4 relative, nested re-solve, sign of the change of that step's
  q). The generic advice is kept when every given input it names points the helpful way; otherwise it is
  replaced by the given inputs only, each in the helpful direction with the value where q reaches 0, e.g. for
  dual(T1=330, P1=120e3, r_c=15, P3=130e5, T4=2300): "Raise T4 (to >= 2383.3 K), r_c (to >= 15.54) or P1
  (to >= 124.3 kPa), or lower P3 (to <= 12545.5 kPa) or T1 (to <= 318.5 K)."

### 9.8 Input validation

Before analysis:
- Values must be real and finite. NaN, inf and complex values raise `InputError`. Python ints and numpy scalars
  are accepted and converted with `float()`.
- Variables of positive kind must be > 0.
- Names are resolved as in §3. v0.1.1: an unknown name (`UnknownInputError`, also from `Cycle.value`) lists the
  known names grouped (state properties and states, parameters, step q/w, cycle quantities), difflib close
  matches, "<cycle> has no state 9 (states: ...)" for a state number that does not exist, and
  `describe(name)`.
- Range: an input outside the solvable range is an `InputError`, not a warning (T outside 1..1e5 K, P outside
  1..1e10 Pa, v outside 1e-6..1e4 m³/kg, eta_th > 1, r_c < 1, ...: the scan ranges of §9.5 and the parameter
  `bounds`). The message names the input and its range. v0.1.1: for a dimensionless input bounded by 1 and a
  value in (1, 100] it adds "efficiencies and ratios are fractions (0.82, not 82)".
- Plausibility (v0.1.1): a given state pressure P < 1 000 Pa or state temperature T < 100 K (thresholds strict:
  1 000 Pa and 100 K pass) raises `InputError` by default ("otto: P1 = 100 Pa: did you mean 100 kPa?
  thermo_cycles works in SI (write P1=100*kPa or 100*bar with thermo_cycles.units). If 100 Pa is really meant,
  pass plausibility='warn' or plausibility='off'."). The keyword `plausibility` on `Cycle`, `requirements`,
  `optimise` and `solve_cycle` (forwarded by `with_inputs`) chooses `"raise"` (default), `"warn"`
  (`InputPlausibilityWarning`, the v0.1 behaviour) or `"off"`. `optimise` checks the fixed inputs and both
  bounds once before the search and never for the search points (the thresholds are lower bounds, so every
  point between accepted bounds is accepted too).

### 9.9 Why this replaces the mode patterns

The old `InputInterpreter` had to anticipate input combinations as modes. The structural analysis decides
solvability for *any* combination and tells you which inputs would complete it. The BLT ordering produces the same
closed-form forward chain the old code wrote by hand. Nothing in the solver is specific to a cycle.

## 10. Results

All results are computed from the solved values:
- `q_in`, `q_out`, `w_net`, `eta_th`: as defined in §9.1. The tests also assert w_net = q_in − q_out
  (internal steps cancel).
- `mep` = w_net / (v_max_state − v_min_state) for closed cycles, otherwise None.
- `bwr` = −w_compression / w_expansion for flow cycles, otherwise None.
- `eta_carnot` = 1 − T_min/T_max over main and hidden states. For every v0.1 step kind T is monotonic along the
  path, so state extremes are path extremes. A unit test asserts this on sampled paths.
- If q_in, v_max − v_min or w_expansion is zero (§9.7, defined results), the solve raises `InvalidCycleError`.
  No eta_th, mep or bwr is reported.
- `eta_ii` = eta_th / eta_carnot.
- `params`: all catalogue parameters, e.g. `r_e` for Atkinson and `cutoff` for Diesel, whether given or derived.
- `StepResult` per step, with q, w and Δs.

## 11. Margins

`limits` maps a key to a limit value. Key grammar: `<name>_max` or `<name>_min`, where `<name>` is one of:
- a property letter T, P, v, u, h, s -> the extreme over all main and hidden states (e.g. `T_max`, `P_max`);
- any solved variable -> that value (`T3_max`, `P2_max`, `eta_th_min`, `w_net_min`, `r_e_max`, `mep_max`);
- `eta_ii_min`, `eta_carnot_max`.

Property letters (`T_max`) are case-exact; other names are case-insensitive. Unknown keys raise
`UnknownInputError` at construction.

```python
@dataclass(frozen=True)
class Margin:
    key: str; kind: Literal["max", "min"]; limit: float; value: float
    where: int | str | None                 # state id holding the extreme, or None
    margin: float                           # max: limit - value ; min: value - limit (same unit as the limit)
    margin_pct: float                       # 100 * margin / |limit|; nan for a zero limit (undefined)
    ok: bool                                # margin >= 0

class MarginReport(Mapping[str, Margin]):
    violated: tuple[Margin, ...]; ok: bool; worst: Margin | None     # lowest margin_pct; a nan margin_pct ranks last
    second_law: SecondLaw   # eta_th, eta_carnot, eta_ii, gap = eta_carnot - eta_th  (always present)
    def table(self) -> str
```

When a limit is violated, `on_violation="warn"` (default) emits one `LimitViolationWarning` listing all violated
limits; `"raise"` raises `LimitViolationError`; `"ignore"` stays silent. The report is always available. The
second-law part of the margins (brief item b) is always present, with or without limits. Sensitivity analysis is
later work.

## 12. Optimisation

```python
@dataclass(frozen=True)
class OptimisationResult:
    x: Mapping[str, float]; value: float; cycle: Cycle        # full solved cycle at the optimum
    feasible: bool; active: tuple[str, ...]                   # limits with |margin_pct| < 1e-3 at the optimum
    evaluations: int; history: tuple[tuple[Mapping[str, float], float | None, bool], ...]
    message: str                 # v0.1.1: with units, "maximum of w_net = 203.45 kJ/kg at r_p = 7.22093; ..."
    # v0.1.1: repr is one short line (optimum, active, feasible, evaluations); the history stays in .history
```

- Before any search, it checks that `fixed` ∪ `vary` keys is well-posed (`requirements` status `ok`). If not, it
  raises the usual Under- or Overdetermined error.
- The objective is a solved-variable or result name (`eta_th`, `w_net`, `mep`, `eta_ii`, `bwr`, any state prop),
  or a callable `f(cycle) -> float`. Exactly one of `maximise` or `minimise` must be given.
- A point is **infeasible** when the solve raises (no solution, invalid cycle) or a limit is violated.
- **1 variable:** evaluate `grid` points (log-spaced if hi/lo > 10). Take the best feasible point. Refine
  with `minimize_scalar(method="bounded")` over the feasible neighbouring interval (both neighbours feasible: both
  sides; only one feasible: that side). If a neighbour is
  infeasible because of a limit, find the boundary with `brentq` on that limit's margin between the feasible and
  infeasible point. The better of the boundary point and the interior refinement wins.
- **2 or more variables:** a Sobol sample (`scipy.stats.qmc`, 64·d points, `seed`), then SLSQP from the 3 best
  feasible samples, with limits as inequality constraints margin/|limit| ≥ 0. The best result wins.
- No feasible point raises `NoFeasiblePointError`. Its message lists each limit with the smallest violation found. `NoFeasiblePointError.closest` is `dict[limit key, Margin]` (the
  least-violated margin per limit), or `None` when no point solved.
- The optimum's states are the optimum's. This fixes the old bug where the stored states belonged to the last grid
  point.

## 13. Plotting

```python
def plot_cycle(cycle, kind="Ts", *, ax=None, n=100, show_states=True, show_hidden=True,
               logscale_P=False, P_unit="Pa", label=None, arrows=True, **line_kw)  # -> Axes or (Fig, (ax1, ax2))
def plot_cycles(cycles, kind="Ts", *, ax=None, labels=None, **kw)                   # overlay; v0.1.1: state
                                     # labels of the 2nd, 3rd ... cycle are primed (3', 3'') and offset
Cycle.plot(kind="Ts", **kw)   # -> plot_cycle(self, kind, **kw)
```

- Curves come from `StepKind.path()` (§7.1). Isentropes are vertical in T-s, isochores and isobars are the true
  exponential curves in T-s, and the P-v curves follow P·v^γ = const and so on. Nothing is drawn as a straight line
  between states, except irreversible steps. Those are drawn dashed, with the hidden isentropic end state as an
  open marker and a dotted isentrope (the textbook convention).
- State labels sit at the points. Direction arrows sit at mid-step.
- Units on the axes: T [K], s [J/(kg K)] (or kJ with `s_unit="kJ"`), P in `P_unit` (Pa, kPa, bar or MPa),
  v [m³/kg].
- If the fluid has `saturation_curve()` (CoolProp later), the dome is drawn.
- matplotlib is imported only inside `plotting.py` functions. Without it you get `MissingDependencyError`: "plotting
  needs matplotlib: pip install 'thermo-cycles[plot]'". The core never imports matplotlib. A test asserts this by
  checking `sys.modules` after `import thermo_cycles`.

## 14. Errors and warnings (`errors.py`)

```
ThermoCyclesError(Exception)
├── InputError(ThermoCyclesError, ValueError)        # bad user input; `except ValueError` still works
│   ├── UnknownCycleError          (.suggestions)
│   ├── UnknownInputError          (.name, .suggestions, .accepted)
│   ├── UnderdeterminedError       (.missing, .given, .add_one_of, .example_sets, .determined)
│   ├── OverdeterminedError        (.conflicting, .drop_one_of)
│   └── InconsistentInputError     (.name, .given, .implied)
├── NoSolutionError                (.variables, .detail)
├── AmbiguousSolutionError         (.variable, .candidates)
├── ConvergenceError               (.variables, .residual)
├── InvalidCycleError              (.problems: tuple[str, ...])
├── FluidError(ThermoCyclesError, ValueError)
│   ├── UnsupportedPairError
│   └── DegenerateStateError
├── LimitViolationError            (.violated)
├── NoFeasiblePointError           (.closest)
└── MissingDependencyError(ThermoCyclesError, ImportError)
Warnings: LimitViolationWarning, CycleValidityWarning, InputPlausibilityWarning (all UserWarning)
```

## 15. Validation plan

All reference values below were reproduced by an independent cold-air recomputation (`probes/refcheck.py`, output
`probes/refcheck.out`). That recomputation matched 58 of 58 printed values within 0.3 %. The only exception is
Rajput, which matches only with its printed k (see case A1). The library's AIR (cp 1004.5, cv 717.5) differs from
Cengel's 1.005/0.718 by ≤ 0.07 %, so tolerances are rtol 3e-3 unless stated.

### 15.1 Textbook reference cases (tests in `tests/test_reference_*.py`)

| id | cycle | source | inputs | printed results to assert |
|---|---|---|---|---|
| O1 | otto | Univ. Waterloo ME 354 Tutorial wk 8 solution (cold-air, k 1.4, cv 0.718) | r_c 10, T1 288 K, P1 100 kPa, q_in 1800 kJ/kg | T2 723.4 K, P2 2511.9 kPa, T3 3230.4 K, P3 11216.6 kPa, T4 1286 K, P4 **446.5 kPa** (445.6 on the source sheet is a typo, P4 = P1·T4/T1 = 100·1286/288 = 446.5; confirmed by Jesper 2026-11-16), q_out 716.6 kJ/kg, w_net 1083.4 kJ/kg, eta 0.602, MEP 1456.4 kPa |
| O2 | otto | Cengel & Boles 7e solutions manual, problem 9-39 (constant cp) | r_c 9.5, T1 308 K, P1 100 kPa, **T4 800 K** | T2 757.9 K, P2 2338 kPa, T3 1969 K, P3 6072 kPa, eta 0.594, MEP 652 kPa (exercises backward propagation) |
| O3 | otto (loop) | derived from O1 | T1 288, r_c 10, q_in 1800e3, P3 11216.6e3 | P1 = 1e5 within 0.1 % (library gives 99 946 Pa because cv 717.5 vs 718); exercises a torn/Newton block |
| D1 | diesel | Cengel & Boles 8e SM 9-57 (cold-air) | r_c 22, cutoff 1.8, T1 343 K, P1 97 kPa | T2 1181 K, T3 2126 K, T4 781 K. The SM's Q_in line uses 2216 K (typo), so only states are asserted; library q_in 949.1 kJ/kg, eta 0.66884 |
| D2 | diesel | Cengel & Boles 7e SM 9-58, states 1-3 only (its expansion is polytropic n = 1.35) | r_c 20, T1 293 K, P1 95 kPa, T3 2200 K | T2 971.1 K, cutoff 2.265, q_in 1235 kJ/kg |
| U1 | dual | Cengel & Boles 7e SM 9-54 (cp 1.005, cv 0.718, k 1.4) | r_c 14, cutoff 1.2, alpha 1.5, T1 253 K, P1 80 kPa | v1 0.9076 m³/kg, P2 3219 kPa, P3 = P4 4829 kPa, T2 727.1 K, T3 (his Tx) 1091 K, T4 (his T3) 1309 K, T5 (his T4) 490.0 K, q_in 480.4 kJ/kg, q_out 170.2 kJ/kg, eta 0.646 |
| A1 | atkinson | Rajput, Engineering Thermodynamics, Example 13.31 (book p. 659-660) | renumbered to 1 = start of compression: T1 300 K, P1 1 bar, P2 4 bar, P3 16 bar; fluid `IdealGas(cp=920, gamma=1.22)` | T2 385.2 K, T3 1540.8 K, T4 934.9 K (rtol 1e-3); q_out 584.1 kJ/kg (rtol 3e-3); q_in 866.7 kJ/kg (rtol 1e-2); w 282.6 kJ/kg (rtol 2e-2); eta 0.326 (rtol 1.5e-2). The book uses k = 1.22 but cp/cv = 0.92/0.75 = 1.2267, so it is internally inconsistent; the looser heat tolerances are the documented price of that. This case also exercises pressure-only inputs (no r_c) |
| B1 | brayton | thermostate docs, "Cold-Air Standard Brayton Cycle Example" | r_p 10, T1 300 K, P1 1 bar, T3 1400 K; gamma 1.401662 (inferred from its printed T2) | T2 580.34 K, T4 723.71 K, bwr 0.4145, eta 0.4831 |
| B2 | brayton_regen | Cengel & Boles 7e SM 9-110 | r_p 8, T1 293 K, T4 1073 K, eta_c 0.87, eta_t 0.93, **dT_regen 10 K** | T2s 530.8, T2 566.3, T5s 592.3, T5 625.9, T3 615.9, T6 576.3 K; w_net 174.7 kJ/kg; q_in = 394.4 kW / 0.8586 kg/s = 459.4 kJ/kg; also eps = 0.83252 as input gives the same states |
| C1 | custom 3-step (via `register_cycle`) | Cengel & Boles 7e SM 9-24 (R 0.3, cv 0.7 kJ/kg·K; its "Properties" line misprints cv = 0.3) | isentropic 1->2, isobaric 2->3 heat_in, isochoric 3->1 heat_out; r = 5, T1 293 K, fluid `IdealGas(R=300, cv=700)` | T2 584.4 K, T3 2922 K, w_12 −204.0, w_23 701.3, q_23 2338, q_31 −1840.3 kJ/kg, eta 0.213 |

The Cengel & Boles solution-manual pages were read from public course mirrors (Simon Fraser University ENSC 461 and
Old Dominion University MET 330; URLs in Sources). These are McGraw-Hill copyrighted solutions. The tests cite them
by problem number and quote only the numbers. Worth knowing: SM 9-65 (UW mirror) computes the isobaric q of a
Diesel with cv instead of cp, so it was **not** used.

### 15.2 Closed-form property tests (`tests/test_closed_form.py`, parametrised grids, rtol 1e-9)

| cycle | identity | grid |
|---|---|---|
| Otto | eta = 1 − r_c^(1−γ), independent of T1, P1, q_in | γ ∈ {1.3, 1.4, 1.67} × r_c ∈ {4, 8, 12, 20} × q_in ∈ {0.5, 1.8} MJ/kg |
| Diesel | eta = 1 − r_c^(1−γ)·(cutoff^γ − 1)/(γ·(cutoff − 1)) | γ × r_c ∈ {12, 18, 22} × cutoff ∈ {1.5, 2, 3} |
| Dual | eta = 1 − r_c^(1−γ)·(alpha·cutoff^γ − 1)/((alpha − 1) + γ·alpha·(cutoff − 1)) | γ × r_c ∈ {10, 14} × alpha ∈ {1.2, 1.5} × cutoff ∈ {1.2, 1.8} |
| Dual limits | alpha = 1 equals Diesel; cutoff = 1 equals Otto (all states) | 2 cases each |
| Atkinson | eta = 1 − γ·(r_e − r_c)/(r_e^γ − r_c^γ) | γ × r_c ∈ {6, 8, 10} × T3 ∈ {1500, 2400} K |
| Brayton | eta = 1 − r_p^((1−γ)/γ); bwr = (T1/T3)·r_p^((γ−1)/γ) | γ × r_p ∈ {4, 8, 16} × T3 ∈ {1200, 1600} K |
| Brayton ideal regen | eps = 1, eta_c = eta_t = 1: eta = 1 − (T1/T4)·r_p^((γ−1)/γ) | r_p ∈ {3, 5, 8}, T4 1400 K (where T5 > T2) |
| all | w_net = q_in − q_out; Σ Δs = 0; Σ Δu (closed) or Σ Δh (flow) = 0; 0 < eta_th < eta_carnot | every catalogue case above |
| paths | first and last path points equal the end states; s constant on isentropic paths; T monotonic on every path | each step kind |

Probe values used in the plan: Atkinson r_c 8, T1 300, P1 1e5, T3 1800 -> r_e 15.8813, T4 595.550 K,
eta 0.627497, mep 312.25 kPa. Otto at T1 300, P1 1e5, T3 1500 has its w_net maximum at r_c = 5^1.25 = 7.4767
with w_net 328.87 kJ/kg; w_net = 250 kJ/kg has two roots, r_c = 2.7362 and 20.4306.

### 15.3 Solver behaviour tests

- Under-determined messages: Atkinson(T1, P1, r_c) -> `add_one_of` ⊇ {T3, P3, q_in, r_e}, and T2 ∉ add_one_of.
  Otto(T1, P1) -> missing = 2, and (r_c, T3) is in example_sets.
- Over-determined: Atkinson(T1, P1, r_c, r_e, T3) -> drop_one_of ⊇ {r_c, r_e, T3}. Otto(T1, u1, ...) -> conflict
  names T1 and u1.
- Redundant consistent: Otto(T1, P1, r_c, q_in, v2 = v1/r_c) with allow_redundant=True -> solves.
  Inconsistent -> `InconsistentInputError` naming the input with given and implied values.
- Ambiguous: Otto(T1 300, P1 1e5, T3 1500, w_net 250e3) -> `AmbiguousSolutionError` with candidates ≈ (2.7362,
  20.4306). No solution: w_net 400e3 -> `NoSolutionError`.
- Validity: Otto(T1 300, P1 1e5, r_c 10, T3 600) -> `InvalidCycleError` naming step 2->3.

### 15.4 Regression against the old code

The old `AtkinsonSolver` DESIGN mode (commit 9c1505b, run by `probes/oldcode.py`) computes a **Diesel** cycle. For
T1 298 K, P1 1e5 Pa, r_c 20, T3 2480.988 K it gives: T2 987.707297, P2 6 628 908.03, v3 0.107415, T4 1081.960052,
P4 363 073.84, s1 87.464217, eta 0.6250059, w_net 937 509.13 J/kg, MEP 1 153 861.65 Pa.
`Cycle("diesel", ...)` must reproduce these to rtol 1e-6. That proves the reused formulas and the entropy reference
carried over.

## 16. Migration of the existing code

| existing file | fate |
|---|---|
| `src/thermo-cycles/` (hyphen, not importable) | `git mv` to `src/thermo_cycles/`; absolute `from core...` imports become package-relative |
| `core/state.py` (`State` bag) | redesigned in `state.py` (§5.1); `get`/`has`/`summary` kept; upper-casing dropped (case-insensitive `get` instead) |
| `core/cycle_state.py` | kept as a name and idea in `cycle_state.py` (§5.2); `mode`/`known`/`internal` removed |
| `core/ideal_gas.py` (`IdealGasCycleBase`) | split: property formulas -> `fluids/ideal_gas.py`; process formulas -> `model/steps.py`. Kept: u = cvT, h = cpT, s reference 273.15 K / 1e5 Pa, isentropic relations. Fixed: `isochoric_from` no longer case-sensitive (moot), `isobaric_from` no longer silently ignores an unknown `known_prop` (moot: steps are relations now), `heat_supply` typed `str` (removed; q_in is a relation), numpy used only for `log` (core uses `math`) |
| `core/input_interpreter.py` | removed. Mode patterns are replaced by the solver (§9). Its bugs disappear with it: lower-case modes vs upper-case checks, `R_E` read but never required |
| `cycles/atkinson.py` | removed. DESIGN mode ≡ `Cycle("diesel", T1=..., P1=..., r_c=..., T3=... or q_in=...)`, kept as a regression test (§15.4). EFFICIENCY mode ≡ `optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th", fixed={T1, P1, T3: T3_MAX}, limits={"P2_max": P_RATIO*P1})`. Its bugs disappear: states overwritten per iteration, `i_max` unbound, hard-coded P1, empty `P2_MAX` branch. The module-level `test_atkinson_design` with its stray `''''''` string goes too |
| `cycles/base.py` (`BaseCycle`, `AtkinsonCycle`) | replaced by `Cycle` (§4); `AtkinsonCycle` not kept (v0.x, no users) |
| `plotting/__init__.py` (empty) | replaced by `plotting.py` |
| `README.md` (2 lines) | rewritten: install, quick start, catalogue table, conventions, references |
| `LICENSE` (MIT, 2025 Jesper Meijerink), `.gitignore` | kept |

**The Atkinson/Diesel fix.** The new `atkinson` is the ideal Atkinson cycle: isochoric heat addition 2->3,
isentropic expansion to P4 = P1 (so r_e = v4/v3 > r_c), isobaric heat rejection 4->1. The cycle the old code
computed is available as `diesel`, unchanged.

## 17. Open choices (also in REPORT.md, "Needs Jesper")

1. PyPI name `thermo-cycles` (free on 2026-11-14). Recommendation: take it, but publishing is his call and outward.
2. Licence: keep MIT (recommended).
3. Python ≥ 3.10 (recommended; matches rocket-analysis-toolkit).
4. scipy as a core dependency (recommended; the solver and optimiser need it).
5. CoolProp: interface designed now, extra *not declared* until implemented (recommended).
6. Old `AtkinsonCycle` class: drop rather than keep a deprecated alias (recommended; v0.x, no users known).

## 18. Sources

- Repo under change: https://github.com/J1mmortal/thermo-cycles (cloned to `T0055/thermo-cycles`, HEAD `9c1505b`).
  Files read: all of `src/thermo-cycles/**`, README, LICENSE.
- Style reference: https://github.com/J1mmortal/rocket-analysis-toolkit (HEAD `5e23a25`), `pyproject.toml`.
- Waterloo ME 354 Tutorial week 8 (Otto) solution:
  https://www.mhtlab.uwaterloo.ca/courses/me354/tutorials/pdffiles/Winter11_tut/ME354-Tut6sol_w07.pdf (pp. 1-6).
- Cengel & Boles, *Thermodynamics: An Engineering Approach* 7e, solutions manual ch. 9 (McGraw-Hill 2011), via
  SFU ENSC 461 mirror https://www.sfu.ca/~mbahrami/ENSC%20461/Suggested%20Problems/Chapter%209/ : files
  `P 9_24.pdf`, `P 9_39.pdf`, `P 9_54.pdf`, `P 9_58.pdf`, `P 9_81.pdf` (Stirling 9-80E/9-81), `P 9_102.pdf`,
  `P 9_110.pdf`.
- Cengel & Boles 8e solutions manual ch. 9 problems 9-33, 9-36E, 9-46, 9-57 (McGraw-Hill 2015), via ODU MET 330:
  https://sites.wp.odu.edu/met330fall24/wp-content/uploads/sites/37475/2025/03/HW1.3-solutions-1.pdf (pp. 1-4).
- Cengel & Boles 7e SM 9-65 (not used; error noted): https://www.mhtlab.uwaterloo.ca/courses/me354/assignments/pdffiles/cengel/9dash65.pdf
- R.K. Rajput, *Engineering Thermodynamics*, ch. 13 "Gas Power Cycles", sec. 13.8 Atkinson cycle and Example 13.31
  (book pp. 657-660), scan at https://deewanbittal.wordpress.com/wp-content/uploads/2017/12/gas-power-cycles_rajpoot1.pdf
  (PDF pp. 54-57).
- thermostate documentation, "Cold-Air Standard Brayton Cycle Example":
  https://thermostate.readthedocs.io/en/stable/cold-air-brayton-cycle-example.html
- Ohio University (Urieli) Otto Problem 3.8 (constant cv 0.717): https://people.ohio.edu/urieli/thermo/Intro/Chapt.1_6/Chapter3d.html
  (cross-check only; not used as a test because cp is not given).
- Local probes (task folder): `probes/refcheck.py` (+`.out`), `probes/testvals.py`, `probes/oldcode.py`,
  `probes/coolprop_pairs.py`; downloaded PDFs in `probes/dl/`.
- Algorithms: Dulmage & Mendelsohn (1958) "Coverings of bipartite graphs", Can. J. Math. 10:517-534;
  Hopcroft & Karp (1973) SIAM J. Comput. 2(4):225-231; Tarjan (1972) SIAM J. Comput. 1(2):146-160. These are the
  standard structural-analysis tools of equation-based modelling (e.g. Modelica compilers).
