# thermo-cycles

A Python library for ideal thermodynamic cycles. You pick a cycle by name, pass whatever inputs you have, and get
every state, every process step, the standard results, margins to your limits, and T-s / P-v plots with the real
process curves. It is the thermodynamics sibling of
[rocket-analysis-toolkit](https://github.com/J1mmortal/rocket-analysis-toolkit).

The package is `thermo_cycles` (distribution `thermo-cycles`, version 0.1.1, Python ≥ 3.10, MIT licence). Core
dependencies are numpy and scipy. matplotlib is optional and only needed for plots.

## Install

```
pip install "thermo-cycles[plot]"
```

The `[plot]` extra adds matplotlib; leave it out (`pip install thermo-cycles`) if you do not need plots. This line
works once the first release is on PyPI. Until then, or to follow the main branch, install from GitHub:

```
pip install "thermo-cycles[plot] @ git+https://github.com/J1mmortal/thermo-cycles"
```

For development (tests included):

```
git clone https://github.com/J1mmortal/thermo-cycles
cd thermo-cycles
pip install -e .[plot]        # plots; use  pip install -e .[plot,dev]  to run the tests too
python -m pytest -q
```

## Quick start

```python
import thermo_cycles as tc
from thermo_cycles import Cycle, IdealGas, optimise, requirements, describe
from thermo_cycles.units import kPa, bar, degC

tc.list_cycles()            # ['atkinson', 'brayton', 'brayton_regen', 'diesel', 'dual', 'otto']
print(describe("atkinson")) # steps, parameters, degrees of freedom, typical input sets, references

c = Cycle("otto", T1=288, P1=100 * kPa, r_c=10, q_in=1800e3)
c.states[3].T, c.states[3].P    # 3232.1 K, 1.1223e7 Pa
c.results.eta_th, c.results.mep # 0.6019, 1.456e6 Pa
c.value("P4")                   # any solved variable by name
print(c.summary())              # states, steps, results, margins

# any well-posed input set works, including "backwards" ones
Cycle("otto", T1=308, P1=100e3, r_c=9.5, T4=800)            # Cengel and Boles 9-39

# limits give margins (and a warning when violated)
a = Cycle("atkinson", T1=298, P1=1e5, r_c=10, T3=2400, limits={"T_max": 2700, "P_max": 90e5})
a.params["r_e"], a.margins["P_max"]

# another fluid (any two of R, cp, cv, gamma) and a what-if without mutation
b = Cycle("atkinson", fluid=IdealGas(cp=920, gamma=1.22), T1=300, P1=1e5, P2=4e5, P3=16e5)
c2 = c.with_inputs(r_c=12)

# optimisation (replaces the old EFFICIENCY mode)
res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
               fixed=dict(T1=298, P1=101325, T3=2700), limits={"P2_max": 70 * 101325})
res.x, res.value, res.active

# plots (needs the [plot] extra)
c.plot("Ts"); c.plot("Pv", logscale_P=True); tc.plot_cycles([c, c2], "Ts")
```

Runnable versions are in `examples/quickstart.py`, `examples/brayton.py` (gas turbines) and
`examples/old_modes_migrated.py`. Design notes and the full contract are in `docs/spec.md`.

## Which cycle?

- Spark-ignition (petrol) engine -> `otto`; compression-ignition (diesel) engine -> `diesel`, or `dual` for
  part constant-volume, part constant-pressure combustion; over-expanded engine -> `atkinson`.
- Gas turbine -> `brayton` / `brayton_regen` (with a regenerator). Mind the numbering: the turbine inlet is
  **T3** in `brayton` but **T4** in `brayton_regen` (there T3 is the regenerator cold-side outlet).
  `describe(name)` lists the role of every state; `examples/brayton.py` shows both cycles, `optimise` over
  r_p for maximum w_net and maximum eta_th, and the regenerator cut-off pressure ratio.

## Catalogue (v0.1)

| name | system | steps (a->b: kind, role) | parameters | inputs needed | typical inputs |
|---|---|---|---|---|---|
| `otto` | closed | 1->2 isentropic comp; 2->3 isochoric heat_in; 3->4 isentropic exp; 4->1 isochoric heat_out | r_c = v1/v2 | 4 | T1, P1, r_c + one of T3, q_in, P3, T4 |
| `diesel` | closed | 1->2 isentropic; 2->3 **isobaric** heat_in; 3->4 isentropic; 4->1 isochoric heat_out | r_c = v1/v2; cutoff = v3/v2 | 4 | T1, P1, r_c + one of cutoff, T3, q_in |
| `dual` | closed | 1->2 isentropic; 2->3 isochoric heat_in; 3->4 isobaric heat_in; 4->5 isentropic; 5->1 isochoric heat_out | r_c = v1/v2; alpha = P3/P2; cutoff = v4/v3 | 5 | T1, P1, r_c, alpha, cutoff; or T1, P1, r_c, P3, T4; or ..., q_in |
| `atkinson` | closed | 1->2 isentropic; 2->3 isochoric heat_in; 3->4 isentropic exp **to P4 = P1**; 4->1 **isobaric** heat_out | r_c = v1/v2; r_e = v4/v3 | 4 | T1, P1, r_c + one of T3, q_in, P3, r_e |
| `brayton` | flow | 1->2 adiabatic comp (eta_c); 2->3 isobaric heat_in; 3->4 adiabatic exp (eta_t); 4->1 isobaric heat_out (turbine inlet = T3) | r_p = P2/P1; eta_c = eta_t = 1 (defaults) | 4 (+2 defaulted) | T1, P1, r_p, T3 |
| `brayton_regen` | flow | 1->2 adiabatic comp (eta_c); 2->3 isobaric internal (regenerator cold side); 3->4 isobaric heat_in; 4->5 adiabatic exp (eta_t) (turbine inlet = T4; T3 is the regenerator cold outlet, a given T3 >= 1000 K raises InputError); 5->6 isobaric internal (regenerator hot side); 6->1 isobaric heat_out | r_p = P2/P1; eta_c, eta_t (default 1); eps = (h3−h2)/(h5−h2); dT_regen = T5−T3 | 5 (+2 defaulted) | T1, P1, r_p, T4, eps (or dT_regen) |

Notes:
- Dual numbering is 1..5. Cengel and Boles' state x is our 3, his 3 is our 4 and his 4 is our 5.
- Atkinson MEP uses v_max = v4 (not v1), because r_e > r_c.
- `brayton_regen` numbering equals Cengel and Boles 9-110, including the hidden states 2s and 5s.
- Validity bounds: cutoff ≥ 1, alpha ≥ 1, r_c, r_e, r_p ≥ 1 at input but > 1 in effect (r = 1 is rejected as a zero-heat / zero-swept-volume / zero-expansion-work cycle), 0 < eta_c, eta_t ≤ 1, 0 ≤ eps ≤ 1. Atkinson's r_e > r_c is not a declared bound; a violation is caught by the step 2->3 validity check.
- Your own cycle: `register_cycle(CycleDef(...))`, about 20 lines of data and no solver code (spec §8.1):

  ```python
  from thermo_cycles.custom import CycleDef, StepDef, ParamDef, ratio, register_cycle

  register_cycle(CycleDef(
      name="three_step", title="isentropic, isobaric, isochoric", aliases=(), system="closed", states=(1, 2, 3),
      steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
             StepDef("isochoric", 3, 1, "heat_out")),
      params=(ParamDef("r_c", "compression ratio v1/v2", ratio("r_c", "v1", "v2")),),
      v_max_state=1, v_min_state=2))
  Cycle("three_step", T1=293, P1=1e5, r_c=5)
  ```

  `thermo_cycles.custom` also has `linear`, `equal`, `product`, `fixed` and `Relation`.

## Conventions

- **SI everywhere, per unit mass.** T [K], P [Pa], v [m³/kg], u, h [J/kg], s [J/(kg·K)], q, w [J/kg], MEP [Pa].
  Ratios and efficiencies are fractions (0.62, not 62 %). Margins in % are labelled `_pct`.
- `thermo_cycles.units` gives constants (`P1=1*bar`, `T1=degC(25)`). The library never converts silently. A given
  P below 1 000 Pa or T below 100 K raises `InputError`, because that usually means kPa, bar or °C; pass
  `plausibility="warn"` (warning only) or `"off"` when such a value is really meant.
- **Signs:** q > 0 is heat into the fluid, w > 0 is work done by the fluid. Closed steps follow q − w = Δu, flow
  steps q − w = Δh. Compression work is negative. `q_in` and `q_out` are positive magnitudes.
- **States** are numbered from 1 (start of compression). Hidden isentropic end points of non-ideal steps are
  `"2s"`, `"5s"`, as in Cengel and Boles.
- **Reference state:** u = cv·T, h = cp·T, s = cp·ln(T/273.15 K) − R·ln(P/10⁵ Pa).
- **Names:** state properties `T1`, `P3`, `v2`, `u4`, `h3`, `s1` (hidden `T2s`); steps `q_23`, `w_12`; results
  `q_in`, `q_out`, `w_net`, `eta_th`, `mep` (closed), `bwr` (flow); parameters `r_c` (compression ratio),
  `r_e`, `cutoff`, `alpha`, `r_p`, `eta_c`, `eta_t`, `eps`, `dT_regen`. Names resolve exactly first, then
  case-insensitively, then through aliases (`R_C`, `r` -> `r_c`). Note that Cengel and Boles call the cutoff ratio
  r_c; here `r_c` is always the compression ratio and the cutoff ratio is `cutoff`.

## Errors that say what to fix

Wrong physics fails loudly. Not enough information, too much information and nonsense cycles each give their own
error class (`thermo_cycles.errors`), with a message that says what to change:

```
>>> Cycle("atkinson", T1=298, P1=1e5, r_c=10)
UnderdeterminedError: atkinson needs 1 more input.
  Given: T1, P1, r_c  (these already fix: v1, T2, P2, v2, v3, P4)
  Add one of: T3, P3, q_in, r_e, T4, v4, q_out, w_net, eta_th, mep
  See thermo_cycles.requirements('atkinson', ...) for the full list.

>>> Cycle("otto", T1=300, P1=1e5, r_c=10, T3=500)
InvalidCycleError: otto: the solved cycle is not physically valid:
  - step 2->3 (isochoric, heat_in) has q = -181.9 kJ/kg < 0: T3 = 500 K is below T2 = 753.57 K. Raise T3 / q_in or lower r_c.
  ...
```

`requirements("atkinson", T1=298, P1=1e5, r_c=10)` returns the same advice as data. Values that are not finite
real numbers, or lie outside the solvable range, raise `InputError`.

## Validation

The numbers are checked against textbook cases, not only against the code itself (tolerance rtol 3e-3 unless a
test says otherwise), and against closed-form identities (rtol 1e-9):

- Otto: University of Waterloo ME 354 Tutorial wk 8 solution (r_c 10, q_in 1800 kJ/kg); Cengel and Boles 7e
  solutions manual 9-39 (T4 given, a "backwards" input set).
- Diesel: Cengel and Boles 8e SM 9-57 and 7e SM 9-58 (states).
- Dual: Cengel and Boles 7e SM 9-54.
- Atkinson: Rajput, *Engineering Thermodynamics*, Example 13.31 (book p. 659-660), with the looser heat tolerances
  its inconsistent k = 1.22 forces.
- Brayton: thermostate documentation, "Cold-Air Standard Brayton Cycle Example".
- Brayton with regeneration: Cengel and Boles 7e SM 9-110.
- Custom three-step cycle: Cengel and Boles 7e SM 9-24.

The solution-manual pages are McGraw-Hill copyrighted; the tests cite them by problem number and use only the
numbers. Full list and tolerances: `docs/spec.md` §15.1.

## Migrating from the old prototype (the "Atkinson fix")

The old `AtkinsonSolver` actually computed a **Diesel** cycle (isobaric heat addition, isochoric heat rejection).
`atkinson` is now the real ideal Atkinson cycle: isochoric heat addition 2->3, isentropic expansion to P4 = P1 (so
r_e = v4/v3 > r_c), isobaric heat rejection 4->1. What the old code computed is `diesel`, unchanged:

- old DESIGN mode -> `Cycle("diesel", T1=..., P1=..., r_c=..., T3=... or q_in=...)`
- old EFFICIENCY mode -> `optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th", fixed={...}, limits={"P2_max": ...})`

See `examples/old_modes_migrated.py`. The old input interpreter, `BaseCycle` and `AtkinsonCycle` classes are gone;
the general solver replaces the per-cycle mode patterns.

## Licence

MIT, see `LICENSE`.
