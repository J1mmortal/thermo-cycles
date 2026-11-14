# thermo-cycles

A Python library for ideal thermodynamic cycles. You pick a cycle by name, pass whatever inputs you have, and
get every state, every process step, the standard results, margins to your limits, and T-s / P-v plots. It is
the thermodynamics sibling of [rocket-analysis-toolkit](https://github.com/J1mmortal/rocket-analysis-toolkit).

> **Status: early work in progress, being rebuilt.** The current code is a first prototype and is not usable
> as a package yet. The sections below say what is here now and what v0.1 will be.

## What it is now (prototype)

- `src/thermo-cycles/`: a first structure that the rebuild keeps:
  - `core/state.py`: a `State` property bag;
  - `core/ideal_gas.py`: reusable ideal-gas process steps;
  - `core/cycle_state.py`: a `CycleState` container;
  - `core/input_interpreter.py`: an input interpreter that infers what to compute from the inputs given;
  - `cycles/atkinson.py`: one cycle front class.
- Known limitations of the prototype:
  - It is not importable. The folder name has a hyphen, there is no `pyproject.toml`, and it uses absolute
    imports such as `from core ...`.
  - The `atkinson` cycle actually computes a **Diesel** cycle: isobaric heat addition and expansion back to V1.
    An ideal Atkinson cycle adds heat at constant volume, expands to P1 and rejects heat at constant pressure.
  - The public solve path fails: the mode names do not match between the interpreter and the solver.
  - The efficiency mode stores the states of the last ratio it tried, not the optimum.
  - Plotting is empty, and there are no tests yet.

## What it will be (v0.1, planned)

Python API only, importable as `thermo_cycles`, with a `src/` layout and pyproject in the same style as
rocket-analysis-toolkit.

```python
from thermo_cycles import Cycle, optimise, requirements

c = Cycle("atkinson", T1=298, P1=1e5, r_c=10, T3=2400, limits={"T_max": 2700, "P_max": 80e5})
c.states[3].T        # any state property
c.results.eta_th     # thermal efficiency (also w_net, q_in, q_out, MEP, eta_Carnot, eta_II, ...)
c.margins["P_max"]   # absolute and % margin to your limit, flagged when violated
c.plot("Ts")         # T-s or P-v plot with the real process curves (matplotlib optional)

requirements("atkinson", T1=298, P1=1e5, r_c=10)
# not enough inputs -> "Add one of: T3, P3, q_in, r_e, ..."

optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
         fixed=dict(T1=298, P1=101325, T3=2700), limits={"P2_max": 70 * 101325})
```

Design:
- **A catalogue of cycles, defined as data.** Each cycle is an ordered list of process steps (isentropic,
  isochoric, isobaric, ...) plus named parameters such as the compression ratio `r_c`. You pick a cycle by name
  and never write steps yourself. A new cycle costs a few lines of catalogue data, not solver code.
- **Almost-free input and one general solver.** Give any parameters you have (T1, P1, r_c, q_in, T3, cutoff, ...).
  The solver first checks whether your inputs are enough, too many or contradictory. If you gave too few, it
  tells you which minimal extra inputs would complete the set, phrased from what you gave. It solves in closed
  form where possible and numerically where not, and it reports when an input set has more than one solution.
- **Optimisation** (for example the best compression ratio within a peak-pressure limit) is a separate function
  on top of the solver.
- **Margins:** the margin to any limit you set on any property, and the efficiency compared with Carnot between
  the cycle's lowest and highest temperatures (second-law efficiency).
- **Pluggable fluid:** ideal gas with constant specific heats (cold-air standard) built in. The fluid interface is
  designed so that CoolProp can be added later for phase-change cycles.
- **Units:** SI, per unit mass.

Catalogue v0.1: Otto, Diesel, Dual, Atkinson, Brayton (with optional component efficiencies), Brayton with
regeneration.

Later: Stirling, Ericsson, Miller, Brayton with intercooling or reheat, and via CoolProp, Rankine and
vapour-compression refrigeration.

Validation: tests against published textbook worked examples and closed-form efficiency checks for each cycle.

## Requirements (planned)

Python 3.10 or newer, numpy and scipy. matplotlib is optional, for the plots.

## Licence

MIT, see [LICENSE](LICENSE).
