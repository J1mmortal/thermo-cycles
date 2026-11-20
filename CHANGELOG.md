# Changelog

All notable changes to `thermo-cycles`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.1.1] - unreleased

First release on PyPI. Usability fixes from the first live trial of v0.1, plus packaging and CI.

### Changed
- **Implausible inputs now raise `InputError` by default** (a state temperature below 100 K, which usually means
  degC, or a state pressure below 1 kPa). Pass `plausibility="warn"` or `plausibility="off"` to `Cycle`,
  `solve_cycle`, `optimise` or `requirements` to get the old warning or no check. `with_inputs` forwards the setting.
  `optimise` checks once before the search.
- **`brayton_regen` given `T3` of 1000 K or more raises `InputError`.** In `brayton_regen` T3 is the regenerator
  cold outlet and the turbine inlet is T4, so `T3=1250` is almost always a mix-up with `brayton`. With component
  efficiencies it used to solve silently to a wrong cycle (T4 about 2350 K); with ideal components it gave a
  cryptic no-solution error. The message reads "T3 = 1250 K is the regenerator cold outlet in brayton_regen;
  the turbine inlet is T4. Did you mean T4=1250?". `plausibility="warn"` turns it into a warning, `"off"` skips
  it (the no-solution error then carries the same hint). `describe()` and the README catalogue name the turbine
  inlet state of `brayton` and `brayton_regen`.
- A cycle definition error raises `CycleDefinitionError` (a `ThermoCyclesError` and a `ValueError`).
- No-solution errors name the state role of each input involved, for example "T3 = 1250 K is the regenerator
  cold-side outlet". `describe()` lists the state roles of every state.
- Clearer errors for unknown input names and for percent values given where a fraction is expected; the
  `optimise` result repr is short and shows units; primed labels on the overlay plots.
- Spec section 15.1 O1: P4 = 446.5 kPa (the source sheet had the typo 445.6).

### Added
- `CycleDef`, `StepDef`, `ParamDef` and a `thermo_cycles.custom` namespace are exported for custom cycles.
- `optimise` reports an optimum that sits on a `vary` bound (`at_bound` and a message).
- Gas-turbine example (`examples/brayton.py`), a "Which cycle?" section and the one-line install in the README;
  `help(Cycle)` points to `describe()`.
- Packaging for PyPI: classifiers, keywords, project URLs, PEP 639 licence metadata, sdist with tests, docs and
  examples; GitHub Actions test matrix and a Trusted Publishing release workflow (see `RELEASING.md`).

### Fixed
- Heat-sign advice follows the given inputs (dual cycle with P3 given).
- `brayton_regen` with `eps = 0` is the plain Brayton cycle at any `r_p`.
- Zero heat input behind a singular Jacobian gives the zero-heat `InvalidCycleError`.

## [0.1.0] - not published

First version: ideal cycles (Otto, Diesel, dual, Atkinson, Brayton, Brayton with regeneration), any well-posed input
set solved in either direction, results and margins to limits, `optimise`, `requirements`, `describe`, T-s and P-v
plots with the real process curves, custom cycle definitions. Merged to `main` (PR #1); never uploaded to PyPI.

[0.1.1]: https://github.com/J1mmortal/thermo-cycles/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/J1mmortal/thermo-cycles/releases/tag/v0.1.0
