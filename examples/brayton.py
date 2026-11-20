"""Gas turbines: the simple and the regenerative Brayton cycle, optimised pressure ratios and the regenerator
cut-off.

Run with ``python examples/brayton.py``. All values are SI, per unit mass (J/kg, K, Pa). The numbering differs
between the two cycles, so check it with ``describe()``:

- ``brayton``:        1 compressor inlet, 2 compressor outlet, 3 TURBINE INLET, 4 turbine outlet.
- ``brayton_regen``:  1 compressor inlet, 2 compressor outlet, 3 regenerator cold-side outlet (combustor inlet),
                      4 TURBINE INLET, 5 turbine outlet, 6 regenerator hot-side outlet.

So the turbine inlet temperature is ``T3`` in ``brayton`` and ``T4`` in ``brayton_regen``.
"""
import thermo_cycles as tc
from thermo_cycles import Cycle, errors, optimise
from thermo_cycles.units import degC, kPa

T1, P1 = degC(15), 101.3 * kPa          # ambient air: 288.15 K, 101 300 Pa
TIT = 1250.0                            # turbine inlet temperature [K]
ETA_C, ETA_T = 0.85, 0.88               # isentropic efficiencies (fractions, not percent)
EPS = 0.8                               # regenerator effectiveness

print(tc.describe("brayton"))
print()

# 1. Simple cycle: the turbine inlet is T3.
simple = Cycle("brayton", T1=T1, P1=P1, r_p=8, T3=TIT, eta_c=ETA_C, eta_t=ETA_T)
r = simple.results
print(f"brayton, r_p = 8: w_net = {r.w_net / 1e3:.2f} kJ/kg, eta_th = {r.eta_th:.4f}, bwr = {r.bwr:.4f}")

# 2. Regenerative cycle at the same pressure ratio: the turbine inlet is T4 here.
regen = Cycle("brayton_regen", T1=T1, P1=P1, r_p=8, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T)
g = regen.results
print(f"brayton_regen, r_p = 8: w_net = {g.w_net / 1e3:.2f} kJ/kg, eta_th = {g.eta_th:.4f} "
      f"(T3 = {regen.states[3].T:.1f} K leaves the regenerator, T4 = {regen.states[4].T:.1f} K enters the turbine)")

# Giving the turbine inlet as T3 in brayton_regen means something else: T3 is the regenerator cold-side outlet.
# With component efficiencies that call would even solve, as a different (much hotter) cycle, so a given T3
# of 1000 K or more raises InputError. plausibility="warn" turns it into a warning, "off" skips the guard.
try:
    Cycle("brayton_regen", T1=T1, P1=P1, r_p=8, T3=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T)
except errors.InputError as exc:
    print(f"\nT3 given as turbine inlet in brayton_regen -> {type(exc).__name__}:\n{exc}")
# With the guard off and ideal components no T4 fits, and the error says which state T3 is.
try:
    Cycle("brayton_regen", T1=T1, P1=P1, r_p=8, T3=TIT, eps=EPS, plausibility="off")
except errors.NoSolutionError as exc:
    print(f"\nSame call, guard off (ideal) -> {type(exc).__name__}:\n{exc}\n")

# 3. Optimise the pressure ratio of the simple cycle for maximum specific work and for maximum efficiency.
fixed = dict(T1=T1, P1=P1, T3=TIT, eta_c=ETA_C, eta_t=ETA_T)
best_w = optimise("brayton", vary={"r_p": (1.5, 40)}, maximise="w_net", fixed=fixed)
best_eta = optimise("brayton", vary={"r_p": (1.5, 40)}, maximise="eta_th", fixed=fixed)
print(best_w.message)
print(best_eta.message)
print(best_w)                           # one short line; the evaluation history is in best_w.history

# 4. The regenerator cut-off: above some r_p the turbine exhaust (T5) is colder than the compressed air (T2), so
#    a regenerator would heat the exhaust instead of the air. thermo_cycles refuses such a cycle and says why.
lo, hi = 2.0, 40.0
for _ in range(60):                     # bisection on "does brayton_regen solve" (with eps > 0)
    mid = 0.5 * (lo + hi)
    try:
        Cycle("brayton_regen", T1=T1, P1=P1, r_p=mid, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T)
        lo = mid
    except errors.InvalidCycleError:
        hi = mid
at = Cycle("brayton_regen", T1=T1, P1=P1, r_p=lo, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T)
print(f"\nregenerator cut-off: r_p = {lo:.4f} (T2 = {at.states[2].T:.2f} K, T5 = {at.states[5].T:.2f} K)")
try:
    Cycle("brayton_regen", T1=T1, P1=P1, r_p=lo * 1.05, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T)
except errors.InvalidCycleError as exc:
    print(f"above it -> {type(exc).__name__}: {str(exc).splitlines()[1].strip()}")

# 5. Regenerative cycle: best efficiency below the cut-off (r_p bounds that cross it are fine: infeasible points
#    are skipped).
best_regen = optimise("brayton_regen", vary={"r_p": (1.5, 40)}, maximise="eta_th",
                      fixed=dict(T1=T1, P1=P1, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T))
print(best_regen.message)

print("\n  r_p   eta simple  eta regen   w simple [kJ/kg]  w regen [kJ/kg]")
for rp in (3, 4, 6, 8, 10, 12):
    s = Cycle("brayton", r_p=rp, **fixed)
    try:
        q = Cycle("brayton_regen", T1=T1, P1=P1, r_p=rp, T4=TIT, eps=EPS, eta_c=ETA_C, eta_t=ETA_T).results
        regen_eta, regen_w = f"{q.eta_th:10.4f}", f"{q.w_net / 1e3:16.1f}"
    except errors.InvalidCycleError:
        regen_eta, regen_w = f"{'cut-off':>10}", f"{'cut-off':>16}"
    print(f"  {rp:4.0f} {s.results.eta_th:10.4f} {regen_eta}   {s.results.w_net / 1e3:14.1f} {regen_w}")
