"""The two modes of the old ``AtkinsonSolver`` (commit 9c1505b), migrated (spec section 16).

The old class computed a Diesel cycle under the name "Atkinson". DESIGN mode becomes a ``Cycle("diesel", ...)``
and EFFICIENCY mode becomes ``optimise("diesel", ...)``. The real Atkinson cycle is now ``Cycle("atkinson", ...)``.
"""
import thermo_cycles as tc
from thermo_cycles import Cycle, optimise

# Old DESIGN mode: T1, P1, r_c and the peak temperature (or q_in) given.
design = Cycle("diesel", T1=298.0, P1=1e5, r_c=20.0, T3=2480.988)
print(f"DESIGN: eta_th = {design.results.eta_th:.6f}, w_net = {design.results.w_net:.1f} J/kg, "
      f"T2 = {design.states[2].T:.3f} K, T4 = {design.states[4].T:.3f} K")

# Old EFFICIENCY mode: vary r_c for the best efficiency at fixed T1, P1, T3, with a cylinder pressure limit.
P1 = 101325.0
best = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
                fixed=dict(T1=298, P1=P1, T3=2700), limits={"P2_max": 70 * P1})
print(f"EFFICIENCY: r_c = {best.x['r_c']:.3f}, eta_th = {best.value:.4f}, active limits: {best.active}")
print(f"            P2 = {best.cycle.states[2].P / 1e5:.2f} bar (limit {70 * P1 / 1e5:.2f} bar)")

# The cycle the old code was meant to compute:
atkinson = Cycle("atkinson", T1=298.0, P1=1e5, r_c=10.0, T3=2400)
print(f"Atkinson (new): r_e = {atkinson.params['r_e']:.2f}, eta_th = {atkinson.results.eta_th:.4f}")
