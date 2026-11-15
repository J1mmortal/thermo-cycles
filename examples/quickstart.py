"""Quick start: the snippets of spec section 4 with Atkinson and Otto.

Run with ``python examples/quickstart.py``. Needs the ``[plot]`` extra for the figure, which is saved as
``quickstart_Ts.png`` in a temporary directory (the path is printed). All values are SI, per unit mass.
"""
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # this script only saves a file

import thermo_cycles as tc
from thermo_cycles import AIR, Cycle, IdealGas, errors, requirements
from thermo_cycles.units import degC, kPa

print(tc.list_cycles())
print(tc.describe("atkinson"))

# Atkinson: give T1, P1, r_c and the peak temperature; the expansion ratio r_e follows.
atk = Cycle("atkinson", T1=298, P1=1e5, r_c=10, T3=2400, limits={"T_max": 2700, "P_max": 90e5})
print(f"\nAtkinson: T3 = {atk.states[3].T:.0f} K, r_e = {atk.params['r_e']:.2f}, "
      f"eta_th = {atk.results.eta_th:.4f}")
print("P_max margin:", atk.margins["P_max"])

# Otto with kPa and degC helpers (the library never converts silently): Cengel and Boles style data.
otto = Cycle("otto", T1=degC(15), P1=100 * kPa, r_c=10, q_in=1800e3)
print(f"Otto: T3 = {otto.states[3].T:.1f} K, eta_th = {otto.results.eta_th:.4f}, "
      f"mep = {otto.results.mep / 1e3:.1f} kPa")
print("P4 from the solved system:", otto.value("P4"))

# Any well-posed input set works, also "backwards" ones: Cengel 9-39 gives T4 instead of T3.
back = Cycle("otto", T1=308, P1=100e3, r_c=9.5, T4=800)
print(f"backwards Otto: T3 = {back.states[3].T:.0f} K")

# Not enough or too much information is reported with a fix, before any solving.
print(requirements("atkinson", T1=298, P1=1e5, r_c=10))
try:
    Cycle("atkinson", T1=298, P1=1e5, r_c=10)
except errors.UnderdeterminedError as exc:
    print(exc)

# Another fluid, a what-if without mutation, and a plot of both cycles on one T-s diagram.
other = Cycle("atkinson", fluid=IdealGas(cp=920, gamma=1.22), T1=300, P1=1e5, P2=4e5, P3=16e5)
print(f"cp = 920, gamma = 1.22: eta_th = {other.results.eta_th:.3f}")
hot = otto.with_inputs(r_c=12)
ax = tc.plot_cycles([otto, hot], "Ts", labels=["r_c = 10", "r_c = 12"])
out = Path(tempfile.mkdtemp(prefix="thermo_cycles_")) / "quickstart_Ts.png"
ax.figure.savefig(out, dpi=120)
print("saved", out)
