"""Regenerative isothermal cycles of the catalogue (spec section 8.2, v0.2): Stirling and Ericsson.

Both are air-standard cycles with isothermal compression at T_L and isothermal expansion at T_H, joined by two
regenerator streams: constant volume (Stirling) or constant pressure (Ericsson). With perfect regeneration they
reach the Carnot efficiency 1 - T_L/T_H.

The numbering follows the catalogue convention (state 1 = start of compression) and puts the regenerator where
``brayton_regen`` has it, so both cycles declare ``regenerators=(((2, 3), (5, 6)),)``:

- 1->2 isothermal compression at T_L (role ``heat_out``: it rejects the heat q_out = R T_L ln r),
- 2->3 regenerator cold side (role ``internal``),
- 3->4 external heating to T_H (role ``heat_in``; it carries the heat the regenerator does not supply and is a
  zero step at eps = 1),
- 4->5 isothermal expansion at T_H (role ``heat_in``),
- 5->6 regenerator hot side (role ``internal``),
- 6->1 external cooling to T_L (role ``heat_out``; zero at eps = 1).

The regenerator effectiveness is the ``brayton_regen`` relation eps = (e3 - e2)/(e5 - e2) on the energy the
streams carry (e = u at constant volume, h at constant pressure), with the energy balance e3 - e2 = e5 - e6. It
defaults to 1 (the ideal cycles of the textbooks).

Roles carry one meaning each: the isothermal steps are tagged with their heat role (heat_out / heat_in), because
q_in and q_out are sums over those roles. Their work still counts in w_net. Both cycles are therefore ``closed``
(piston) cycles with an MEP. A steady-flow Ericsson (isothermal compressor and turbine) would need its isothermal
steps in both the heat and the work tallies (for bwr); the step model has one role per step, so that variant is
not offered. Per unit mass its q_in, q_out, w_net, eta_th and states are the same as the closed cycle's.
"""

from __future__ import annotations

import math

from ..model.cycle_def import CycleDef, ParamDef, StepDef, register_cycle
from ..model.relations import linear, ratio
from .gas_turbine import _effectiveness


def _regen(e: str) -> "tuple[ParamDef, object]":
    """The effectiveness parameter (default 1) and the regenerator energy balance, in energy letter ``e``."""
    eps = ParamDef("eps", f"regenerator effectiveness ({e}3 - {e}2)/({e}5 - {e}2), 0 <= eps <= 1 (1: ideal)",
                   _effectiveness("eps", f"{e}2", f"{e}3", f"{e}5"), default=1.0, bounds=(0.0, 1.0))
    balance = linear(f"{e}3", {f"{e}2": 1.0, f"{e}5": 1.0, f"{e}6": -1.0},
                     f"regenerator energy balance {e}3 - {e}2 = {e}5 - {e}6")
    return eps, balance


_STIRLING_EPS, _STIRLING_BALANCE = _regen("u")

STIRLING = CycleDef(
    name="stirling", title="Ideal Stirling cycle (air-standard, regenerator effectiveness eps)", aliases=(),
    system="closed",
    states=(1, 2, 3, 4, 5, 6),
    steps=(StepDef("isothermal", 1, 2, "heat_out"), StepDef("isochoric", 2, 3, "internal"),
           StepDef("isochoric", 3, 4, "heat_in"), StepDef("isothermal", 4, 5, "heat_in"),
           StepDef("isochoric", 5, 6, "internal"), StepDef("isochoric", 6, 1, "heat_out")),
    params=(ParamDef("r_c", "compression ratio v1/v2 (> 1)", ratio("r_c", "v1", "v2"), bounds=(1.0, math.inf)),
            _STIRLING_EPS),
    extra_relations=(_STIRLING_BALANCE,),
    v_max_state=1, v_min_state=2,
    suggest=("r_c", "T4", "q_in", "P4", "eps"),
    typical_inputs=(("T1", "P1", "r_c", "T4"), ("T1", "P1", "r_c", "P4"), ("T1", "P1", "r_c", "q_in"),
                    ("T1", "P1", "r_c", "T4", "eps")),
    description="Isothermal compression at T_L (1->2, heat rejected), constant-volume regenerator cold side "
                "(2->3), constant-volume external heating to T_H (3->4, zero at eps = 1), isothermal expansion at "
                "T_H (4->5, heat added), constant-volume regenerator hot side (5->6), constant-volume external "
                "cooling to T_L (6->1, zero at eps = 1). eps = 1 (default) is the ideal cycle with Carnot "
                "efficiency 1 - T1/T4. Cengel's states 1, 2, 3, 4 are our 4, 5, 1, 2.",
    references=("Cengel & Boles 7e SM 9-81", "Cengel & Boles 7e SM 9-80E"),
    regenerators=(((2, 3), (5, 6)),),
)

register_cycle(STIRLING, replace=True)
