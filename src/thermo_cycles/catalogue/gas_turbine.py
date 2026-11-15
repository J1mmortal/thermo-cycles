"""Gas-turbine (steady-flow) cycles of the catalogue (spec section 8.2): Brayton with component efficiencies and
regenerative Brayton.

Both are air-standard flow cycles: per unit mass, q - w = dh on every step. Compression and expansion are
``adiabatic`` steps with isentropic efficiencies ``eta_c`` and ``eta_t`` (default 1.0, which makes the hidden
isentropic end states ``2s`` / ``4s`` / ``5s`` coincide with the real ones).

``brayton_regen`` follows the numbering of Cengel & Boles 7e SM 9-110: 1 compressor inlet, 2 compressor exit,
3 regenerator cold-side exit (combustor inlet), 4 turbine inlet, 5 turbine exit, 6 regenerator hot-side exit.
The two regenerator streams are ``isobaric`` steps with role ``internal``, declared cold side first (2→3, the
stream being heated) and hot side second (5→6). The validity check of spec 9.7 relies on that order.
"""

from __future__ import annotations

import math

from ..model.cycle_def import CycleDef, ParamDef, StepDef, register_cycle
from ..model.relations import Relation, linear, ratio

_ETA_BOUNDS = (0.0, 1.0)   # spec 8.2: 0 < eta <= 1 (eta = 0 is degenerate and fails in the efficiency relation)


def _effectiveness(eps: str, hc_in: str, hc_out: str, hh_in: str) -> Relation:
    """Regenerator effectiveness ``eps = (hc_out - hc_in) / (hh_in - hc_in)``, as the residual
    ``eps * (hh_in - hc_in) - (hc_out - hc_in) = 0`` with an explicit inverse for every variable."""
    nm = f"{eps} = ({hc_out} - {hc_in})/({hh_in} - {hc_in})"

    def div(x, y, var):
        if y == 0:
            raise ZeroDivisionError(f"relation {nm!r}: cannot solve for {var!r}, the divisor is zero")
        return x / y

    return Relation(
        nm, (eps, hc_in, hc_out, hh_in),
        lambda e, ci, co, hi: e * (hi - ci) - (co - ci),
        {eps: lambda **k: div(k[hc_out] - k[hc_in], k[hh_in] - k[hc_in], eps),
         hc_out: lambda **k: k[hc_in] + k[eps] * (k[hh_in] - k[hc_in]),
         hh_in: lambda **k: k[hc_in] + div(k[hc_out] - k[hc_in], k[eps], hh_in),
         hc_in: lambda **k: div(k[hc_out] - k[eps] * k[hh_in], 1.0 - k[eps], hc_in)},
        1e5)


def _r_p() -> ParamDef:
    return ParamDef("r_p", "pressure ratio P2/P1 (> 1)", ratio("r_p", "P2", "P1"), bounds=(1.0, math.inf))


def _eta(name: str, what: str) -> ParamDef:
    return ParamDef(name, f"isentropic efficiency of the {what} (0 < {name} <= 1)", None, default=1.0,
                    bounds=_ETA_BOUNDS)


BRAYTON = CycleDef(
    name="brayton", title="Brayton cycle (air-standard gas turbine, component efficiencies)", aliases=(),
    system="flow",
    states=(1, 2, 3, 4),
    steps=(StepDef("adiabatic", 1, 2, "compression", "eta_c"), StepDef("isobaric", 2, 3, "heat_in"),
           StepDef("adiabatic", 3, 4, "expansion", "eta_t"), StepDef("isobaric", 4, 1, "heat_out")),
    params=(_r_p(), _eta("eta_c", "compressor"), _eta("eta_t", "turbine")),
    suggest=("r_p", "T3", "q_in", "eta_c", "eta_t"),
    typical_inputs=(("T1", "P1", "r_p", "T3"), ("T1", "P1", "r_p", "q_in")),
    description="Adiabatic compression (eta_c), constant-pressure heat addition, adiabatic expansion (eta_t), "
                "constant-pressure heat rejection. eta_c = eta_t = 1 gives the ideal Brayton cycle.",
    references=('thermostate documentation, "Cold-Air Standard Brayton Cycle Example"',),
)

BRAYTON_REGEN = CycleDef(
    name="brayton_regen", title="Regenerative Brayton cycle (air-standard gas turbine)", aliases=(),
    system="flow",
    states=(1, 2, 3, 4, 5, 6),
    steps=(StepDef("adiabatic", 1, 2, "compression", "eta_c"), StepDef("isobaric", 2, 3, "internal"),
           StepDef("isobaric", 3, 4, "heat_in"), StepDef("adiabatic", 4, 5, "expansion", "eta_t"),
           StepDef("isobaric", 5, 6, "internal"), StepDef("isobaric", 6, 1, "heat_out")),
    params=(_r_p(), _eta("eta_c", "compressor"), _eta("eta_t", "turbine"),
            ParamDef("eps", "regenerator effectiveness (h3 - h2)/(h5 - h2), 0 <= eps <= 1",
                     _effectiveness("eps", "h2", "h3", "h5"), bounds=(0.0, 1.0)),
            ParamDef("dT_regen", "regenerator approach temperature T5 - T3",
                     linear("dT_regen", {"T5": 1.0, "T3": -1.0}, "dT_regen = T5 - T3"))),
    extra_relations=(linear("h3", {"h2": 1.0, "h5": 1.0, "h6": -1.0},
                            "regenerator energy balance h3 - h2 = h5 - h6"),),
    suggest=("r_p", "T4", "eps", "dT_regen", "eta_c", "eta_t"),
    typical_inputs=(("T1", "P1", "r_p", "T4", "eps"), ("T1", "P1", "r_p", "T4", "dT_regen")),
    description="Adiabatic compression (eta_c), regenerator cold side, constant-pressure heat addition, adiabatic "
                "expansion (eta_t), regenerator hot side, constant-pressure heat rejection. The regenerator "
                "passes the heat h5 - h6 of the turbine exhaust to the compressed air (h3 - h2).",
    references=("Cengel & Boles 7e SM 9-110",),
)

register_cycle(BRAYTON, replace=True)
register_cycle(BRAYTON_REGEN, replace=True)
