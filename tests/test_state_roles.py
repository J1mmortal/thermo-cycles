"""Fix 2 (v0.1.1): state roles in words, in describe() and in no-solution errors (trial T0089 finding 2)."""

import pytest

from thermo_cycles import Cycle, describe, get_cycle
from thermo_cycles.errors import NoSolutionError
from thermo_cycles.model.roles import state_role, state_roles


def test_roles_brayton_and_regen():
    r = state_roles(get_cycle("brayton"))
    assert r[1] == "compressor inlet (after 4->1 heat_out)"
    assert r[2] == "compressor outlet (before 2->3 heat_in)"
    assert r[3] == "turbine inlet (after 2->3 heat_in)"
    assert r[4] == "turbine outlet (before 4->1 heat_out)"
    g = state_roles(get_cycle("brayton_regen"))
    assert g[3] == "regenerator cold-side outlet (before 3->4 heat_in)"
    assert g[4] == "turbine inlet (after 3->4 heat_in)"
    assert g[5] == "turbine outlet (before 5->6 regenerator hot side)"
    assert g[6] == "regenerator hot-side outlet (before 6->1 heat_out)"
    assert g[2] == "compressor outlet (before 2->3 regenerator cold side)"


def test_roles_piston():
    o = state_roles(get_cycle("otto"))
    assert o[3] == "start of expansion (after 2->3 heat_in)"
    assert o[1] == "start of compression (after 4->1 heat_out)"
    d = state_roles(get_cycle("dual"))
    assert d[3] == "end of isochoric heat addition (before 3->4 heat_in)"
    assert d[4] == "start of expansion (after 3->4 heat_in)"
    assert state_role(get_cycle("dual"), "3") == d[3]


def test_describe_lists_state_roles():
    text = describe("brayton_regen")
    assert "State roles:" in text
    assert "  4: turbine inlet (after 3->4 heat_in)" in text
    assert "  3: regenerator cold-side outlet (before 3->4 heat_in)" in text
    assert "  3: turbine inlet (after 2->3 heat_in)" in describe("brayton")


def test_trial_call_names_the_input_and_its_role():
    with pytest.raises(NoSolutionError) as ei:
        Cycle("brayton_regen", T1=288.15, P1=101300, r_p=8, T3=1250, eps=.8, plausibility="off")
    msg = str(ei.value)
    # the original detail stays first
    assert msg.startswith("brayton_regen: no solution. relation 'h(T) 4' cannot be evaluated")
    assert "T3 = 1250 K is the regenerator cold-side outlet in brayton_regen (state 3" in msg
    assert "the turbine inlet is T4" in msg
    assert "eps = 0.8 is the regenerator effectiveness" in msg
    assert "describe('brayton_regen')" in msg
    assert "T1 = " not in msg.split("\n", 1)[1]          # far-away inputs are not blamed


def test_piston_case_names_the_input_and_its_role():
    with pytest.raises(NoSolutionError) as ei:
        Cycle("otto", T1=300, P1=1e5, T3=2000, w_net=2e6)
    msg = str(ei.value)
    assert msg.startswith("otto: no solution. w_net = 2000 kJ/kg exceeds what T1, P1, T3 allow")
    assert "T3 = 2000 K is the start of expansion in otto (state 3, after 2->3 heat_in)" in msg
    assert "T1 = 300 K is the start of compression in otto" in msg
    assert "w_net = 2000 kJ/kg is a whole-cycle quantity" in msg
    assert "describe('otto')" in msg
