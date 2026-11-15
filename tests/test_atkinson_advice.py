# tests/test_atkinson_advice.py  (the brief's example message)
import pytest
from thermo_cycles import Cycle
from thermo_cycles.errors import UnderdeterminedError, OverdeterminedError

def test_brief_example():
    with pytest.raises(UnderdeterminedError) as ei:
        Cycle("atkinson", T1=298, P1=1e5, r_c=10)
    assert {"T3", "q_in", "P3", "r_e"} <= set(ei.value.add_one_of)
    assert "T2" not in ei.value.add_one_of and "P4" not in ei.value.add_one_of

def test_r_e_plus_T3_over_determines():
    with pytest.raises(OverdeterminedError) as ei:
        Cycle("atkinson", T1=298, P1=1e5, r_c=10, r_e=14, T3=2000)
    assert {"r_c", "r_e", "T3"} <= set(ei.value.drop_one_of)
