# tests/test_cycle_def.py
import pytest
import thermo_cycles
from thermo_cycles.model.cycle_def import get_cycle, list_cycles, register_cycle, CycleDef, StepDef
from thermo_cycles.errors import UnknownCycleError

def test_otto_registered():
    d = get_cycle("OTTO")
    assert d.name == "otto" and d.system == "closed" and d.states == (1, 2, 3, 4)
    assert [s.kind for s in d.steps] == ["isentropic", "isochoric", "isentropic", "isochoric"]
    assert [s.role for s in d.steps] == ["compression", "heat_in", "expansion", "heat_out"]
    assert [p.name for p in d.params] == ["r_c"] and (d.v_max_state, d.v_min_state) == (1, 2)
    assert "otto" in list_cycles()

def test_unknown_cycle_suggests():
    with pytest.raises(UnknownCycleError) as ei:
        get_cycle("oto")
    assert "otto" in ei.value.suggestions

def test_register_rejects_open_loop():
    bad = CycleDef(name="bad", title="", aliases=(), system="closed", states=(1, 2, 3),
                   steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in")))
    with pytest.raises(ValueError):
        register_cycle(bad)
