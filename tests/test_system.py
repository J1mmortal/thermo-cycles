# tests/test_system.py
from thermo_cycles.fluids import AIR
from thermo_cycles.model.cycle_def import get_cycle
from thermo_cycles.solver.system import build_system

def test_otto_counts():
    sys_ = build_system(get_cycle("otto"), AIR)
    assert len(sys_.variables) == 38            # spec §9.1 worked count
    assert len(sys_.relations) == 34            # -> 4 inputs needed
    assert {"T1", "q_23", "w_41", "q_in", "q_out", "w_net", "eta_th", "mep", "r_c"} <= set(sys_.variables)
    assert "bwr" not in sys_.variables
    assert sys_.var_kind["T3"] == "positive" and sys_.var_kind["q_23"] == "signed"
    assert sys_.candidates[0] == "r_c" and "u3" not in sys_.candidates[:12]
    assert sys_.equivalent["u3"] == "T3" and sys_.equivalent["h3"] == "T3"

def test_with_inputs_appends_fixed():
    sys_ = build_system(get_cycle("otto"), AIR)
    rels = sys_.with_inputs({"T1": 288.0, "P1": 1e5, "r_c": 10.0, "q_in": 1.8e6})
    assert len(rels) == 38
