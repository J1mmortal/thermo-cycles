import dataclasses, pytest
from thermo_cycles.state import State, PROPS

def test_state_frozen_and_accessors():
    s = State(id=3, T=2400.0, P=6.5e6, v=0.1, u=1.7e6, h=2.4e6, s=1000.0)
    assert PROPS == ("T", "P", "v", "u", "h", "s")
    assert s.get("t") == 2400.0 and s.has("T", "P") and not s.has("x_quality")
    assert s.as_dict()["P"] == 6.5e6 and s.x is None
    assert "State#3" in s.summary()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.T = 1.0
