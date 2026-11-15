# tests/test_describe_and_examples.py  (plan Task 14, spec section 4)
import runpy, pathlib, pytest
import thermo_cycles as tc

def test_describe_atkinson():
    text = tc.describe("atkinson")
    assert "1→2 isentropic (compression)" in text and "4→1 isobaric (heat_out)" in text
    assert "r_e" in text and "needs 4 inputs" in text and "Rajput" in text

def test_list_cycles_v1():
    assert tc.list_cycles() == ["atkinson", "brayton", "brayton_regen", "diesel", "dual", "otto"]

@pytest.mark.parametrize("script", ["quickstart.py", "old_modes_migrated.py"])
def test_examples_run(script, monkeypatch):
    pytest.importorskip("matplotlib").use("Agg")
    runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / script), run_name="__main__")
