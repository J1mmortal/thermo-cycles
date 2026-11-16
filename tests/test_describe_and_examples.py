# tests/test_describe_and_examples.py  (plan Task 14, spec section 4)
import runpy, pathlib, pytest
import thermo_cycles as tc

def test_describe_atkinson():
    text = tc.describe("atkinson")
    assert "1->2 isentropic (compression)" in text and "4->1 isobaric (heat_out)" in text
    assert "r_e" in text and "needs 4 inputs" in text and "Rajput" in text

def test_list_cycles_v1():
    assert tc.list_cycles() == ["atkinson", "brayton", "brayton_regen", "diesel", "dual", "otto", "stirling"]

EXAMPLES = ["quickstart.py", "old_modes_migrated.py", "brayton.py"]


@pytest.mark.parametrize("script", EXAMPLES)
def test_examples_run(script, monkeypatch):
    pytest.importorskip("matplotlib").use("Agg")
    runpy.run_path(str(pathlib.Path(__file__).parents[1] / "examples" / script), run_name="__main__")


@pytest.mark.parametrize("script", EXAMPLES)
def test_examples_survive_cp1252_pipe(script):
    """Printed output must be ASCII-safe: a cp1252 pipe or redirected file must not raise UnicodeEncodeError."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    path = pathlib.Path(__file__).parents[1] / "examples" / script
    r = subprocess.run([sys.executable, str(path)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")


def test_brayton_example_runs_with_warnings_as_errors():
    """v0.1.1: examples/brayton.py under ``-W error`` and a cp1252 stdout (no warning, ASCII-safe)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    path = pathlib.Path(__file__).parents[1] / "examples" / "brayton.py"
    r = subprocess.run([sys.executable, "-W", "error", str(path)], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=300)
    assert r.returncode == 0, r.stderr.decode("cp1252", "replace")
    out = r.stdout.decode("cp1252")
    assert "regenerator cut-off: r_p = 13.91" in out
    assert "the turbine inlet is T4" in out


README = (pathlib.Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")


def test_readme_which_cycle_and_one_line_install():
    assert "Which cycle?" in README
    assert "`brayton` / `brayton_regen`" in README
    assert 'pip install "thermo-cycles[plot] @ git+https://github.com/J1mmortal/thermo-cycles"' in README
    assert "examples/brayton.py" in README


def test_help_cycle_points_to_describe_and_readme():
    doc = tc.Cycle.__doc__
    assert "describe(" in doc and "README" in doc
    assert "spec section 4" not in doc
