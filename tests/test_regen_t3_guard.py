"""brayton_regen given T3 >= 1000 K: T3 is the regenerator cold outlet, the turbine inlet is T4 (T0089 finding 2)."""
import pytest
import thermo_cycles as tc
from thermo_cycles.errors import InputError, InputPlausibilityWarning, NoSolutionError

BASE = dict(T1=300.0, P1=100e3, r_p=8.0, eps=0.75)
MSG = ("T3 = 1250 K is the regenerator cold outlet in brayton_regen; the turbine inlet is T4. "
       "Did you mean T4=1250?")
NONIDEAL = dict(eta_c=0.8, eta_t=0.85)


def test_silent_case_raises_with_message():
    with pytest.raises(InputError) as e:
        tc.Cycle("brayton_regen", T3=1250.0, **BASE, **NONIDEAL)
    assert MSG in str(e.value)
    assert "plausibility='warn'" in str(e.value)


def test_ideal_case_raises_the_same_by_default():
    with pytest.raises(InputError, match="Did you mean T4=1250"):
        tc.Cycle("brayton_regen", T3=1250.0, **BASE)


def test_ideal_case_no_solution_carries_the_hint_when_opted_out():
    with pytest.raises(NoSolutionError) as e:
        tc.Cycle("brayton_regen", T3=1250.0, plausibility="off", **BASE)
    assert MSG in str(e.value)


def test_hint_not_added_to_unrelated_no_solution():
    # a failing solve with a normal T3 must not mention the mix-up
    with pytest.raises(Exception) as e:
        tc.Cycle("brayton_regen", T1=300.0, P1=100e3, r_p=8.0, T4=250.0, eps=0.75)
    assert "Did you mean" not in str(e.value)


def test_warn_opt_out_warns_and_solves():
    with pytest.warns(InputPlausibilityWarning, match="Did you mean T4=1250"):
        c = tc.Cycle("brayton_regen", T3=1250.0, plausibility="warn", **BASE, **NONIDEAL)
    assert c.states[3].T == pytest.approx(1250.0)


@pytest.mark.parametrize("T3", [700.0, 800.0, 900.0, 999.0])
def test_legitimate_regenerator_outlet_no_error(T3, recwarn):
    c = tc.Cycle("brayton_regen", T3=T3, **BASE, **NONIDEAL)
    assert c.states[3].T == pytest.approx(T3)
    assert not [w for w in recwarn if issubclass(w.category, InputPlausibilityWarning)]


def test_threshold_is_1000_K():
    with pytest.raises(InputError):
        tc.Cycle("brayton_regen", T3=1000.0, **BASE, **NONIDEAL)


def test_t4_given_is_fine():
    c = tc.Cycle("brayton_regen", T4=1250.0, **BASE, **NONIDEAL)
    assert c.states[4].T == pytest.approx(1250.0)


def test_t3_and_t4_both_given_still_guarded_when_t3_high():
    with pytest.raises(InputError, match="Did you mean"):
        tc.Cycle("brayton_regen", T3=1250.0, T4=1400.0, T1=300.0, P1=100e3, r_p=8.0, **NONIDEAL)


def test_brayton_t3_unaffected(recwarn):
    c = tc.Cycle("brayton", T1=300.0, P1=100e3, r_p=8.0, T3=1250.0, **NONIDEAL)
    assert c.states[3].T == pytest.approx(1250.0)
    assert not [w for w in recwarn if issubclass(w.category, InputPlausibilityWarning)]


def test_requirements_also_guarded():
    with pytest.raises(InputError, match="Did you mean T4=1250"):
        tc.requirements("brayton_regen", T3=1250.0, T1=300.0, P1=100e3, r_p=8.0)


def test_describe_states_turbine_inlet():
    reg = tc.describe("brayton_regen")
    assert "Turbine inlet: T4." in reg and "T3 is the regenerator cold outlet" in reg
    assert "Turbine inlet: T3." in tc.describe("brayton")


def test_readme_row_names_turbine_inlet():
    import pathlib
    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    row = next(l for l in text.splitlines() if l.startswith("| `brayton_regen`"))
    assert "turbine inlet = T4" in row
    row = next(l for l in text.splitlines() if l.startswith("| `brayton` "))
    assert "turbine inlet = T3" in row
