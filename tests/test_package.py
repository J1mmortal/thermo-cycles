import importlib, sys
import pytest

def test_import_and_version():
    tc = importlib.import_module("thermo_cycles")
    assert tc.__version__ == "0.1.0"

def test_error_hierarchy():
    from thermo_cycles import errors as E
    assert issubclass(E.UnderdeterminedError, E.InputError)
    assert issubclass(E.InputError, ValueError) and issubclass(E.InputError, E.ThermoCyclesError)
    assert issubclass(E.FluidError, ValueError)
    assert issubclass(E.MissingDependencyError, ImportError)
    for w in (E.LimitViolationWarning, E.CycleValidityWarning, E.InputPlausibilityWarning):
        assert issubclass(w, UserWarning)
    e = E.UnderdeterminedError("msg", missing=1, given=("T1",), add_one_of=("T3",), example_sets=(), determined=())
    assert e.missing == 1 and e.add_one_of == ("T3",) and str(e) == "msg"

def test_units():
    from thermo_cycles.units import kPa, bar, MPa, atm, kJ, L, degC
    assert (kPa, bar, MPa, atm, kJ, L) == (1e3, 1e5, 1e6, 101325.0, 1e3, 1e-3)
    assert degC(25) == pytest.approx(298.15)

def test_no_old_modules_and_no_matplotlib_import():
    import thermo_cycles  # noqa: F401
    assert "matplotlib" not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("thermo_cycles.core")
