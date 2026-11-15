# Extra tests for Task 14 (not in the plan): describe() content for every cycle, aliases, errors, final __all__.
import pytest
import thermo_cycles as tc
from thermo_cycles import errors


def test_describe_numbered_steps_and_roles_for_all_cycles():
    for name in tc.list_cycles():
        defn = tc.get_cycle(name)
        text = tc.describe(name)
        assert defn.title in text
        for st in defn.steps:
            assert f"{st.a}->{st.b} {st.kind} ({st.role})" in text
        for p in defn.params:
            assert p.name in text and p.description in text
        for ref in defn.references:
            assert ref in text


def test_describe_dof_matches_requirements():
    # "needs N inputs" is the number of inputs an empty set is missing (spec 8.2 DOF column)
    expected = {"otto": 4, "diesel": 4, "dual": 5, "atkinson": 4, "brayton": 4, "brayton_regen": 5}
    for name, n in expected.items():
        assert tc.requirements(name).missing == n
        assert f"needs {n} inputs" in tc.describe(name)


def test_describe_defaults_and_typical_sets():
    text = tc.describe("brayton")
    assert "eta_c" in text and "default 1" in text
    assert "T1, P1, r_p, T3" in text
    text = tc.describe("brayton_regen")
    assert "T1, P1, r_p, T4, eps" in text and "T1, P1, r_p, T4, dT_regen" in text


def test_describe_accepts_case_and_unknown_name():
    assert tc.describe("OTTO") == tc.describe("otto")
    with pytest.raises(errors.UnknownCycleError):
        tc.describe("ottto")


def test_describe_is_a_str_and_reaches_cycle_module():
    from thermo_cycles.cycle import describe
    assert describe is tc.describe and isinstance(tc.describe("dual"), str)


def test_final_all_is_importable_and_complete():
    required = {"Cycle", "IdealGas", "AIR", "requirements", "describe", "list_cycles", "register_cycle",
                "optimise", "plot_cycles", "errors", "units", "__version__"}
    assert required <= set(tc.__all__)
    assert len(tc.__all__) == len(set(tc.__all__))
    for n in tc.__all__:
        assert hasattr(tc, n), n
    ns = {}
    exec("from thermo_cycles import *", ns)
    assert required <= set(ns)


def test_margin_types_are_public():
    assert {"Margin", "MarginReport", "SecondLaw"} <= set(tc.__all__)
    c = tc.Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800, limits={"T_max": 2000})
    assert isinstance(c.margins, tc.MarginReport) and isinstance(c.margins["T_max"], tc.Margin)
    assert isinstance(c.margins.second_law, tc.SecondLaw)


def test_catalogue_exports_all_piston_cycles():
    from thermo_cycles import catalogue
    for n in ("OTTO", "DIESEL", "DUAL", "ATKINSON", "BRAYTON", "BRAYTON_REGEN"):
        assert n in catalogue.__all__ and getattr(catalogue, n).name == n.lower()
