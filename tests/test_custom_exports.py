"""5d (v0.1.1): what the README needs for a custom cycle is importable from the package
(T0091 open issue 2)."""

import pytest

import thermo_cycles as tc
from thermo_cycles.model.cycle_def import _unregister


def test_definition_classes_at_top_level():
    from thermo_cycles import CycleDef, ParamDef, StepDef   # noqa: F401
    for n in ("CycleDef", "StepDef", "ParamDef"):
        assert n in tc.__all__


def test_custom_namespace_has_every_helper():
    from thermo_cycles import custom
    for n in ("CycleDef", "StepDef", "ParamDef", "register_cycle", "Relation", "ratio", "linear", "equal",
              "product", "fixed"):
        assert hasattr(custom, n) and n in custom.__all__, n


def test_readme_custom_example_runs():
    from thermo_cycles.custom import CycleDef, ParamDef, StepDef, ratio, register_cycle
    try:
        register_cycle(CycleDef(
            name="readme_3step", title="three-step cycle", aliases=(), system="closed", states=(1, 2, 3),
            steps=(StepDef("isentropic", 1, 2, "compression"), StepDef("isobaric", 2, 3, "heat_in"),
                   StepDef("isochoric", 3, 1, "heat_out")),
            params=(ParamDef("r_c", "compression ratio v1/v2", ratio("r_c", "v1", "v2")),),
            v_max_state=1, v_min_state=2))
        c = tc.Cycle("readme_3step", T1=293, P1=1e5, r_c=5)
        assert c.results.eta_th > 0
    finally:
        _unregister("readme_3step")


def test_help_register_cycle_says_where_the_classes_are():
    doc = tc.register_cycle.__doc__
    assert "thermo_cycles.custom" in doc and "CycleDef" in doc


def test_readme_shows_the_import():
    import pathlib
    text = (pathlib.Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert "from thermo_cycles.custom import" in text
