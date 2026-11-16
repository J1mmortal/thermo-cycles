"""5f (v0.1.1): an optimum on a vary bound is reported in res.message and res.at_bound (T0091 open issue 6)."""

import pytest

from thermo_cycles import optimise


def test_upper_bound_ideal_brayton_eta():
    # ideal Brayton: eta_th = 1 - r_p**(-0.4/1.4) rises with r_p, so the maximum is the upper bound
    res = optimise("brayton", vary={"r_p": (2, 30)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5, T3=1300))
    assert res.x["r_p"] == pytest.approx(30)
    assert dict(res.at_bound) == {"r_p": "upper"}
    assert "on the upper bound of r_p (30)" in res.message
    assert "widen" in res.message
    assert "at_bound" in repr(res) or "bound" in repr(res)


def test_lower_bound_minimise():
    res = optimise("brayton", vary={"r_p": (2, 30)}, minimise="eta_th", fixed=dict(T1=300, P1=1e5, T3=1300),
                   grid=9)
    assert dict(res.at_bound) == {"r_p": "lower"}
    assert "on the lower bound of r_p (2)" in res.message


def test_diesel_without_pressure_limit_hits_r_c_bound():
    res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th", fixed=dict(T1=298, P1=101325, T3=2700))
    assert dict(res.at_bound) == {"r_c": "upper"}


def test_interior_optimum_has_no_bound():
    res = optimise("brayton", vary={"r_p": (1.5, 40)}, maximise="w_net",
                   fixed=dict(T1=288.15, P1=101300, T3=1250, eta_c=0.85, eta_t=0.88))
    assert dict(res.at_bound) == {}
    assert "bound" not in res.message.split(". ")[0]


def test_limit_bound_optimum_is_not_a_vary_bound():
    res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th", fixed=dict(T1=298, P1=101325, T3=2700),
                   limits={"P2_max": 70 * 101325})
    assert dict(res.at_bound) == {} and res.active == ("P2_max",)


def test_two_variables_one_on_a_bound():
    res = optimise("dual", vary={"r_c": (8, 18), "alpha": (1.0, 2.0)}, maximise="eta_th",
                   fixed=dict(T1=300, P1=1e5, q_in=1.5e6), limits={"P_max": 9e6})
    for k, side in res.at_bound.items():
        lo, hi = {"r_c": (8, 18), "alpha": (1.0, 2.0)}[k]
        assert res.x[k] == pytest.approx(lo if side == "lower" else hi, rel=1e-6)
        assert f"on the {side} bound of {k}" in res.message
