import pytest
from thermo_cycles import optimise
from thermo_cycles.errors import NoFeasiblePointError, UnderdeterminedError

def test_old_efficiency_mode_replacement_boundary_optimum():
    # old EFFICIENCY mode: T1 298, T3_MAX 2700, r_c 5..60, P2/P1 <= 70 (P1 was hard-coded 101325)
    res = optimise("diesel", vary={"r_c": (5, 60)}, maximise="eta_th",
                   fixed=dict(T1=298, P1=101325, T3=2700), limits={"P2_max": 70 * 101325})
    assert res.x["r_c"] == pytest.approx(70 ** (1 / 1.4), rel=1e-5)       # 20.7935
    assert res.value == pytest.approx(0.623769, rel=1e-5)
    assert res.active == ("P2_max",) and res.feasible
    assert res.cycle.states[2].P == pytest.approx(70 * 101325, rel=1e-5)   # states are the optimum's

def test_interior_optimum_max_work():
    res = optimise("otto", vary={"r_c": (1.5, 30)}, maximise="w_net", fixed=dict(T1=300, P1=1e5, T3=1500))
    assert res.x["r_c"] == pytest.approx(5 ** 1.25, rel=1e-4)              # 7.4767
    assert res.value == pytest.approx(328.87e3, rel=1e-4) and res.active == ()

def test_two_variables():
    # Brayton max w_net: T3 pinned by T_max, then r_p* = (T3/T1)^(gamma/(2(gamma-1))) = 18.7175,
    # w* = cp (T3 - 2 sqrt(T1 T3) + T1) = 516.674 kJ/kg
    res = optimise("brayton", vary={"r_p": (2, 40), "T3": (800, 2000)}, maximise="w_net",
                   fixed=dict(T1=300, P1=1e5), limits={"T_max": 1600})
    assert res.feasible and res.active == ("T_max",)
    assert res.x["T3"] == pytest.approx(1600, rel=1e-4) and res.x["r_p"] == pytest.approx(18.7175, rel=1e-3)
    assert res.value == pytest.approx(516.674e3, rel=1e-4)

def test_callable_objective_and_errors():
    res = optimise("otto", vary={"r_c": (2, 20)}, minimise=lambda c: -c.results.eta_th,
                   fixed=dict(T1=300, P1=1e5, q_in=1e6), limits={"T_max": 2500})
    assert res.cycle.results.T_max <= 2500 * (1 + 1e-6)
    with pytest.raises(NoFeasiblePointError):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5, q_in=1e6),
                 limits={"P_max": 1e5})
    with pytest.raises(UnderdeterminedError):
        optimise("otto", vary={"r_c": (5, 10)}, maximise="eta_th", fixed=dict(T1=300, P1=1e5))
