# tests/test_reference_brayton.py  (spec §15.1 B1, B2; incl. Review Focus 5)
import pytest
from thermo_cycles import Cycle, IdealGas
from thermo_cycles.errors import InvalidCycleError

def test_B1_thermostate_cold_air():
    c = Cycle("brayton", fluid=IdealGas(R=287, gamma=1.401662), T1=300, P1=1e5, r_p=10, T3=1400)
    assert c.states[2].T == pytest.approx(580.34, rel=1e-4) and c.states[4].T == pytest.approx(723.71, rel=1e-4)
    assert c.results.bwr == pytest.approx(0.4145, rel=5e-4) and c.results.eta_th == pytest.approx(0.4831, rel=5e-4)
    assert c.results.mep is None

def test_B2_cengel7e_9_110_regen_with_efficiencies():
    c = Cycle("brayton_regen", T1=293, P1=1e5, r_p=8, T4=1073, eta_c=0.87, eta_t=0.93, dT_regen=10)
    h = c.hidden_states
    assert h["2s"].T == pytest.approx(530.8, rel=3e-3) and h["5s"].T == pytest.approx(592.3, rel=3e-3)
    exp = {2: 566.3, 5: 625.9, 3: 615.9, 6: 576.3}
    for i, T in exp.items():
        assert c.states[i].T == pytest.approx(T, rel=3e-3), i
    assert c.results.w_net == pytest.approx(174.7e3, rel=3e-3)
    assert c.results.q_in == pytest.approx(394.4 / 0.8586 * 1e3, rel=3e-3)
    assert c.params["eps"] == pytest.approx(0.83252, rel=1e-4)
    c2 = Cycle("brayton_regen", T1=293, P1=1e5, r_p=8, T4=1073, eta_c=0.87, eta_t=0.93, eps=c.params["eps"])
    assert c2.states[3].T == pytest.approx(c.states[3].T, rel=1e-9)

def test_regenerator_cannot_heat_hot_stream():          # Review Focus 5: r_p 40 -> T5 < T2
    with pytest.raises(InvalidCycleError, match="regenerator"):
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=40, T4=1200, eps=0.8)


# ---- additions beyond the plan (brief T0082: spec 9.7 regenerator rule on both sides of the threshold,
# solver work budget for the new cycles) ----------------------------------------------------------------

from thermo_cycles import AIR, get_cycle                                      # noqa: E402
from thermo_cycles.solver.numeric import MAX_WORK, solve_numeric             # noqa: E402
from thermo_cycles.solver.system import build_system                          # noqa: E402

# Ideal cycle, AIR (gamma 1.4), T1 300 K, T4 1200 K: T5 = T2 exactly when (r_p^0.2857)^2 = T4/T1 = 4,
# i.e. r_p = 4^(1.4/0.8) = 11.3137. Below it T5 > T2 (regeneration possible), above it T5 < T2.
RP_STAR = 4 ** (1.4 / 0.8)


@pytest.mark.parametrize("eps", [0.0, 0.5, 1.0])
def test_regenerator_valid_just_below_threshold(eps):
    c = Cycle("brayton_regen", T1=300, P1=1e5, r_p=RP_STAR * 0.99, T4=1200, eps=eps)
    s, tol = c.states, 1e-9 * 1200
    assert s[5].T > s[2].T and s[5].T >= s[3].T - tol and s[6].T >= s[2].T - tol
    assert c.results.w_net == pytest.approx(c.results.q_in - c.results.q_out, rel=1e-12)


@pytest.mark.parametrize("eps", [0.0, 0.5, 1.0])
def test_regenerator_invalid_just_above_threshold(eps):
    with pytest.raises(InvalidCycleError, match="regenerator") as ei:
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=RP_STAR * 1.01, T4=1200, eps=eps)
    assert any("regenerator" in p and "T5" in p for p in ei.value.problems)


def test_regenerator_message_names_steps_and_fix():
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=40, T4=1200, eps=0.8)
    msg = str(ei.value)
    assert "2→3" in msg and "5→6" in msg and "r_p" in msg and "brayton" in msg


def test_regenerator_check_warn_returns_states():
    from thermo_cycles.errors import CycleValidityWarning
    with pytest.warns(CycleValidityWarning, match="regenerator"):
        c = Cycle("brayton_regen", T1=300, P1=1e5, r_p=40, T4=1200, eps=0.8, check="warn")
    assert c.states[5].T < c.states[2].T


def test_eps_input_outside_bounds_is_an_input_error():   # Task 7 decision: out-of-range inputs -> InputError
    from thermo_cycles.errors import InputError
    with pytest.raises(InputError, match="eps"):
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=5, T4=1400, eps=1.2)


def test_negative_dT_regen_overheats_cold_side():
    # T5 > T2 here, but dT_regen < 0 asks for T3 > T5 (eps > 1): the regenerator rule and the eps bound fire.
    with pytest.raises(InvalidCycleError) as ei:
        Cycle("brayton_regen", T1=300, P1=1e5, r_p=5, T4=1400, dT_regen=-20)
    probs = ei.value.problems
    assert any("regenerator" in p and "T5" in p and "T3" in p and "dT_regen" in p for p in probs)
    assert any("eps" in p and "bounds" in p for p in probs)


@pytest.mark.parametrize("name,inputs", [
    ("brayton", {"T1": 300.0, "P1": 1e5, "r_p": 10.0, "T3": 1400.0}),
    ("brayton", {"T1": 300.0, "P1": 1e5, "r_p": 10.0, "q_in": 6e5, "eta_c": 0.85, "eta_t": 0.9}),
    ("brayton_regen", {"T1": 293.0, "P1": 1e5, "r_p": 8.0, "T4": 1073.0, "eta_c": 0.87, "eta_t": 0.93,
                       "dT_regen": 10.0}),
    ("brayton_regen", {"T1": 293.0, "P1": 1e5, "r_p": 8.0, "T4": 1073.0, "eta_c": 0.87, "eta_t": 0.93,
                       "eps": 0.83252}),
])
def test_regular_solve_reports_work_well_below_budget(name, inputs):
    _, rep = solve_numeric(build_system(get_cycle(name), AIR), inputs)
    assert 0 < rep.work < MAX_WORK / 10


def test_internal_steps_must_cancel():
    # The energy balance h3 - h2 = h5 - h6 always holds after a solve, so perturb a solved step result to
    # check that validity_problems reports internal heat that does not cancel (w_net = q_in - q_out, spec 10).
    import dataclasses
    from thermo_cycles.results import validity_problems
    c = Cycle("brayton_regen", T1=293, P1=1e5, r_p=8, T4=1073, eta_c=0.87, eta_t=0.93, dT_regen=10)
    defn, vals = get_cycle("brayton_regen"), c._cs.values
    assert validity_problems(defn, vals, c.steps) == []
    steps = tuple(dataclasses.replace(s, q=s.q + 1e3) if (s.a, s.b) == (5, 6) else s for s in c.steps)
    assert any("internal steps 2→3, 5→6" in p for p in validity_problems(defn, vals, steps))
