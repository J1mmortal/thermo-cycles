"""Fix 4 (v0.1.1): smaller usability items of trial T0089 (findings 4, 5, 6, 7, 8)."""

import pytest

from thermo_cycles import Cycle, optimise
from thermo_cycles.errors import InputError, UnknownInputError

GT = dict(T1=288.15, P1=101300, r_p=8, T3=1250)


# unknown names list the valid ones -------------------------------------------------------------------------
def test_unknown_input_lists_known_inputs():
    with pytest.raises(UnknownInputError) as ei:
        Cycle("brayton", TIT=1250, **{k: v for k, v in GT.items() if k != "T3"})
    msg = str(ei.value)
    assert msg.startswith("brayton: unknown input 'TIT'.")
    assert "Known inputs:" in msg
    assert "state properties T, P, v, u, h, s at states 1, 2, 3, 4" in msg
    assert "parameters r_p, eta_c, eta_t" in msg
    assert "q_in" in msg and "w_net" in msg
    assert "describe('brayton')" in msg
    assert "TIT" not in ei.value.accepted


def test_unknown_input_close_match_is_suggested():
    with pytest.raises(UnknownInputError, match=r"Did you mean: eta_c"):
        Cycle("brayton", etac=0.8, **GT)


def test_unknown_state_number_says_which_states_exist():
    c = Cycle("brayton", **GT)
    with pytest.raises(UnknownInputError) as ei:
        c.value("T9")
    msg = str(ei.value)
    assert msg.startswith("brayton: unknown variable 'T9'.")
    assert "brayton has no state 9 (states: 1, 2, 3, 4)" in msg
    assert "Known variables:" in msg
    assert "Cycle results are also accepted:" in msg and "eta_carnot" in msg


# percent instead of fraction ---------------------------------------------------------------------------------
def test_percent_efficiency_says_fractions():
    with pytest.raises(InputError) as ei:
        Cycle("brayton", eta_c=82, **GT)
    msg = str(ei.value)
    assert "eta_c = 82 is outside the range" in msg
    assert "efficiencies and ratios are fractions (0.82, not 82)" in msg


def test_fraction_hint_only_for_plausible_percent():
    with pytest.raises(InputError) as ei:
        Cycle("brayton", eta_c=150, **GT)
    assert "fractions" not in str(ei.value)
    with pytest.raises(InputError) as ei:
        Cycle("brayton_regen", T1=288.15, P1=101300, r_p=8, T4=1250, eps=80)
    assert "(0.8, not 80)" in str(ei.value)


# optimise: short repr, units in the message ------------------------------------------------------------------
@pytest.fixture(scope="module")
def best_w():
    return optimise("brayton", vary={"r_p": (1.5, 40)}, maximise="w_net",
                    fixed=dict(T1=288.15, P1=101300, T3=1250, eta_c=0.85, eta_t=0.88))


def test_optimisation_repr_is_short(best_w):
    text = repr(best_w)
    assert len(text) < 400 and "\n" not in text
    assert "history" not in text or "len(history)" in text or ".history" in text
    assert text.startswith("OptimisationResult(")
    assert "218.67 kJ/kg" in text
    assert len(best_w.history) == best_w.evaluations     # still there


def test_optimisation_message_has_units(best_w):
    assert best_w.message.startswith("maximum of w_net = 218.67 kJ/kg at r_p = 7.84491;")
    eta = optimise("brayton", vary={"T3": (900, 1500)}, maximise="eta_th", fixed=dict(T1=288.15, P1=101300, r_p=8),
                   grid=9)
    # ideal Brayton: eta_th = 1 - r_p**(-0.4/1.4) = 0.447955 for any T3 (dimensionless, so no unit); T3 in K
    assert eta.message.startswith("maximum of eta_th = 0.447955 at T3 = ")
    assert eta.message.split(";")[0].endswith(" K")


# float noise: values as computed, summary rounded ------------------------------------------------------------
def test_results_not_rounded_but_summary_is():
    c = Cycle("brayton", **GT)
    assert c.results.P_max == pytest.approx(810400, rel=1e-12)       # value as computed (may carry float noise)
    s = c.summary()
    assert "P 101.30..810.40 kPa" in s
    assert "99999" not in s and "00000001" not in s


# overlaid plots: state labels do not collide -----------------------------------------------------------------
def test_overlay_state_labels_do_not_collide():
    pytest.importorskip("matplotlib").use("Agg")
    from thermo_cycles import plot_cycles
    simple = Cycle("brayton", eta_c=0.85, eta_t=0.88, **GT)
    regen = Cycle("brayton_regen", T1=288.15, P1=101300, r_p=8, T4=1250, eps=0.8, eta_c=0.85, eta_t=0.88)
    ax = plot_cycles([simple, regen], "Ts")
    texts = [t for t in ax.texts if t.get_text()]
    labels = [t.get_text() for t in texts]
    assert "3" in labels and "3'" in labels and "5s'" in labels and "4s" in labels
    pos = {}
    for t in texts:                                      # same data point and same offset -> collision
        key = (round(float(t.xy[0]), 6), round(float(t.xy[1]), 6), tuple(t.xyann))
        assert key not in pos, f"{t.get_text()} collides with {pos[key]}"
        pos[key] = t.get_text()
    offsets = {t.get_text(): tuple(t.xyann) for t in texts}
    assert offsets["1"] != offsets["1'"]
    # solid/dashed convention kept: irreversible steps dashed for both cycles
    assert {l.get_linestyle() for l in ax.lines if l.get_gid() == "step 1-2"} == {"--"}


def test_single_cycle_labels_unchanged():
    pytest.importorskip("matplotlib").use("Agg")
    ax = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800).plot("Ts")
    assert sorted(t.get_text() for t in ax.texts if t.get_text()) == ["1", "2", "3", "4"]
