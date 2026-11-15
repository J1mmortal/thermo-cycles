# tests/test_plotting.py  (plan Task 13, spec section 13)
import subprocess
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from thermo_cycles import Cycle, IdealGas, plot_cycles   # noqa: E402
from thermo_cycles.errors import InputError, MissingDependencyError      # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def lines(ax):
    return {l.get_gid(): l for l in ax.lines if l.get_gid() and l.get_gid().startswith("step")}


def _gid(ax, prefix):
    return {l.get_gid(): l for l in ax.lines if l.get_gid() and l.get_gid().startswith(prefix)}


def _brayton():
    """The plan's literal Task 13 cycle: the catalogue Brayton with eta_c = 0.85, eta_t = 0.9."""
    return Cycle("brayton", T1=300, P1=1e5, r_p=10, T3=1400, eta_c=0.85, eta_t=0.9)


def test_Ts_true_curves():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    L = lines(c.plot("Ts"))
    s12 = L["step 1-2"].get_xdata()
    assert np.ptp(s12) < 1e-6                                   # isentrope vertical in T-s
    T23 = L["step 2-3"].get_ydata(); s23 = L["step 2-3"].get_xdata()
    assert len(T23) > 10
    mid = len(T23) // 2                                         # isochore is curved, not a chord
    chord = T23[0] + (T23[-1] - T23[0]) * (s23[mid] - s23[0]) / (s23[-1] - s23[0])
    assert abs(T23[mid] - chord) > 1.0


def test_Ts_isochore_matches_closed_form():
    # ideal gas, v = const: s - s2 = cv ln(T/T2) exactly, so T(s) = T2 exp((s - s2)/cv)
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    cv = c.fluid.cv
    l = lines(c.plot("Ts"))["step 2-3"]
    s, T = np.asarray(l.get_xdata()), np.asarray(l.get_ydata())
    assert T == pytest.approx(T[0] * np.exp((s - s[0]) / cv), rel=1e-9)
    assert (T[0], T[-1]) == pytest.approx((c.states[2].T, c.states[3].T), rel=1e-12)


def test_Pv_isentrope_follows_P_vgamma():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    l = lines(c.plot("Pv"))["step 1-2"]
    v, P = np.asarray(l.get_xdata()), np.asarray(l.get_ydata())
    g = c.fluid.gamma
    assert P * v ** g == pytest.approx(P[0] * v[0] ** g, rel=1e-9)
    mid = len(P) // 2                                           # curved, not the chord between the end states
    chord = P[0] + (P[-1] - P[0]) * (v[mid] - v[0]) / (v[-1] - v[0])
    assert len(P) > 10 and abs(P[mid] - chord) > 1e3


def test_Pv_and_units_and_irreversible_dashed():
    c = _brayton()
    ax = c.plot("Pv", P_unit="kPa")
    L = lines(ax)
    assert L["step 1-2"].get_linestyle() == "--" and L["step 2-3"].get_linestyle() == "-"
    assert max(L["step 2-3"].get_ydata()) == pytest.approx(1000.0, rel=1e-9)     # 1e6 Pa in kPa
    assert "kPa" in ax.get_ylabel()


@pytest.mark.parametrize("unit,factor", [("Pa", 1.0), ("kPa", 1e3), ("bar", 1e5), ("MPa", 1e6)])
def test_pressure_units(unit, factor):
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    ax = c.plot("Pv", P_unit=unit)
    assert lines(ax)["step 3-4"].get_ydata()[0] == pytest.approx(c.states[3].P / factor, rel=1e-12)
    assert f"[{unit}]" in ax.get_ylabel()


def test_bad_options_raise_input_error():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    for kw in ({"P_unit": "psi"}, {"s_unit": "MJ"}, {"n": 1}, {"linestyle": ":"}):
        with pytest.raises(InputError):
            c.plot("Ts", **kw)
    with pytest.raises(InputError, match="kind"):
        c.plot("hs")
    with pytest.raises(InputError):
        plot_cycles([], "Ts")
    with pytest.raises(InputError, match="labels"):
        plot_cycles([c, c], "Ts", labels=["a"])


def test_s_unit_kJ_and_axis_labels():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    a1 = c.plot("Ts"); a2 = c.plot("Ts", s_unit="kJ")
    assert lines(a2)["step 2-3"].get_xdata() == pytest.approx(lines(a1)["step 2-3"].get_xdata() / 1e3, rel=1e-12)
    assert "J/(kg K)" in a1.get_xlabel() and "kJ/(kg K)" in a2.get_xlabel() and "K" in a1.get_ylabel()
    ax = c.plot("Pv", logscale_P=True)
    assert ax.get_yscale() == "log" and "m" in ax.get_xlabel()


def test_hidden_state_open_marker_and_dotted_isentrope():
    c = _brayton()
    ax = c.plot("Ts")
    mark = _gid(ax, "hidden")
    assert set(mark) == {"hidden 2s", "hidden 4s"}
    m = mark["hidden 2s"]
    assert m.get_markerfacecolor() == "none" and m.get_linestyle() in ("None", "none", "")
    assert (m.get_xdata()[0], m.get_ydata()[0]) == pytest.approx((c.hidden_states["2s"].s, c.hidden_states["2s"].T))
    iso = _gid(ax, "isentrope")["isentrope 1-2s"]
    assert iso.get_linestyle() == ":" and np.ptp(iso.get_xdata()) < 1e-6
    assert iso.get_ydata()[-1] == pytest.approx(c.hidden_states["2s"].T, rel=1e-9)
    ax2 = c.plot("Ts", show_hidden=False)                       # show_hidden=False removes them
    assert not _gid(ax2, "hidden") and not _gid(ax2, "isentrope")


def test_no_hidden_decoration_for_reversible_cycle():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    ax = c.plot("Ts")
    assert not _gid(ax, "hidden") and not _gid(ax, "isentrope")
    assert all(l.get_linestyle() == "-" for l in lines(ax).values())


def test_state_markers_labels_and_arrows():
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    ax = c.plot("Ts")
    assert set(_gid(ax, "state")) == {f"state {i}" for i in (1, 2, 3, 4)}
    texts = [t.get_text() for t in ax.texts]
    for i in "1234":
        assert i in texts
    assert sum(1 for t in ax.texts if t.arrow_patch is not None) == 4
    ax2 = c.plot("Ts", show_states=False, arrows=False)
    assert not _gid(ax2, "state") and not ax2.texts


def test_overlay_and_both():
    a = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800); b = a.with_inputs(r_c=10)
    ax = plot_cycles([a, b], "Ts", labels=["r=8", "r=10"])
    assert len([t.get_text() for t in ax.get_legend().get_texts()]) == 2
    fig, (ax1, ax2) = a.plot("both")
    assert ax1 is not ax2


def test_overlay_keeps_both_cycles_and_distinct_colours():
    a = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800); b = a.with_inputs(r_c=10)
    ax = plot_cycles([a, b], "Pv", labels=["a", "b"])
    s12 = [l for l in ax.lines if l.get_gid() == "step 1-2"]
    assert len(s12) == 2 and s12[0].get_color() != s12[1].get_color()
    assert s12[0].get_xdata()[-1] != pytest.approx(s12[1].get_xdata()[-1])
    fig, (ax1, ax2) = plot_cycles([a, b], "both", labels=["a", "b"])
    assert len(ax1.get_legend().get_texts()) == 2 and ax2.get_xlabel() != ax1.get_xlabel()


def test_plot_on_given_axes_and_label():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    out = c.plot("Ts", ax=ax, label="otto r=8")
    assert out is ax and [t.get_text() for t in ax.get_legend().get_texts()] == ["otto r=8"]
    plt.close(fig)


def test_saturation_dome_drawn_when_fluid_has_curve():
    class DomeGas(IdealGas):
        def saturation_curve(self):
            T = np.linspace(300.0, 500.0, 20)
            return {"T": T, "P": 1e5 * (T / 300.0) ** 4, "s_f": 100.0 + 2 * T, "s_g": 3000.0 - 2 * T,
                    "v_f": np.full(20, 1e-3), "v_g": 1.0 / (T - 250.0)}
    c = Cycle("otto", fluid=DomeGas(R=287.0, gamma=1.4), T1=300, P1=1e5, r_c=8, T3=1800)
    dome = _gid(c.plot("Ts"), "saturation")["saturation dome"]
    assert len(dome.get_xdata()) == 40 and max(dome.get_ydata()) == pytest.approx(500.0)
    assert "saturation dome" in _gid(c.plot("Pv"), "saturation")
    assert not _gid(Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800).plot("Ts"), "saturation")


def test_missing_matplotlib_raises_with_exact_message(monkeypatch):
    c = Cycle("otto", T1=300, P1=1e5, r_c=8, T3=1800)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(MissingDependencyError) as e:
        c.plot("Ts")
    assert str(e.value) == "plotting needs matplotlib: pip install 'thermo-cycles[plot]'"
    with pytest.raises(ImportError):                       # MissingDependencyError is also an ImportError
        plot_cycles([c], "Ts")


def test_import_does_not_pull_in_matplotlib():
    code = ("import sys, thermo_cycles; from thermo_cycles import plot_cycles; "
            "print('matplotlib' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
