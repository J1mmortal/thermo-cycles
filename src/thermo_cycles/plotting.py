"""T-s and P-v plots with the true process curves (spec section 13).

matplotlib is optional and imported only inside the functions of this module; without it the functions raise
``MissingDependencyError``. Every step line carries ``gid="step a-b"``, so the plotted data can be read back from
``ax.lines``. Other artists use the gids ``"state i"``, ``"hidden is"``, ``"isentrope a-bs"`` and
``"saturation dome"``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .errors import InputError, MissingDependencyError
from .model.steps import KINDS

__all__ = ["plot_cycle", "plot_cycles"]

KINDS_PLOT = ("Ts", "Pv", "both")
P_UNITS = {"Pa": 1.0, "kPa": 1e3, "bar": 1e5, "MPa": 1e6}
S_UNITS = {"J": 1.0, "kJ": 1e3}
_MSG_NO_MPL = "plotting needs matplotlib: pip install 'thermo-cycles[plot]'"
_NOLEGEND = "_nolegend_"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:        # also raised when sys.modules["matplotlib"] is None
        raise MissingDependencyError(_MSG_NO_MPL) from exc
    return plt


def _colour(i: int) -> str:
    import matplotlib as mpl
    cols = mpl.rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]
    return cols[i % len(cols)]


def _check_options(kind, n, P_unit, s_unit, line_kw):
    if kind not in KINDS_PLOT:
        raise InputError(f"plot kind must be one of {KINDS_PLOT}, got {kind!r}.")
    if P_unit not in P_UNITS:
        raise InputError(f"P_unit must be one of {tuple(P_UNITS)}, got {P_unit!r}.")
    if s_unit not in S_UNITS:
        raise InputError(f"s_unit must be one of {tuple(S_UNITS)}, got {s_unit!r}.")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise InputError(f"n (points per step) must be an integer >= 2, got {n!r}.")
    bad = sorted(k for k in line_kw if k in ("linestyle", "ls", "gid"))
    if bad:
        raise InputError(f"{bad} cannot be set: the line style marks reversible (solid) and irreversible "
                         f"(dashed) steps and gid identifies the step. Use color, linewidth, alpha, ... instead.")


def _check_cycle(cycle):
    if not all(hasattr(cycle, a) for a in ("definition", "states", "hidden_states", "fluid")):
        raise InputError(f"expected a thermo_cycles.Cycle, got {type(cycle).__name__}.")


def _xy(kind, P_unit, s_unit):
    """Return (get(state) -> (x, y), xlabel, ylabel) for one panel."""
    if kind == "Ts":
        f = S_UNITS[s_unit]
        return (lambda st: (st.s / f, st.T)), f"s [{'kJ' if s_unit == 'kJ' else 'J'}/(kg K)]", "T [K]"
    f = P_UNITS[P_unit]
    return (lambda st: (st.v, st.P / f)), "v [m³/kg]", f"P [{P_unit}]"


def _arrow(ax, xs, ys, colour):
    """Direction arrow at mid-step, from the 45 % to the 55 % point along the polyline index."""
    t = np.linspace(0.0, 1.0, len(xs))
    x0, x1 = np.interp([0.45, 0.55], t, xs)
    y0, y1 = np.interp([0.45, 0.55], t, ys)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0, "mutation_scale": 12})


def _dome(ax, fluid, kind, P_unit, s_unit):
    curve = getattr(fluid, "saturation_curve", None)
    if curve is None:
        return
    sat = curve()
    if kind == "Ts":
        f = S_UNITS[s_unit]
        x = np.concatenate([np.asarray(sat["s_f"]), np.asarray(sat["s_g"])[::-1]]) / f
        y = np.concatenate([np.asarray(sat["T"]), np.asarray(sat["T"])[::-1]])
    else:
        f = P_UNITS[P_unit]
        x = np.concatenate([np.asarray(sat["v_f"]), np.asarray(sat["v_g"])[::-1]])
        y = np.concatenate([np.asarray(sat["P"]), np.asarray(sat["P"])[::-1]]) / f
    ax.plot(x, y, color="0.5", lw=1.0, gid="saturation dome", label=_NOLEGEND)


def _draw_panel(cycle, ax, kind, *, n, show_states, show_hidden, logscale_P, P_unit, s_unit, label, arrows,
                colour, line_kw):
    get, xlabel, ylabel = _xy(kind, P_unit, s_unit)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if kind == "Pv" and logscale_P:
        ax.set_yscale("log")
    fluid = cycle.fluid
    if not getattr(ax, "_thermo_cycles_dome", False):
        _dome(ax, fluid, kind, P_unit, s_unit)
        ax._thermo_cycles_dome = True
    states, hidden = cycle.states, cycle.hidden_states
    labelled = False
    for st in cycle.definition.steps:
        k = KINDS[st.kind]
        sa, sb = states[st.a], states[st.b]
        pts = k.path(fluid, sa, sb, n)
        if pts is None:
            pts = [sa, sb]      # irreversible step: no path, straight dashed line between the end states
        xy = np.array([get(p) for p in pts], dtype=float)
        # only the first reversible step of a cycle carries the legend label; fall back to the first step
        first_ok = (not labelled) and (k.reversible or all(not KINDS[s.kind].reversible
                                                           for s in cycle.definition.steps))
        use = label if (label is not None and first_ok) else _NOLEGEND
        labelled = labelled or first_ok
        ax.plot(xy[:, 0], xy[:, 1], linestyle="-" if k.reversible else "--", color=colour, gid=f"step {st.a}-{st.b}",
                label=use, **line_kw)
        if arrows:
            _arrow(ax, xy[:, 0], xy[:, 1], colour)
        if show_hidden:
            for hid in k.hidden_states(str(st.b)):
                hs = hidden[hid]
                iso = KINDS["isentropic"].path(fluid, sa, hs, n)
                ixy = np.array([get(p) for p in iso], dtype=float)
                ax.plot(ixy[:, 0], ixy[:, 1], linestyle=":", color=colour, lw=1.0, gid=f"isentrope {st.a}-{hid}",
                        label=_NOLEGEND)
                hx, hy = get(hs)
                ax.plot([hx], [hy], linestyle="none", marker="o", markerfacecolor="none", color=colour,
                        gid=f"hidden {hid}", label=_NOLEGEND)
                ax.annotate(hid, (hx, hy), textcoords="offset points", xytext=(5, 5), fontsize=8, color=colour)
    if show_states:
        for sid, s in states.items():
            x, y = get(s)
            ax.plot([x], [y], linestyle="none", marker="o", color=colour, gid=f"state {sid}", label=_NOLEGEND)
            ax.annotate(str(sid), (x, y), textcoords="offset points", xytext=(5, 5), color=colour)


def _setup_axes(plt, kind, ax):
    """Return (fig_or_None, {panel kind: Axes}, result) for plot_cycle(s)."""
    if kind == "both":
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        else:
            try:
                axes = tuple(ax)
            except TypeError:
                raise InputError("for kind='both' pass ax=None or a pair of Axes (ax_Ts, ax_Pv).") from None
            if len(axes) != 2:
                raise InputError("for kind='both' pass ax=None or a pair of Axes (ax_Ts, ax_Pv).")
            fig = axes[0].figure
        return {"Ts": axes[0], "Pv": axes[1]}, (fig, (axes[0], axes[1]))
    if ax is None:
        _, ax = plt.subplots()
    return {kind: ax}, ax


def plot_cycle(cycle, kind="Ts", *, ax=None, n=100, show_states=True, show_hidden=True, logscale_P=False,
               P_unit="Pa", label=None, arrows=True, s_unit="J", **line_kw) -> Any:
    """Plot one solved cycle in T-s, P-v or both (``kind="both"`` returns ``(fig, (ax_Ts, ax_Pv))``).

    Reversible steps are solid lines along the true process path; irreversible (adiabatic with efficiency)
    steps are dashed straight lines, with the hidden isentropic end state as an open marker and a dotted
    isentrope to it. ``label`` adds a legend entry. ``P_unit`` is Pa, kPa, bar or MPa and ``s_unit`` is J or kJ.
    Extra keywords (color, linewidth, alpha, ...) go to the step lines."""
    return _plot_many([cycle], kind, ax=ax, labels=[label], n=n, show_states=show_states,
                      show_hidden=show_hidden, logscale_P=logscale_P, P_unit=P_unit, arrows=arrows, s_unit=s_unit,
                      line_kw=line_kw)


def plot_cycles(cycles: Sequence, kind="Ts", *, ax=None, labels=None, **kw) -> Any:
    """Overlay several cycles on one set of axes, each in its own colour, with a legend when ``labels`` is
    given. Keywords as for :func:`plot_cycle`."""
    cycles = list(cycles)
    if not cycles:
        raise InputError("plot_cycles needs at least one cycle.")
    if labels is None:
        labels = [None] * len(cycles)
    elif len(labels) != len(cycles):
        raise InputError(f"labels has {len(labels)} entries for {len(cycles)} cycles; give one label per cycle.")
    known = {"n", "show_states", "show_hidden", "logscale_P", "P_unit", "arrows", "s_unit"}
    opts = {k: kw.pop(k) for k in list(kw) if k in known}
    return _plot_many(cycles, kind, ax=ax, labels=list(labels), line_kw=kw, **opts)


def _plot_many(cycles, kind, *, ax, labels, line_kw, n=100, show_states=True, show_hidden=True, logscale_P=False,
               P_unit="Pa", arrows=True, s_unit="J"):
    _check_options(kind, n, P_unit, s_unit, line_kw)
    for c in cycles:
        _check_cycle(c)
    plt = _pyplot()
    panels, result = _setup_axes(plt, kind, ax)
    for i, (c, lab) in enumerate(zip(cycles, labels)):
        kw = dict(line_kw)
        colour = kw.pop("color", None) or kw.pop("c", None) or _colour(i)
        for pk, pax in panels.items():
            _draw_panel(c, pax, pk, n=n, show_states=show_states, show_hidden=show_hidden, logscale_P=logscale_P,
                        P_unit=P_unit, s_unit=s_unit, label=lab, arrows=arrows, colour=colour, line_kw=kw)
    if any(lab is not None for lab in labels):
        for pax in panels.values():
            pax.legend()
    return result
