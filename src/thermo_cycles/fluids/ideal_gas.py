"""Calorically perfect ideal gas (spec section 6.2). Uses ``math`` only."""

from __future__ import annotations

import math

from ..errors import FluidError
from ..state import PROPS
from .base import FluidBase

_RTOL = 1e-6


def _clean_constant(name, value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise FluidError(f"IdealGas: {name} must be a number, got {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise FluidError(f"IdealGas: {name} must be finite and positive, got {value!r}")
    return v


class IdealGas(FluidBase):
    """P v = R T, u = cv T, h = cp T, s = cp ln(T/T_ref) - R ln(P/P_ref).

    Give exactly two of R, cp, cv, gamma (none: AIR's R=287, gamma=1.4). Three or four
    consistent values (relative 1e-6) are accepted.
    """

    degenerate_pairs = (frozenset(("T", "u")), frozenset(("T", "h")), frozenset(("u", "h")))

    def __init__(self, *, R=None, cp=None, cv=None, gamma=None, name="ideal gas",
                 T_ref=273.15, P_ref=1e5):
        given = {k: v for k, v in (("R", R), ("cp", cp), ("cv", cv), ("gamma", gamma)) if v is not None}
        if not given:
            given = {"R": 287.0, "gamma": 1.4}
        given = {k: _clean_constant(k, v) for k, v in given.items()}
        if len(given) < 2:
            raise FluidError(f"IdealGas: give two of R, cp, cv, gamma; only {sorted(given)} given.")
        g = given
        nan = float("nan")
        # derive the full set from whichever two are available
        if "R" in g and "cp" in g:
            R_, cp_ = g["R"], g["cp"]
            cv_ = cp_ - R_
        elif "R" in g and "cv" in g:
            R_, cv_ = g["R"], g["cv"]
            cp_ = cv_ + R_
        elif "R" in g and "gamma" in g:
            R_ = g["R"]
            cv_ = R_ / (g["gamma"] - 1.0) if g["gamma"] != 1.0 else nan
            cp_ = g["gamma"] * cv_
        elif "cp" in g and "cv" in g:
            cp_, cv_ = g["cp"], g["cv"]
            R_ = cp_ - cv_
        elif "cp" in g and "gamma" in g:
            cp_ = g["cp"]
            cv_ = cp_ / g["gamma"]
            R_ = cp_ - cv_
        else:  # cv, gamma
            cv_ = g["cv"]
            cp_ = g["gamma"] * cv_
            R_ = cp_ - cv_
        if not (math.isfinite(R_) and math.isfinite(cp_) and math.isfinite(cv_)) or R_ <= 0 or cv_ <= 0:
            raise FluidError(
                f"IdealGas: constants {given} give R={R_:g}, cp={cp_:g}, cv={cv_:g}; "
                f"need R > 0 and cv > 0 (so gamma > 1)."
            )
        derived = {"R": R_, "cp": cp_, "cv": cv_, "gamma": cp_ / cv_}
        for k, v in given.items():
            if not math.isclose(v, derived[k], rel_tol=_RTOL):
                raise FluidError(
                    f"IdealGas: inconsistent constants: given {k}={v:g} but the others imply {k}={derived[k]:g} "
                    f"(cp = cv + R and gamma = cp/cv must hold to {_RTOL:g} relative)."
                )
        self.R, self.cp, self.cv, self.gamma = R_, cp_, cv_, derived["gamma"]
        self.T_ref = _clean_constant("T_ref", T_ref)
        self.P_ref = _clean_constant("P_ref", P_ref)
        self.name = name
        self.props = PROPS

    def __repr__(self):
        return f"IdealGas(R={self.R:g}, gamma={self.gamma:g}, name={self.name!r})"

    # -- closed forms for all 12 independent pairs -------------------------
    def _T_from_P_s(self, P, s):
        return self.T_ref * math.exp((s + self.R * math.log(P / self.P_ref)) / self.cp)

    def _P_from_T_s(self, T, s):
        return self.P_ref * math.exp((self.cp * math.log(T / self.T_ref) - s) / self.R)

    def _evaluate(self, **pair):
        R, cp, cv = self.R, self.cp, self.cv
        a, b = sorted(pair, key=PROPS.index)
        va, vb = pair[a], pair[b]
        for k, v in pair.items():
            if k in ("T", "P", "v") and v <= 0:
                raise FluidError(f"IdealGas: {k} must be positive, got {v:g}")
        T = P = None
        if (a, b) == ("T", "P"):
            T, P = va, vb
        elif (a, b) == ("T", "v"):
            T = va
            P = R * T / vb
        elif (a, b) == ("T", "s"):
            T = va
            P = self._P_from_T_s(T, vb)
        elif (a, b) == ("P", "v"):
            P = va
            T = P * vb / R
        elif (a, b) == ("P", "u"):
            P = va
            T = vb / cv
        elif (a, b) == ("P", "h"):
            P = va
            T = vb / cp
        elif (a, b) == ("P", "s"):
            P = va
            T = self._T_from_P_s(P, vb)
        elif (a, b) == ("v", "u"):
            T = vb / cv
            P = R * T / va
        elif (a, b) == ("v", "h"):
            T = vb / cp
            P = R * T / va
        elif (a, b) == ("v", "s"):
            T = self.T_ref * math.exp((vb + R * math.log(R * self.T_ref / (self.P_ref * va))) / cv)
            P = R * T / va
        elif (a, b) == ("u", "s"):
            T = va / cv
            P = self._P_from_T_s(T, vb) if T > 0 else None
        elif (a, b) == ("h", "s"):
            T = va / cp
            P = self._P_from_T_s(T, vb) if T > 0 else None
        else:  # pragma: no cover - degenerate pairs are rejected before
            raise FluidError(f"IdealGas: unsupported pair ({a}, {b})")
        if T is None or T <= 0 or P is None or P <= 0:
            raise FluidError(f"IdealGas: ({a}={va:g}, {b}={vb:g}) gives no physical state (T and P must be positive)")
        return {"T": T, "P": P, "v": R * T / P, "u": cv * T, "h": cp * T,
                "s": cp * math.log(T / self.T_ref) - R * math.log(P / self.P_ref)}
