"""SI conversion constants. Multiply a value in the named unit to get SI: ``P1 = 100 * kPa``."""

K = 1.0
Pa = 1.0
kPa = 1e3
bar = 1e5
MPa = 1e6
atm = 101325.0
J = 1.0
kJ = 1e3
MJ = 1e6
L = 1e-3


def degC(t: float) -> float:
    """Convert a temperature in degrees Celsius to kelvin."""
    return t + 273.15
