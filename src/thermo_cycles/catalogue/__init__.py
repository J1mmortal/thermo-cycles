"""Built-in cycle catalogue (spec section 8.2). Importing this package registers the cycles."""

from . import piston  # noqa: F401  (registers otto, diesel, dual, atkinson)
from .piston import ATKINSON, DIESEL, DUAL, OTTO
from . import gas_turbine  # noqa: F401  (registers brayton, brayton_regen)
from .gas_turbine import BRAYTON, BRAYTON_REGEN

__all__ = ["OTTO", "DIESEL", "DUAL", "ATKINSON", "BRAYTON", "BRAYTON_REGEN"]
