"""Built-in cycle catalogue (spec section 8.2). Importing this package registers the cycles."""

from . import piston  # noqa: F401  (registers otto)
from .piston import OTTO
from . import gas_turbine  # noqa: F401  (registers brayton, brayton_regen)
from .gas_turbine import BRAYTON, BRAYTON_REGEN

__all__ = ["OTTO", "BRAYTON", "BRAYTON_REGEN"]
