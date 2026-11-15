"""Built-in cycle catalogue (spec section 8.2). Importing this package registers the cycles."""

from . import piston  # noqa: F401  (registers otto)
from .piston import OTTO

__all__ = ["OTTO"]
