"""Fluid models."""

from .base import Fluid, FluidBase
from .ideal_gas import IdealGas

AIR = IdealGas(R=287.0, gamma=1.4, name="air (cold-air standard)")

__all__ = ["Fluid", "FluidBase", "IdealGas", "AIR"]
