"""Exception and warning classes (spec section 14).

Every error that carries structured data takes it as keyword arguments and stores it
as an attribute. Attributes not given default to ``None`` (single values) or ``()``
(collections).
"""

from __future__ import annotations


class ThermoCyclesError(Exception):
    """Base class of every error raised by thermo_cycles.

    Subclasses list their attributes in ``_fields`` as ``(name, default)`` pairs.
    """

    _fields: tuple = ()

    def __init__(self, message: str = "", **attrs):
        super().__init__(message)
        known = dict(self._fields)
        unknown = set(attrs) - set(known)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected attribute(s) {sorted(unknown)}; "
                f"accepted: {sorted(known)}"
            )
        for name, default in known.items():
            setattr(self, name, attrs.get(name, default))


class InputError(ThermoCyclesError, ValueError):
    """Bad user input; ``except ValueError`` still works."""


class CycleDefinitionError(ThermoCyclesError, ValueError):
    """A bad cycle definition (``register_cycle``, ``build_system``, relation helpers such as ``linear``);
    ``except ValueError`` still works (v0.1.1)."""


class UnknownCycleError(InputError):
    _fields = (("suggestions", ()),)


class UnknownInputError(InputError):
    _fields = (("name", None), ("suggestions", ()), ("accepted", ()))


class UnderdeterminedError(InputError):
    _fields = (("missing", None), ("given", ()), ("add_one_of", ()),
               ("example_sets", ()), ("determined", ()))


class OverdeterminedError(InputError):
    _fields = (("conflicting", ()), ("drop_one_of", ()))


class InconsistentInputError(InputError):
    _fields = (("name", None), ("given", None), ("implied", None))


class NoSolutionError(ThermoCyclesError):
    _fields = (("variables", ()), ("detail", None))


class AmbiguousSolutionError(ThermoCyclesError):
    _fields = (("variable", None), ("candidates", ()))


class ConvergenceError(ThermoCyclesError):
    _fields = (("variables", ()), ("residual", None))


class InvalidCycleError(ThermoCyclesError):
    _fields = (("problems", ()),)


class FluidError(ThermoCyclesError, ValueError):
    """Fluid model cannot evaluate the requested state."""


class UnsupportedPairError(FluidError):
    pass


class DegenerateStateError(FluidError):
    pass


class LimitViolationError(ThermoCyclesError):
    _fields = (("violated", ()),)


class NoFeasiblePointError(ThermoCyclesError):
    _fields = (("closest", None),)


class MissingDependencyError(ThermoCyclesError, ImportError):
    """An optional dependency (e.g. matplotlib) is not installed."""


class LimitViolationWarning(UserWarning):
    pass


class CycleValidityWarning(UserWarning):
    pass


class InputPlausibilityWarning(UserWarning):
    pass
