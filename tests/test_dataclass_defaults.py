"""v0.1.1 / py3.11: no dataclass field may use a mutable or unhashable default (CI failure on 3.11, run 44658530754).

Python 3.11 changed dataclasses to reject any default whose class has ``__hash__ is None`` (3.10 only rejected
list, dict and set); ``types.MappingProxyType`` is unhashable before 3.12, so ``at_bound: Mapping = MappingProxyType({})``
made ``import thermo_cycles.optimise`` raise ValueError on 3.11. These tests apply the 3.11 rule on every Python version.
"""

import dataclasses
import importlib
import pkgutil
import types

import pytest

import thermo_cycles

_OPT = importlib.import_module("thermo_cycles.optimise")  # the name ``thermo_cycles.optimise`` is the function
_UNSAFE = (list, dict, set, bytearray, types.MappingProxyType)


def _package_dataclasses():
    for info in pkgutil.walk_packages(thermo_cycles.__path__, "thermo_cycles."):
        mod = importlib.import_module(info.name)
        for obj in vars(mod).values():
            if dataclasses.is_dataclass(obj) and isinstance(obj, type) and obj.__module__ == mod.__name__:
                yield obj


def test_no_dataclass_field_has_a_mutable_or_unhashable_default():
    bad = []
    for cls in _package_dataclasses():
        for f in dataclasses.fields(cls):
            d = f.default
            if d is dataclasses.MISSING:
                continue
            # the Python 3.11 rule, plus the 3.10 types, checked on every version
            if isinstance(d, _UNSAFE) or type(d).__hash__ is None:
                bad.append(f"{cls.__module__}.{cls.__name__}.{f.name} = {type(d).__name__}")
    assert not bad, "use field(default_factory=...) for: " + ", ".join(bad)


def test_at_bound_default_is_a_fresh_read_only_empty_mapping():
    field = {f.name: f for f in dataclasses.fields(_OPT.OptimisationResult)}["at_bound"]
    assert field.default is dataclasses.MISSING
    assert field.default_factory is not dataclasses.MISSING
    a, b = field.default_factory(), field.default_factory()
    assert isinstance(a, types.MappingProxyType) and dict(a) == {}
    assert a is not b
    with pytest.raises(TypeError):
        a["x"] = "lower"
