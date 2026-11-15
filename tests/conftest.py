"""Shared fixtures. The cycle registry is restored after every test, so a cycle registered by one test
(custom cycles, definition tests) never leaks into ``list_cycles()`` of another."""
import pytest

from thermo_cycles.model import cycle_def


@pytest.fixture(autouse=True)
def _restore_cycle_registry():
    before = dict(cycle_def._REGISTRY)
    yield
    cycle_def._REGISTRY.clear()
    cycle_def._REGISTRY.update(before)
