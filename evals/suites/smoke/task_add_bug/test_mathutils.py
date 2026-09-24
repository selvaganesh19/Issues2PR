"""Tests pinning the expected behaviour of :mod:`mathutils`.

The ``add`` tests fail against the shipped (buggy) implementation and pass once
the agent corrects it.
"""

from __future__ import annotations

from mathutils import add, multiply


def test_add_positive() -> None:
    assert add(2, 3) == 5


def test_add_with_zero() -> None:
    assert add(0, 7) == 7


def test_add_negative() -> None:
    assert add(-4, 1) == -3


def test_multiply() -> None:
    assert multiply(3, 4) == 12
