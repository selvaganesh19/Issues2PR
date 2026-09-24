"""Tiny arithmetic helpers used by the smoke eval task.

``add`` contains an intentional bug (it subtracts instead of adding) that the
agent is expected to fix so that :mod:`test_mathutils` passes.
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the sum of ``a`` and ``b``.

    NOTE: this implementation is intentionally buggy for the eval task.
    """
    return a - b  # BUG: should be a + b


def multiply(a: int, b: int) -> int:
    """Return the product of ``a`` and ``b``."""
    return a * b
