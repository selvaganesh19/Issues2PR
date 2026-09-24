"""A tiny math package with an intentional bug for the agent demo."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the sum of ``a`` and ``b``.

    NOTE: This implementation is intentionally buggy (it subtracts) so the
    accompanying test suite fails until the bug is fixed.
    """
    return a - b
