"""Derived spatial bounds, and the percent-to-absolute mapping over them.

These were three lists of two floats mutated in place on
``DataToLayerInterface``, with an empty extent spelled ``[inf, -inf]`` and every
consumer expected to remember that. The arithmetic that maps a slider
percentage onto an axis is the same arithmetic that the auto-offset removal
turned out to hinge on (§3.5.1), so it is worth having in one tested place
rather than inlined at each call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Bounds", "EMPTY"]


@dataclass(frozen=True)
class Bounds:
    """A closed interval on one axis, in nanometres.

    An empty interval is ``(inf, -inf)``: the identity for :meth:`union`, which
    is why accumulating extents over datasets can start from it and needs no
    special case for the first one.
    """

    low: float = np.inf
    high: float = -np.inf

    @property
    def is_empty(self):
        return not (np.isfinite(self.low) and np.isfinite(self.high))

    @property
    def span(self):
        return 0.0 if self.is_empty else float(self.high - self.low)

    @property
    def centre(self):
        return None if self.is_empty else 0.5 * (self.low + self.high)

    def union(self, other):
        """The smallest interval containing both."""
        if isinstance(other, Bounds):
            low, high = other.low, other.high
        else:
            low, high = other
        return Bounds(min(self.low, low), max(self.high, high))

    @classmethod
    def of(cls, values):
        """Bounds of a set of values; empty for an empty set."""
        values = np.asarray(values)
        if values.size == 0:
            return EMPTY
        return cls(float(np.min(values)), float(np.max(values)))

    def percent_to_absolute(self, percent_pair):
        """Map a ``[low%, high%]`` pair onto this interval, in nanometres.

        ``absolute = low + percent / 100 * (high - low)``.

        The specialised form ``percent / 100 * high`` that this replaced was
        only correct while every axis had been translated to start at zero --
        the job the auto-offset used to do. An empty interval has nothing to
        map onto, so the percentages come back unchanged rather than becoming
        NaN.
        """
        percent = np.asarray(percent_pair, dtype=float)
        if self.is_empty:
            return percent
        return self.low + percent / 100.0 * self.span

    def as_tuple(self):
        return (self.low, self.high)


#: The empty interval; the identity for union.
EMPTY = Bounds()
