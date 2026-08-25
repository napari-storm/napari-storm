"""Where a dataset sits in world space, as a value rather than an assumption.

Localizations are rendered in true nanometres and reference images are placed
by scale and translation, but nothing until now *named* that relationship. The
auto-offset defect (§3.5) was possible precisely because the placement of a
dataset was an implicit, mutable, shared variable rather than a property of the
dataset with a type.

This is the minimal form: an anisotropic scale and a translation, per axis, in
nanometres. It is deliberately not an affine — rotation and landmark
registration are Level 4, and a matrix nobody can fill in would be a worse
answer than an honest scale-and-shift that everything can.

The identity transform is the default and costs nothing: :meth:`apply` returns
its input unchanged rather than multiplying by ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WorldTransform"]


@dataclass(frozen=True)
class WorldTransform:
    """Scale then translate, per axis, into world nanometres.

    Attributes:
        scale: multiplier per axis, keyed ``x``/``y``/``z``.
        translation_nm: offset per axis, applied after scaling.
    """

    scale: tuple = (1.0, 1.0, 1.0)
    translation_nm: tuple = (0.0, 0.0, 0.0)

    #: Order the tuples are in.  Not the renderer's column order -- see
    #: ``localization_table`` on why axis order lives at the render boundary.
    AXES = ("x", "y", "z")

    @property
    def is_identity(self):
        return (
            tuple(self.scale) == (1.0, 1.0, 1.0)
            and tuple(self.translation_nm) == (0.0, 0.0, 0.0)
        )

    def _index(self, axis):
        try:
            return self.AXES.index(axis)
        except ValueError:
            raise KeyError(f"unknown axis {axis!r}") from None

    def apply_axis(self, axis, values):
        """Transform one coordinate column.  Identity returns it unchanged."""
        index = self._index(axis)
        scale = float(self.scale[index])
        translation = float(self.translation_nm[index])
        if scale == 1.0 and translation == 0.0:
            return values
        return np.asarray(values, dtype=np.float32) * scale + translation

    def inverse_axis(self, axis, values):
        """Map world nanometres back onto the dataset's own coordinates.

        Needed wherever a bound expressed in world space has to be compared
        against untransformed data -- a render-range filter, for instance.
        """
        index = self._index(axis)
        scale = float(self.scale[index])
        translation = float(self.translation_nm[index])
        if scale == 0:
            raise ValueError(f"transform scales {axis!r} to zero; not invertible")
        if scale == 1.0 and translation == 0.0:
            return values
        return (np.asarray(values, dtype=np.float64) - translation) / scale

    def with_translation(self, **axes):
        """A copy with some translations replaced, e.g. ``with_translation(x=10)``."""
        translation = list(self.translation_nm)
        for axis, value in axes.items():
            translation[self._index(axis)] = float(value)
        return WorldTransform(tuple(self.scale), tuple(translation))

    def with_scale(self, **axes):
        """A copy with some scales replaced."""
        scale = list(self.scale)
        for axis, value in axes.items():
            scale[self._index(axis)] = float(value)
        return WorldTransform(tuple(scale), tuple(self.translation_nm))


#: Shared instance; every dataset starts here.
IDENTITY = WorldTransform()
