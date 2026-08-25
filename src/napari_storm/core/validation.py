"""Guards on scientific input, before it reaches arithmetic that cannot cope.

Two classes of bad value get into an SMLM file and then into the render maths:

* **Non-finite positions.** A NaN coordinate is not a measurement. Left in, it
  becomes a NaN vertex, and a NaN vertex poisons the bounding box, the camera
  framing and the mesh the GPU is handed.

* **Zero, negative or non-finite uncertainty and photon counts.** These reach
  ``1 / (sigma_x * sigma_y * sigma_z)`` and ``psf / sqrt(photons)`` directly.
  Zero gives infinity, which the subsequent min-max normalization turns into
  NaN for *every* localization -- one bad row silently blanks the channel.

The old guard on all of this was `assert`, which vanishes under ``python -O``:
the check most needed on a malformed file is the one not present in an
optimized interpreter. These raise instead, and say which column was wrong and
how many rows.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "InvalidLocalizationData",
    "non_finite_mask",
    "sanitize_positive",
    "require_positive_maximum",
]


class InvalidLocalizationData(ValueError):
    """Input that cannot be rendered, with a message naming the column."""


def non_finite_mask(*columns):
    """Rows where any of *columns* is NaN or infinite."""
    bad = None
    for values in columns:
        column_bad = ~np.isfinite(np.asarray(values, dtype=np.float64))
        bad = column_bad if bad is None else (bad | column_bad)
    return bad if bad is not None else np.zeros(0, dtype=bool)


def sanitize_positive(values, name, minimum):
    """Replace non-finite and non-positive entries with *minimum*.

    Returns ``(sanitized, n_repaired)``. Repairing rather than raising is the
    right call for uncertainty and photon counts: a handful of zero-photon rows
    in a million is a detector artefact, not a reason to refuse the file. The
    count comes back so the caller can say so out loud.

    Raises :class:`InvalidLocalizationData` when *every* entry is unusable,
    because there is then nothing to normalize against and the result would be
    an entirely blank channel.
    """
    values = np.asarray(values)
    usable = np.isfinite(values) & (values > 0)
    n_repaired = int(values.size - np.count_nonzero(usable))
    if n_repaired == 0:
        return values, 0
    if not usable.any():
        raise InvalidLocalizationData(
            f"every {name} value is zero, negative or non-finite; "
            f"this dataset cannot be rendered in variable-Gaussian mode"
        )
    return np.where(usable, values, minimum), n_repaired


def require_positive_maximum(values, name):
    """Fail loudly when a value array has nothing to normalize against.

    Previously `assert np.max(values) > 0`, which is absent under ``python -O``
    and reports nothing about the cause when it is present.
    """
    values = np.asarray(values)
    if values.size == 0:
        raise InvalidLocalizationData(f"no {name} values to render")
    largest = np.max(values)
    if not np.isfinite(largest) or largest <= 0:
        raise InvalidLocalizationData(
            f"{name} has no positive finite maximum (largest is {largest}); "
            f"nothing can be normalized against it"
        )
    return values
