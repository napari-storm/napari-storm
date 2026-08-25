"""Deterministic SMLM fixtures for benchmarking and regression tests.

Level 0 of docs/modernization-review.md calls for reproducible 100k / 1M / 5M
localization fixtures plus an extreme-Gaussian case, so that performance claims
are comparable between runs and between machines.

Everything here is seeded and depends only on numpy, so a fixture can be
regenerated exactly rather than shipped as a large binary.
"""
from __future__ import annotations

import numpy as np

from ..localization_dataset_types import LocalizationDataBaseClass

__all__ = [
    "FIXTURE_SIZES",
    "make_localizations",
    "make_dataset",
    "extreme_gaussian_dataset",
]

# The sizes named in the modernization plan.
FIXTURE_SIZES = {"100k": 100_000, "1M": 1_000_000, "5M": 5_000_000}

# A field of view deliberately *not* placed at the origin, so that anything
# which quietly assumes coordinates start at zero shows up immediately.
DEFAULT_ORIGIN_NM = (10_000.0, 10_000.0, 0.0)
DEFAULT_FOV_NM = (20_000.0, 20_000.0, 1_000.0)


def make_localizations(n, *, seed=0, zdim=False, origin=None, fov=None):
    """Return a record array of *n* localizations in nanometres.

    Points are drawn from a fixed number of Gaussian clusters so the data has
    realistic local density variation rather than being uniform noise — density
    is what drives fragment cost in the renderer.
    """
    origin = DEFAULT_ORIGIN_NM if origin is None else origin
    fov = DEFAULT_FOV_NM if fov is None else fov
    rng = np.random.default_rng(seed)

    n_clusters = max(1, n // 2_000)
    centres = rng.uniform(0, 1, size=(n_clusters, 3)) * np.asarray(fov)
    which = rng.integers(0, n_clusters, size=n)
    spread = np.asarray(fov) * 0.01
    points = centres[which] + rng.normal(0, 1, size=(n, 3)) * spread
    points += np.asarray(origin)

    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    locs = np.zeros(n, dtype=fields)
    locs["x_pos_nm"] = points[:, 0]
    locs["y_pos_nm"] = points[:, 1]
    if zdim:
        locs["z_pos_nm"] = points[:, 2]
    return np.rec.array(locs)


def make_dataset(n, *, seed=0, zdim=False, name=None, origin=None, fov=None):
    """A LocalizationDataBaseClass holding *n* deterministic localizations."""
    if name is None:
        name = f"fixture_{n}"
    return LocalizationDataBaseClass(
        make_localizations(n, seed=seed, zdim=zdim, origin=origin, fov=fov),
        name=name,
        zdim_present=zdim,
    )


def extreme_gaussian_dataset(n=1_000, *, seed=0):
    """A small dataset packed into a tiny area.

    Combined with a large FWHM this is the fragment/fill-rate stress case: the
    localization count is trivial, but every splat covers a large part of the
    screen, so cost is driven by overdraw rather than by geometry.
    """
    return make_dataset(
        n, seed=seed, name="extreme_gaussian", fov=(50.0, 50.0, 10.0)
    )
