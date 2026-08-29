"""Explicit limits on render memory and screen-space Gaussian cost.

Two independent costs can take the plugin down, and neither is bounded by the
other (P0-04 and §3.2 of ``docs/modernization-review.md``):

* **Host/GPU memory** grows with localization count.  Every localization becomes
  six vertices with repeated centre, sigma and value attributes, measured at
  :data:`RENDER_BYTES_PER_LOCALIZATION` bytes of host-side layer arrays.
* **Fragment cost** grows with the on-screen area of each splat and is
  independent of how many localizations are loaded.  The measured
  extreme-Gaussian fixture draws one splat covering ~12 000x the field of view;
  no memory limit can bound that.

Both limits degrade rather than fail: over-budget datasets render a uniform
subsample, and over-large splats are clamped to a fraction of the field of view.
"""

from __future__ import annotations

import os

__all__ = [
    "RENDER_BYTES_PER_LOCALIZATION",
    "DEFAULT_RENDER_BUDGET_MB",
    "RENDER_BUDGET_ENV_VAR",
    "MAX_SPLAT_FRACTION_OF_FOV",
    "default_render_budget_mb",
    "max_localizations_for_budget",
    "render_bytes_for",
    "cap_splat_size_nm",
]

#: Host-side bytes per localization held by one ``Particles`` layer.  Measured,
#: not estimated -- ``test_particles_host_arrays_use_352_bytes_per_localization``
#: fails if the renderer's dtypes regress.  Excludes VisPy buffers and GPU
#: copies, so the true footprint is higher and this budget is optimistic.
RENDER_BYTES_PER_LOCALIZATION = 352

#: Default ceiling on host-side render arrays across all loaded datasets.
#: 2 GB is roughly 6.1M localizations, which is above the largest benchmark
#: fixture and below what a 16 GB machine will tolerate alongside napari itself.
DEFAULT_RENDER_BUDGET_MB = 2048.0

#: Set to a number of megabytes to override the default; set to 0 to disable the
#: budget entirely (useful on a workstation with plenty of RAM).
RENDER_BUDGET_ENV_VAR = "NAPARI_STORM_RENDER_BUDGET_MB"

#: A single splat may not span more than this fraction of the field of view.
#: The Gaussian is clipped at the quad edge, so this is a visible change only
#: for splats already far larger than anything they are drawn on top of.
MAX_SPLAT_FRACTION_OF_FOV = 0.5


def default_render_budget_mb():
    """Budget in megabytes, honouring :data:`RENDER_BUDGET_ENV_VAR`.

    An unparseable or negative value falls back to the default rather than
    raising: this is read while a viewer is starting up, and a typo in an
    environment variable should not prevent the plugin from loading.
    """
    raw = os.environ.get(RENDER_BUDGET_ENV_VAR)
    if raw is None:
        return DEFAULT_RENDER_BUDGET_MB
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_RENDER_BUDGET_MB
    if value < 0:
        return DEFAULT_RENDER_BUDGET_MB
    return value


def render_bytes_for(n_localizations):
    """Host-side render bytes a dataset of *n_localizations* will occupy."""
    return int(n_localizations) * RENDER_BYTES_PER_LOCALIZATION


def max_localizations_for_budget(budget_mb):
    """Largest localization count that fits in *budget_mb*.

    Returns ``None`` when the budget is disabled (``0`` or ``None``), meaning
    "no limit"; otherwise always at least 1, so a budget set absurdly low
    degrades to a nearly empty layer instead of an empty one that cannot be
    constructed at all.
    """
    if not budget_mb:
        return None
    allowed = int(float(budget_mb) * 1e6 // RENDER_BYTES_PER_LOCALIZATION)
    return max(allowed, 1)


def cap_splat_size_nm(size_nm, fov_nm, fraction=MAX_SPLAT_FRACTION_OF_FOV):
    """Clamp a billboard edge length to *fraction* of the field of view.

    *fov_nm* of ``None`` or a non-positive value means the extent is not known
    yet -- during the very first layer creation, for instance -- and the size is
    returned unchanged rather than clamped against a meaningless bound.
    """
    if fov_nm is None or not fov_nm > 0 or not fraction > 0:
        return float(size_nm), False
    limit = float(fov_nm) * float(fraction)
    if float(size_nm) <= limit:
        return float(size_nm), False
    return limit, True
