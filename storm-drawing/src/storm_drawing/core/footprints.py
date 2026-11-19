"""What each localization can be drawn as: the footprint palette.

Every backend draws a localization as a camera-facing square -- a billboard --
and a footprint decides what appears on it.  The Gaussian is the
reconstruction: summed with its neighbours, it is the super-resolution image,
and the only footprint the exporter writes.  Everything else in the palette is
an *alternative visualisation*: points, spheres, uncertainty ellipses and the
sprites napari-particles shipped, which show the same localizations differently
and claim nothing quantitative.

This module is the palette as data -- names, labels, how each one blends and
how much of the billboard it needs -- so a host can offer the choice before any
viewer exists.  The shader for each footprint lives with the backends, keyed by
the same name; ``_tests/test_footprint_palette.py`` keeps the two in step.

Host-free like the rest of ``napari_storm.core``: no Qt, no napari, no GL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BLEND_ADDITIVE",
    "BLEND_OPAQUE",
    "DEFAULT_MIN_SIZE_PX",
    "FOOTPRINTS",
    "FOOTPRINT_DISC",
    "FOOTPRINT_GAUSSIAN",
    "Footprint",
    "GROUP_RECONSTRUCTION",
    "GROUP_VISUALISATION",
    "PALETTE",
    "footprint_named",
    "validate_footprint",
    "validate_min_size_px",
]

#: Summed with the depth test off: overlaps add up.  What a density is.
BLEND_ADDITIVE = "additive"

#: Depth-tested and opaque: the nearer localization hides the farther one,
#: whatever order they are drawn in.  What a marker is.
BLEND_OPAQUE = "opaque"

GROUP_RECONSTRUCTION = "reconstruction"
GROUP_VISUALISATION = "visualisation"

#: Smallest on-screen diameter of a visualisation footprint's one-sigma
#: outline, in screen pixels, unless a host says otherwise.  The same floor
#: napari puts under its own Points markers.
DEFAULT_MIN_SIZE_PX = 2.0


@dataclass(frozen=True)
class Footprint:
    """One entry of the palette.

    Attributes:
        name: stable identifier, what `LayerAppearance.footprint` takes and a
            saved scene records.
        label: what a control shows.
        group: ``"reconstruction"`` or ``"visualisation"``.
        blend: ``"additive"`` or ``"opaque"``.
        extent_sigmas: half the edge of the square the footprint is drawn on,
            in sigmas of the widest localization.  3 is the whole Gaussian
            billboard (the Gaussian is cut at three sigma); a marker drawn at
            its one-sigma outline needs 1, and napari-particles' sprites keep
            the 2.5 they were drawn on before 3.1.
        uncertainty: the footprint draws each localization's own one-sigma
            ellipse, so with widths taken from the localization uncertainty it
            shows that uncertainty as its size and shape -- a ring becomes an
            uncertainty ellipse.  The others draw one shape per dataset.
        description: one sentence for a tooltip.
    """

    name: str
    label: str
    group: str
    blend: str
    extent_sigmas: float
    uncertainty: bool
    description: str

    @property
    def reconstruction(self):
        """True for the Gaussian, the one footprint that is a measurement."""
        return self.group == GROUP_RECONSTRUCTION

    @property
    def exportable(self):
        """Whether the image exporter writes this footprint.  Only the Gaussian."""
        return self.reconstruction


FOOTPRINT_GAUSSIAN = "gaussian"
FOOTPRINT_DISC = "disc"

#: The palette, in the order a control lists it.
PALETTE = (
    Footprint(
        FOOTPRINT_GAUSSIAN,
        "Gaussian (scientific)",
        GROUP_RECONSTRUCTION,
        BLEND_ADDITIVE,
        3.0,
        True,
        "Each localization as a Gaussian, summed: the reconstruction, and what "
        "an export writes.",
    ),
    # -- markers: opaque, depth-tested, drawn at the one-sigma outline
    Footprint(
        FOOTPRINT_DISC,
        "Points",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        True,
        "Opaque points, one sigma in radius; nearer points hide farther ones.",
    ),
    Footprint(
        "outlined",
        "Outlined points",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        True,
        "Points with a dark rim, so overlapping ones stay apart.",
    ),
    Footprint(
        "ring",
        "Rings",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        True,
        "The one-sigma outline alone: with widths from the uncertainty, an "
        "uncertainty ellipse per localization.",
    ),
    Footprint(
        "sphere",
        "Spheres",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        True,
        "Lit spheres, for depth perception in 3-D.",
    ),
    Footprint(
        "cross",
        "Crosses",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        False,
        "A plus sign per localization.",
    ),
    Footprint(
        "square",
        "Squares",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        False,
        "A filled square per localization.",
    ),
    Footprint(
        "diamond",
        "Diamonds",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        1.0,
        False,
        "A filled diamond per localization.",
    ),
    # -- the sprites napari-particles shipped, on the whole billboard
    Footprint(
        "dome",
        "Domes",
        GROUP_VISUALISATION,
        BLEND_OPAQUE,
        2.5,
        False,
        "napari-particles' sphere: brightest at the centre, cut at the rim.",
    ),
    Footprint(
        "glow",
        "Glow",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' particle: a bright core with a long halo.",
    ),
    Footprint(
        "bubble",
        "Bubbles",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' bubble: a shell, brightest at its rim.",
    ),
    Footprint(
        "bubble_thin",
        "Thin bubbles",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' second bubble: a thin bright rim.",
    ),
    Footprint(
        "airy",
        "Airy-like",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' airy: concentric rings fading outwards.",
    ),
    Footprint(
        "fresnel",
        "Fresnel",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' fresnel: a flat core with rippled edges.",
    ),
    Footprint(
        "fractal",
        "Julia fractal",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' fractal: a Julia set on every localization.",
    ),
    Footprint(
        "gaussian_poly",
        "Polynomial Gaussian",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        True,
        "napari-particles' gaussian2: a polynomial stand-in for the Gaussian.",
    ),
    Footprint(
        "tile",
        "Tiles",
        GROUP_VISUALISATION,
        BLEND_ADDITIVE,
        2.5,
        False,
        "napari-particles' none: the whole billboard, flat.",
    ),
)

#: Every footprint name, in palette order.
FOOTPRINTS = tuple(footprint.name for footprint in PALETTE)

_BY_NAME = {footprint.name: footprint for footprint in PALETTE}


def footprint_named(name):
    """The palette entry called *name*, or a ValueError naming the ones there are."""
    try:
        return _BY_NAME[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"footprint must be one of {FOOTPRINTS}, not {name!r}"
        ) from None


def validate_footprint(name):
    """*name*, if the palette has it; a ValueError otherwise."""
    return footprint_named(name).name


def validate_min_size_px(min_size_px):
    """*min_size_px* as a float, or a ValueError if it is not a size."""
    try:
        value = float(min_size_px)
    except (TypeError, ValueError):
        raise ValueError(f"min_size_px must be a number, not {min_size_px!r}") from None
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"min_size_px must be finite and >= 0, not {min_size_px!r}")
    return value
