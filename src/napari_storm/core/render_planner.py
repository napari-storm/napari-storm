"""Turning a dataset and its settings into a :class:`RenderRequest`.

This is the "render planner" of §4: it decides *what* should be drawn and hands
the result to a backend, which decides *how*. Everything here is numpy and
value objects, so the decision can be tested without a viewer — which matters,
because this is where the Gaussian sizing and intensity weighting live, and
those are science, not presentation.

The arithmetic is carried over unchanged from the widget it used to live in.
What is new is that it no longer reaches into a Qt-configured object for its
settings or into a dataset class for its columns: it takes a
:class:`GaussianSettings`, a :class:`~napari_storm.core.LocalizationTable` and a
few flags describing what the format actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localization_table import ACTIVE
from .renderer import DEFAULT_COLORMAP, Changed, RenderRequest
from .validation import (
    InvalidLocalizationData,
    require_positive_maximum,
    sanitize_positive,
)
from .world_transform import IDENTITY

__all__ = ["GaussianSettings", "DatasetTraits", "RenderPlanner"]

#: Percentile above which variable-mode sigmas are clipped, so a handful of
#: enormous uncertainties cannot set the scale for everything else.
PERCENTILE_CLIP = 99

#: Billboard edge length as a multiple of the largest sigma.
SIGMA_TO_SIZE_FACTOR = 5

#: Substituted for a zero, negative or non-finite uncertainty or photon count.
MIN_USABLE_UNCERTAINTY = 1e-3

#: The plane flat data is pinned to.  Mirrors `ns_constants.FLAT_DATA_Z_NM`,
#: which the host side imports; core must not import from the host package.
FLAT_DATA_Z_NM = 1.0


@dataclass(frozen=True)
class GaussianSettings:
    """How localizations are turned into Gaussians.

    The subset of the render configuration that affects *what* is drawn rather
    than how it is coloured. Extracted as a value object so planning can be
    exercised without constructing the Qt-backed configuration object.
    """

    #: 0 = one fixed size for every localization, 1 = size from uncertainty.
    mode: int = 0
    fixed_sigma_xy_nm: float = 20 / 2.354
    fixed_sigma_z_nm: float = 20 / 2.354
    var_psf_sigma_xy_nm: float = 300 / 2.354
    var_psf_sigma_z_nm: float = 700 / 2.354
    var_sigma_min_xy_nm: float = 10 / 2.354
    var_sigma_min_z_nm: float = 10 / 2.354
    #: Colour by depth instead of by intensity.  Fixed mode only.
    z_color_encoding: bool = False


@dataclass(frozen=True)
class DatasetTraits:
    """What a format actually recorded, as opposed to what it could have.

    Variable-size mode needs *some* measure of uncertainty. Which one is
    available is a property of the file, and the planner has to be told rather
    than guess from the presence of a column full of ones.
    """

    zdim_present: bool = False
    sigma_present: bool = False
    photon_count_present: bool = False
    pixel_size_nm: float = 1.0

    @property
    def uncertainty_defined(self):
        return self.sigma_present or self.photon_count_present


class RenderPlanner:
    """Builds render requests.  Holds no state beyond how to report repairs."""

    def __init__(self, on_repaired=None):
        """
        Args:
            on_repaired: ``callable(column, n_repaired, n_total)``, called when
                unusable uncertainty or photon values had to be substituted.
                The planner does not decide how loudly to say so.
        """
        self.on_repaired = on_repaired

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _repaired(self, values, name, minimum):
        """*values* with unusable entries replaced by *minimum*, and reported."""
        values = np.asarray(values)
        repaired, n_repaired = sanitize_positive(values, name, minimum)
        if n_repaired and self.on_repaired is not None:
            self.on_repaired(name, n_repaired, values.size)
        return repaired

    def _usable_sigma(self, rows, axis):
        """A selected sigma column, in nanometres, with unusable rows repaired.

        **The substitute is the median of the usable rows**, and the reasoning
        is worth keeping because two earlier answers were both wrong.

        This method and :meth:`sigmas` repair the same dead row for different
        questions. :meth:`sigmas` asks *how wide to draw a row whose width is
        unknown*, and the declared floor is a good answer: draw it minimally.
        This asks *how bright it should be*, and because the weight is
        ``1 / product``, the floor is the worst answer available -- the
        narrowest width is the weight-maximising one, so repairing to it makes
        every failed fit the brightest thing in the dataset.

        The original :data:`MIN_USABLE_UNCERTAINTY` sentinel was worse again,
        being absolute: it did not scale with the column, so the same dead row
        meant one width in a pixel-native file and a hundredth of it in a
        nanometre-native one, and the two disagreed on what was drawn.

        A median is unit-covariant, so it keeps the units agreeing, and it is
        the honest reading of a failed fit: no information about brightness,
        so take the typical value rather than an extreme one. A handful of dead
        rows then cannot set the scale that every other row is normalized
        against.
        """
        values = np.asarray(rows.sigma_nm(axis))
        usable = np.isfinite(values) & (values > 0)
        # A column with nothing usable is sanitize_positive's error to raise,
        # and its message names the column; the substitute is never reached.
        typical = float(np.median(values[usable])) if usable.any() else 1.0
        return self._repaired(values, rows.sigma_column_name(axis), typical)

    def _usable_photons(self, rows):
        """Selected photon counts with unusable rows repaired.

        Photons are a count, not a length, so the unit argument above does not
        apply and the original sentinel is still the right floor.
        """
        return self._repaired(rows.photons(), "photon_count", MIN_USABLE_UNCERTAINTY)

    def coordinates(self, rows, traits, transform=IDENTITY):
        """The ``(N, 3)`` float32 ``(z, y, x)`` array the renderer consumes.

        The one place the renderer's axis order is applied; the table keys
        positions by axis name so that this ordering lives here and nowhere
        else. 2-D data is pinned to the z = 1 plane, which is the convention
        reference images are placed against.

        **This used to return ``(z, x, y)``,** which is not napari's order, and
        the cost was paid three times over. A reconstruction was transposed
        against every ordinary layer sharing the viewer -- an embedding host's
        ROI shapes, a points overlay -- which is a misregistration rather than
        a cosmetic complaint. Reference images were transposed back to meet it
        (see ``pyqt.image_layer_controls``), and the exporter transposed
        coordinates the other way, so the image on screen disagreed with the
        OME-TIFF written from it. Worst of all it was silently wrong on its
        own terms: :meth:`sigmas` has always returned ``(z, y, x)`` and the
        backends pair the two column-wise, so an anisotropic PSF had its
        widths applied to the wrong lateral axis -- invisible while sigma_x
        and sigma_y agree, which in most data they nearly do.
        """
        n = rows.n
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, 1] = transform.apply_axis("y", rows.coordinate_nm("y"))
        coords[:, 2] = transform.apply_axis("x", rows.coordinate_nm("x"))
        if traits.zdim_present:
            coords[:, 0] = transform.apply_axis("z", rows.coordinate_nm("z"))
        else:
            coords[:, 0] = FLAT_DATA_Z_NM
        return coords

    # ------------------------------------------------------------------
    # Gaussian sizing and weighting
    # ------------------------------------------------------------------

    def values(self, rows, settings, traits):
        """Per-localization intensity, or depth when Z encoding is on."""
        if settings.mode == 0:
            values = np.ones(rows.n, dtype=np.float32)
        else:
            if not traits.uncertainty_defined:
                raise InvalidLocalizationData(
                    "variable-Gaussian mode needs either uncertainty or photon "
                    "counts, and this dataset carries neither"
                )
            values = self._variable_values(rows, settings, traits)

        if settings.z_color_encoding:
            if not traits.zdim_present:
                raise InvalidLocalizationData("no z coordinate to colour by")
            values = np.array(rows.coordinate_nm("z"), dtype=np.float32, copy=True)
            values = _normalized(values)

        return require_positive_maximum(values, "render values")

    def _variable_values(self, rows, settings, traits):
        """Unit-volume weighting: brighter where the Gaussian is tighter."""
        if traits.sigma_present:
            product = self._usable_sigma(rows, "x") * self._usable_sigma(rows, "y")
            if traits.zdim_present:
                product = product * self._usable_sigma(rows, "z")
        else:
            photons = self._usable_photons(rows)
            psf_xy = settings.var_psf_sigma_xy_nm / traits.pixel_size_nm
            sigma_xy = psf_xy / np.sqrt(photons)
            if traits.zdim_present:
                psf_z = settings.var_psf_sigma_z_nm / traits.pixel_size_nm
                product = sigma_xy**2 * (psf_z / np.sqrt(photons))
            else:
                product = sigma_xy**2

        values = _normalized(1.0 / product)
        # Map the 99th percentile to 1 so a few very tight localizations do not
        # push everything else into the bottom of the colour range.
        #
        # Guarded, because the percentile is zero whenever 99% of rows sit at
        # the minimum -- reachable from a fitter that writes one nominal width
        # for most rows and fits the rest. Dividing anyway produced NaN, and
        # the failure surfaced downstream as "render values has no positive
        # finite maximum", naming neither the column nor the cause. There is
        # simply nothing to clip against here: `_normalized` has already put
        # the values in [0, 1].
        ceiling = np.percentile(values, PERCENTILE_CLIP)
        if ceiling > 0:
            values = values / ceiling
        return values

    def sigmas(self, rows, settings, traits):
        """``((N, 3) normalized sigmas, billboard edge in nm)``."""
        n = rows.n
        if settings.mode == 0:
            sigma_nm = np.empty((n, 3), dtype=np.float32)
            sigma_nm[:, 0] = settings.fixed_sigma_z_nm
            sigma_nm[:, 1] = settings.fixed_sigma_xy_nm
            sigma_nm[:, 2] = settings.fixed_sigma_xy_nm
        elif traits.sigma_present:
            x = _clipped(rows.sigma_nm("x"), settings.var_sigma_min_xy_nm)
            y = _clipped(rows.sigma_nm("y"), settings.var_sigma_min_xy_nm)
            # A 2-D fit records no axial width, and reading one unconditionally
            # raised a bare AttributeError on an entirely ordinary file -- the
            # value branch already guarded on `zdim_present` and this one did
            # not. Absent, the axial width is the declared floor, which is what
            # a present-but-zero-filled column already clipped to.
            if rows.has_sigma_axis("z"):
                z = _clipped(rows.sigma_nm("z"), settings.var_sigma_min_z_nm)
            else:
                z = np.full(n, settings.var_sigma_min_z_nm, dtype=np.float64)
            sigma_nm = np.stack([z, y, x], axis=1).astype(np.float32)
        else:
            photons = self._usable_photons(rows)
            xy = _clipped(
                settings.var_psf_sigma_xy_nm / np.sqrt(photons),
                settings.var_sigma_min_xy_nm,
            )
            z = _clipped(
                settings.var_psf_sigma_z_nm / np.sqrt(photons),
                settings.var_sigma_min_z_nm,
            )
            sigma_nm = np.stack([z, xy, xy], axis=1).astype(np.float32)

        largest = np.max(sigma_nm)
        require_positive_maximum(np.asarray([largest]), "Gaussian sigma")
        return sigma_nm / largest, float(SIGMA_TO_SIZE_FACTOR * largest)

    # ------------------------------------------------------------------
    # The whole request
    # ------------------------------------------------------------------

    def plan(
        self,
        table,
        settings,
        traits,
        *,
        name,
        transform=IDENTITY,
        colormap=DEFAULT_COLORMAP,
        antialias=0.0,
        changed=Changed.EVERYTHING,
        size_limit=None,
        selection=ACTIVE,
    ):
        """Everything a backend needs to draw this dataset as it stands now.

        *size_limit*, when given, caps the billboard edge -- the screen-space
        budget of P0-04, applied here because it changes what is drawn and so
        belongs to planning rather than to a backend.

        *selection* chooses which rows to plan over. The renderer wants
        :data:`~napari_storm.core.localization_table.ACTIVE`, the rows it can
        afford to draw. An export wants
        :data:`~napari_storm.core.localization_table.FILTERED`, the rows the
        user actually selected, because budget thinning is an accommodation to
        the GPU and has no business in a saved result.
        """
        if settings.mode == 1 and not traits.uncertainty_defined:
            raise InvalidLocalizationData(
                "variable-Gaussian mode needs either uncertainty or photon "
                "counts, and this dataset carries neither"
            )
        if settings.z_color_encoding and not traits.zdim_present:
            raise InvalidLocalizationData("no z coordinate to colour by")
        rows = table.selection(selection)
        sigmas, size = self.sigmas(rows, settings, traits)
        if size_limit is not None:
            size = min(size, float(size_limit))
        return RenderRequest(
            coords=self.coordinates(rows, traits, transform),
            sigmas=sigmas,
            size=size,
            values=self.values(rows, settings, traits),
            name=name,
            colormap=colormap,
            antialias=antialias,
            active_ids=rows.ids,
            changed=changed,
        )


def _normalized(values):
    """Scale to ``[0, 1]``, or leave alone when every value is the same."""
    low, high = np.min(values), np.max(values)
    if high == low:
        return values
    return (values - low) / (high - low)


def _clipped(values, minimum):
    """Lift values to *minimum*, then clip the extreme tail.

    Both halves matter: the floor keeps a zero uncertainty from collapsing a
    Gaussian to nothing, and the percentile ceiling keeps a few enormous ones
    from setting the billboard size for the entire dataset.
    """
    values = np.asarray(values, dtype=np.float64).copy()
    values[values < minimum] = minimum
    ceiling = np.percentile(values, PERCENTILE_CLIP)
    values[values > ceiling] = ceiling
    return values
