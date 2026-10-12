from dataclasses import dataclass, field

import numpy as np

from .core.footprints import DEFAULT_MIN_SIZE_PX, FOOTPRINT_GAUSSIAN
from .core.traces import COLOR_BY_TRACE, DEFAULT_TRACE_WIDTH_PX
from .memory_budget import MAX_SPLAT_FRACTION_OF_FOV, default_render_budget_mb
from .ns_constants import (
    DEFAULT_COLOR_CODING_COLORMAP,
    DEFAULT_FIXED_FWHM_XY_NM,
    DEFAULT_FIXED_FWHM_Z_NM,
    DEFAULT_GRID_LINE_DISTANCE_UM,
    DEFAULT_SCALEBAR_NM,
    DEFAULT_VAR_MIN_FWHM_XY_NM,
    DEFAULT_VAR_MIN_FWHM_Z_NM,
    DEFAULT_VAR_PSF_FWHM_XY_NM,
    DEFAULT_VAR_PSF_FWHM_Z_NM,
    FWHM_TO_SIGMA,
)


@dataclass
class RenderConfig:
    """Configuration for rendering localizations.

    Replaces the pattern where DataToLayerInterface reaches into
    self.parent.* to read ~15 render-related attributes.
    """

    # Gaussian render mode: 0 = fixed, 1 = variable
    gaussian_mode: int = 0

    # Fixed gaussian sigma (nm)
    fixed_sigma_xy_nm: float = DEFAULT_FIXED_FWHM_XY_NM / FWHM_TO_SIGMA
    fixed_sigma_z_nm: float = DEFAULT_FIXED_FWHM_Z_NM / FWHM_TO_SIGMA

    # Variable gaussian PSF sigma (nm)
    var_psf_sigma_xy_nm: float = DEFAULT_VAR_PSF_FWHM_XY_NM / FWHM_TO_SIGMA
    var_psf_sigma_z_nm: float = DEFAULT_VAR_PSF_FWHM_Z_NM / FWHM_TO_SIGMA

    # Variable gaussian minimum sigma (nm)
    var_sigma_min_xy_nm: float = DEFAULT_VAR_MIN_FWHM_XY_NM / FWHM_TO_SIGMA
    var_sigma_min_z_nm: float = DEFAULT_VAR_MIN_FWHM_Z_NM / FWHM_TO_SIGMA

    # Colour coding: colour every localization by one property -- "x", "y",
    # "z" or a column name -- instead of the channel colormap.  None is off.
    color_by: str | None = None
    color_colormap: str = DEFAULT_COLOR_CODING_COLORMAP
    # Property values at the two ends of the colormap, shared by every
    # dataset.  None follows the data: see DataToLayerInterface.color_limits.
    color_limits: tuple[float, float] | None = None

    # What every localization is drawn as: a name from core.footprints.PALETTE.
    # The scientific Gaussian, or an alternative visualisation.
    footprint: str = FOOTPRINT_GAUSSIAN
    # Smallest on-screen diameter of a visualisation, in screen pixels.
    min_size_px: float = DEFAULT_MIN_SIZE_PX

    # Traces: connect each trace's localizations in time order, for every
    # dataset that identifies its traces.  See core.traces.
    traces: bool = False
    trace_color_by: str = COLOR_BY_TRACE
    trace_width_px: float = DEFAULT_TRACE_WIDTH_PX

    # Dimensionality
    zdim: bool | None = None

    # Render range sliders (percent, 2-element arrays)
    range_x_percent: np.ndarray = field(default_factory=lambda: np.arange(2) * 100)
    range_y_percent: np.ndarray = field(default_factory=lambda: np.arange(2) * 100)
    range_z_percent: np.ndarray = field(default_factory=lambda: np.arange(2) * 100)

    # Grid plane
    grid_plane_line_distance_um: float = DEFAULT_GRID_LINE_DISTANCE_UM
    # Issue #38: how far past the data the grid is allowed to run, as a
    # percentage of each axis' span, added at both ends.  0 keeps the old
    # behaviour of stopping exactly at the render range.
    grid_plane_margin_percent: float = 0.0

    # Scalebar
    scalebar_size_nm: int = DEFAULT_SCALEBAR_NM
    scalebar_enabled: bool = False

    # Resource limits (P0-04).  render_budget_mb caps host-side render arrays
    # across all loaded datasets; 0 disables it.  max_splat_fraction_of_fov caps
    # a single Gaussian's on-screen footprint, which memory cannot bound.
    render_budget_mb: float = field(default_factory=default_render_budget_mb)
    max_splat_fraction_of_fov: float = MAX_SPLAT_FRACTION_OF_FOV

    @property
    def z_color_encoding(self):
        """1 while colouring by z, else 0.  Kept for embedders written before
        colour coding took any property; new code sets `color_by`."""
        return int(self.color_by == "z")

    @z_color_encoding.setter
    def z_color_encoding(self, value):
        if value:
            self.color_by = "z"
        elif self.color_by == "z":
            self.color_by = None
