from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .memory_budget import (MAX_SPLAT_FRACTION_OF_FOV,
                            default_render_budget_mb)
from .ns_constants import (
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

    # Z color encoding: False/0 = off, True/1 = on
    z_color_encoding: int = 0

    # Dimensionality
    zdim: Optional[bool] = None

    # Render range sliders (percent, 2-element arrays)
    range_x_percent: np.ndarray = field(
        default_factory=lambda: np.arange(2) * 100
    )
    range_y_percent: np.ndarray = field(
        default_factory=lambda: np.arange(2) * 100
    )
    range_z_percent: np.ndarray = field(
        default_factory=lambda: np.arange(2) * 100
    )

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
