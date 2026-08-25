import numpy as np
import pytest

from napari_storm.ns_constants import (
    DEFAULT_FIXED_FWHM_XY_NM,
    DEFAULT_FIXED_FWHM_Z_NM,
    DEFAULT_GRID_LINE_DISTANCE_UM,
    FWHM_TO_SIGMA,
)
from napari_storm.render_config import RenderConfig


def test_defaults():
    rc = RenderConfig()
    assert rc.gaussian_mode == 0
    assert rc.fixed_sigma_xy_nm == pytest.approx(
        DEFAULT_FIXED_FWHM_XY_NM / FWHM_TO_SIGMA
    )
    assert rc.fixed_sigma_z_nm == pytest.approx(
        DEFAULT_FIXED_FWHM_Z_NM / FWHM_TO_SIGMA
    )
    assert rc.z_color_encoding == 0
    assert rc.zdim is None
    assert rc.scalebar_enabled is False
    assert rc.grid_plane_line_distance_um == pytest.approx(
        DEFAULT_GRID_LINE_DISTANCE_UM
    )
    np.testing.assert_array_equal(rc.range_x_percent, [0, 100])
    np.testing.assert_array_equal(rc.range_y_percent, [0, 100])
    np.testing.assert_array_equal(rc.range_z_percent, [0, 100])


def test_independent_range_arrays():
    """Each RenderConfig instance should have independent range arrays."""
    rc1 = RenderConfig()
    rc2 = RenderConfig()
    rc1.range_x_percent[0] = 25
    assert rc2.range_x_percent[0] == 0


def test_mutations():
    rc = RenderConfig()
    rc.gaussian_mode = 1
    assert rc.gaussian_mode == 1
    rc.zdim = True
    assert rc.zdim is True
    rc.scalebar_enabled = True
    assert rc.scalebar_enabled is True
    rc.scalebar_size_nm = 1000
    assert rc.scalebar_size_nm == 1000
