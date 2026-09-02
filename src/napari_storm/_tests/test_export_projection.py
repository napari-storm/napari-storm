"""A 2-D export can project onto XZ and YZ, not only XY.

The rasterizer always sums over axis 0 and draws axes 1 and 2, so a projection
is a permutation of the (z, y, x) columns rather than a second code path.

The failure worth testing for is not the shape -- every permutation gives the
same shape -- but permuting the coordinates while leaving the widths behind.
That draws a Gaussian of the wrong extent in the right place, which looks
entirely plausible unless you measure it.
"""

import numpy as np
import pytest

from napari_storm.core.ome_export import ExportChannel
from napari_storm.core.raster import GaussianGrid, rasterize
from napari_storm.image_export import (
    PROJECTION_XY,
    PROJECTION_XZ,
    PROJECTION_YZ,
    project_bounds,
    project_channel,
    projection_axes,
)


def _channel(coords, sigmas):
    coords = np.asarray(coords, dtype=float)
    return ExportChannel(
        name="c",
        coords_nm=coords,
        sigmas_nm=np.broadcast_to(np.asarray(sigmas, float), coords.shape).copy(),
        values=np.ones(len(coords)),
    )


def test_the_xy_projection_is_what_exports_did_before():
    assert projection_axes(PROJECTION_XY) == (0, 1, 2)


def test_an_unknown_projection_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown projection"):
        projection_axes("diagonal")


def test_xy_leaves_the_channel_untouched():
    channel = _channel([[1.0, 2.0, 3.0]], [4.0, 5.0, 6.0])
    assert project_channel(channel, PROJECTION_XY) is channel


@pytest.mark.parametrize(
    "projection, expected_coords, expected_sigmas",
    [
        (PROJECTION_XZ, [2.0, 1.0, 3.0], [5.0, 4.0, 6.0]),
        (PROJECTION_YZ, [3.0, 1.0, 2.0], [6.0, 4.0, 5.0]),
    ],
)
def test_widths_are_permuted_with_the_coordinates(
    projection, expected_coords, expected_sigmas
):
    channel = _channel([[1.0, 2.0, 3.0]], [4.0, 5.0, 6.0])
    projected = project_channel(channel, projection)
    assert list(projected.coords_nm[0]) == expected_coords
    assert list(projected.sigmas_nm[0]) == expected_sigmas


def test_bounds_follow_the_same_permutation():
    bounds = ((0.0, 10.0), (0.0, 20.0), (0.0, 30.0))
    assert project_bounds(bounds, PROJECTION_XZ) == (
        (0.0, 20.0),
        (0.0, 10.0),
        (0.0, 30.0),
    )


def _rendered_extent(projection):
    """Rasterize one Gaussian that is wide in z only, and measure the result.

    Widths are 4 nm laterally and 40 nm axially, so whichever image axis z
    lands on has to come out ten times longer than the other.
    """
    channel = project_channel(_channel([[0.0, 0.0, 0.0]], [40.0, 4.0, 4.0]), projection)
    bounds = project_bounds(
        ((-150.0, 150.0), (-150.0, 150.0), (-150.0, 150.0)), projection
    )
    grid = GaussianGrid.covering(bounds, pixel_size_nm=2.0)
    image = rasterize(channel.coords_nm, channel.sigmas_nm, channel.values, grid)[0]
    lit = image > 0.01 * image.max()
    return lit.any(axis=1).sum(), lit.any(axis=0).sum()  # rows, columns


def test_xy_shows_the_axial_width_in_neither_image_axis():
    """Looking down z, a z-elongated Gaussian is round."""
    rows, columns = _rendered_extent(PROJECTION_XY)
    assert rows == pytest.approx(columns, rel=0.15)


def test_xz_puts_the_axial_width_on_the_row_axis():
    """Image is (z, x), so the long axis is rows."""
    rows, columns = _rendered_extent(PROJECTION_XZ)
    assert rows > 5 * columns


def test_yz_puts_the_axial_width_on_the_row_axis_too():
    """Image is (z, y): same row axis, different column axis."""
    rows, columns = _rendered_extent(PROJECTION_YZ)
    assert rows > 5 * columns
