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

from napari_storm._dock_widget import napari_storm
from napari_storm.core.ome_export import ExportChannel
from napari_storm.core.raster import GaussianGrid, rasterize
from napari_storm.image_export import (
    PROJECTION_XY,
    PROJECTION_XZ,
    PROJECTION_YZ,
    ExportOptions,
    plan_from_widget,
    project_bounds,
    project_channel,
    projection_axes,
)
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


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


# ------------------------------------------------------- through the widget
#
# The permutation helpers above are exercised in isolation, which is how the
# bug they were meant to prevent still shipped: `export_bounds_nm` handed the
# planner ``(0, 0)`` for z, and after permutation that degenerate interval
# landed on an axis of the image rather than on the one being summed over.
# Nothing below reaches the helpers directly.


def _dataset_at_depth(z_low=200.0, z_high=400.0, n=200, anisotropic=False):
    """A 3-D dataset whose z coordinates are nowhere near zero."""
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")]
    if anisotropic:
        fields += [
            ("sigma_x_pixels", "f4"),
            ("sigma_y_pixels", "f4"),
            ("sigma_z_pixels", "f4"),
        ]
    locs = np.zeros(n, dtype=fields)
    # A distinct span per axis, so a transposition cannot pass unnoticed.
    locs["x_pos_nm"] = np.linspace(1_000, 3_000, n)
    locs["y_pos_nm"] = np.linspace(1_000, 7_000, n)
    locs["z_pos_nm"] = np.linspace(z_low, z_high, n)
    if anisotropic:
        locs["sigma_x_pixels"] = 20.0
        locs["sigma_y_pixels"] = 60.0
        locs["sigma_z_pixels"] = 100.0
    dataset = LocalizationDataBaseClass(
        np.rec.array(locs), name="deep", zdim_present=True
    )
    if anisotropic:
        dataset.sigma_present = True
    return dataset


def _widget_at_depth(make_napari_viewer, **kwargs):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_dataset_at_depth(**kwargs)])
    return widget


def _planned(widget, projection, pixel_size_nm=25.0):
    return plan_from_widget(
        widget, ExportOptions(pixel_size_nm=pixel_size_nm, projection=projection)
    )


def _drawn(plan):
    """The plan's first channel, rasterized onto its own grid."""
    channel = plan.channels[0]
    return rasterize(channel.coords_nm, channel.sigmas_nm, channel.values, plan.grid)


def test_xy_is_one_plane_of_y_by_x(make_napari_viewer):
    plan = _planned(_widget_at_depth(make_napari_viewer), PROJECTION_XY)

    nz, ny, nx = plan.grid.shape
    assert nz == 1
    # y spans 6000 nm and x spans 2000 nm at 25 nm/px.
    assert ny == pytest.approx(240, abs=2)
    assert nx == pytest.approx(80, abs=2)
    assert _drawn(plan).sum() > 0


@pytest.mark.parametrize("projection", [PROJECTION_XZ, PROJECTION_YZ])
def test_a_projection_onto_z_is_taller_than_one_pixel(make_napari_viewer, projection):
    """The regression: z bounds of (0, 0) made this exactly one row."""
    plan = _planned(_widget_at_depth(make_napari_viewer), projection)

    nz, rows, _columns = plan.grid.shape
    assert nz == 1  # still a projection, not a stack
    # 200 nm of z at 25 nm/px, and emphatically not 1.
    assert rows == pytest.approx(8, abs=2)


@pytest.mark.parametrize("projection", [PROJECTION_XZ, PROJECTION_YZ])
def test_localizations_away_from_the_origin_are_actually_drawn(
    make_napari_viewer, projection
):
    """The image used to be positioned at z=0 with every localization outside."""
    plan = _planned(_widget_at_depth(make_napari_viewer), projection)

    assert _drawn(plan).sum() > 0


def test_the_z_extent_follows_the_render_range_percentages(make_napari_viewer):
    widget = _widget_at_depth(make_napari_viewer, z_low=0.0, z_high=1_000.0)
    full = _planned(widget, PROJECTION_XZ).grid.shape[1]
    # Enough rows that halving them is a statement about the range rather than
    # about rounding -- the degenerate bounds this guards against gave 1.
    assert full > 20

    widget.render_range_slider_z_percent = np.array([0, 50])
    half = _planned(widget, PROJECTION_XZ).grid.shape[1]

    assert half == pytest.approx(0.5 * full, abs=1)


def test_widths_stay_attached_to_their_axes_through_the_planner(make_napari_viewer):
    """Permuting coordinates without their widths draws the wrong shape."""
    widget = _widget_at_depth(make_napari_viewer, anisotropic=True)
    widget.render_config.gaussian_mode = 1

    xy = _planned(widget, PROJECTION_XY).channels[0]
    xz = _planned(widget, PROJECTION_XZ).channels[0]

    # XZ images (z, x): z moves from column 0 to column 1, width and all.
    assert xz.coords_nm[0, 1] == pytest.approx(xy.coords_nm[0, 0])
    assert xz.sigmas_nm[0, 1] == pytest.approx(xy.sigmas_nm[0, 0])
    assert xz.coords_nm[0, 2] == pytest.approx(xy.coords_nm[0, 2])
    assert xz.sigmas_nm[0, 2] == pytest.approx(xy.sigmas_nm[0, 2])


def test_flat_data_still_collapses_to_a_single_row(make_napari_viewer):
    """A 2-D dataset has no z window, and must not acquire one."""
    locs = np.zeros(50, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 1_000, 50)
    locs["y_pos_nm"] = np.linspace(0, 2_000, 50)
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(
        [LocalizationDataBaseClass(np.rec.array(locs), name="flat", zdim_present=False)]
    )

    assert _planned(widget, PROJECTION_XZ).grid.shape[1] == 1
