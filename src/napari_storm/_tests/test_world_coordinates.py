"""Localizations render in true nanometres, and layers never move each other.

Covers register item P0-06 and the reasoning in docs/modernization-review.md
section 3.5.1: the auto-offset was introduced so multicolour datasets would
overlap, but a translation common to every dataset cannot affect their relative
alignment.  These tests pin down both halves -- overlap still holds, and loading
one dataset no longer displaces another.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.DataToLayerInterface import DataToLayerInterface
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


def _channel(name, x0, x1, y0, y1, n=50):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(x0, x1, n)
    locs["y_pos_nm"] = np.linspace(y0, y1, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _asymmetric_channel(name="asymmetric", zdim=False):
    """Cartesian data whose X and Y extents cannot be mistaken for each other."""
    axis_points = 21
    x = np.repeat(np.linspace(10_000, 12_000, axis_points), axis_points)
    y = np.tile(np.linspace(40_000, 50_000, axis_points), axis_points)
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    locs = np.zeros(x.size, dtype=fields)
    locs["x_pos_nm"] = x
    locs["y_pos_nm"] = y
    if zdim:
        locs["z_pos_nm"] = np.linspace(-500, 500, x.size)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=zdim)


# ------------------------------------------------------------- napari's axes


def test_coordinates_are_in_napari_axis_order(make_napari_viewer):
    """A reconstruction must not be transposed against an ordinary layer.

    Found by an embedding host, not by us, and that is the whole point: inside
    our own dock widget the old ``(z, x, y)`` order was self-consistent and so
    invisible. Share the viewer with anything else -- a host's ROI shapes, a
    points overlay, our own reference images -- and the reconstruction came out
    transposed against it, which is a misregistration rather than a cosmetic
    complaint.

    Pinned against a napari Points layer built from the same localizations,
    because the claim is about agreeing with napari rather than about any
    convention of ours.
    """
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    ds = _asymmetric_channel()  # x spans 2_000 nm, y spans 10_000 nm
    widget.get_dataset_from_test_mode([ds])

    coords = widget.data_to_layer_itf.get_coords_from_locs(ds)
    points = viewer.add_points(
        np.stack([np.zeros(len(ds.x_pos_nm)), ds.y_pos_nm, ds.x_pos_nm], axis=1),
        size=1,
    )
    extent = points.extent.world

    for axis in (1, 2):
        assert coords[:, axis].max() - coords[:, axis].min() == pytest.approx(
            extent[1][axis] - extent[0][axis], abs=1
        )
    # And say it the blunt way too: the wide axis is x, and x is the last column.
    assert coords[:, 2].max() - coords[:, 2].min() == pytest.approx(2_000, abs=1)
    assert coords[:, 1].max() - coords[:, 1].min() == pytest.approx(10_000, abs=1)


def test_sigmas_and_coordinates_share_an_axis_order(make_napari_viewer):
    """An anisotropic PSF must be drawn wide along the axis it is wide on.

    The backends broadcast sigmas against coords and upload them column-wise,
    so the two disagreeing was not a documentation problem: a 300 nm x-width
    was applied along y. It stayed hidden because sigma_x and sigma_y are
    nearly equal in most real data.
    """
    from napari_storm.core import (
        DatasetTraits,
        GaussianSettings,
        LocalizationTable,
        RenderPlanner,
    )

    n = 100
    records = np.rec.array(
        np.zeros(
            n,
            dtype=[
                ("x_pos_nm", "f4"),
                ("y_pos_nm", "f4"),
                ("z_pos_nm", "f4"),
                ("sigma_x_pixels", "f4"),
                ("sigma_y_pixels", "f4"),
                ("sigma_z_pixels", "f4"),
            ],
        )
    )
    records.x_pos_nm = np.linspace(0, 1_000, n)
    records.y_pos_nm = np.linspace(0, 100, n)
    records.sigma_x_pixels = 300.0  # wide along x
    records.sigma_y_pixels = 50.0  # narrow along y
    records.sigma_z_pixels = 100.0

    request = RenderPlanner().plan(
        LocalizationTable(records, sigma_scale_nm=1.0),
        GaussianSettings(mode=1),
        DatasetTraits(zdim_present=True, sigma_present=True, pixel_size_nm=1.0),
        name="anisotropic",
    )

    # Column 2 carries x for both, so it must carry the larger width.
    assert request.sigmas[0, 2] > request.sigmas[0, 1]


# ------------------------------------------------------- percent_to_absolute


def test_percent_to_absolute_spans_the_axis():
    assert DataToLayerInterface.percent_to_absolute([10_000, 12_000], [0, 100]) == (
        pytest.approx([10_000, 12_000])
    )
    assert DataToLayerInterface.percent_to_absolute([10_000, 12_000], [50, 50]) == (
        pytest.approx([11_000, 11_000])
    )


def test_percent_to_absolute_matches_the_old_offset_arithmetic():
    """The new form must be the old one generalized, not a behaviour change.

    Old:  pct/100 * max_offsetspace - offset,  with offset = -min_raw
    New:  min_raw + pct/100 * (max_raw - min_raw)
    """
    min_raw, max_raw = 10_000.0, 12_000.0
    offset = -min_raw
    max_offsetspace = max_raw - min_raw  # what render_range held under the offset

    for pct in ([0, 100], [25, 75], [10, 90], [0, 0], [100, 100]):
        old = np.asarray(pct) / 100 * np.ones(2) * max_offsetspace - offset
        new = DataToLayerInterface.percent_to_absolute([min_raw, max_raw], pct)
        assert new == pytest.approx(old)


def test_percent_to_absolute_tolerates_an_unset_axis():
    """Ranges start as [inf, -inf]; that must not produce nan bounds."""
    out = DataToLayerInterface.percent_to_absolute([np.inf, -np.inf], [0, 100])
    assert np.all(np.isfinite(out))


# ------------------------------------------------------------- world frames


def test_localizations_render_at_their_true_position(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    ds = _channel("A", 10_000, 12_000, 10_000, 12_000)
    widget.get_dataset_from_test_mode([ds])

    coords = widget.data_to_layer_itf.get_coords_from_locs(ds)
    assert coords[:, 1].min() == pytest.approx(10_000, abs=1)
    assert coords[:, 2].min() == pytest.approx(10_000, abs=1)


def test_channels_overlap(make_napari_viewer):
    """The behaviour the offset was introduced to provide.

    Two channels covering the same field of view must land on top of each other
    without any translation being applied.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    red = _channel("red", 10_000, 12_000, 10_000, 12_000)
    green = _channel("green", 10_000, 12_000, 10_000, 12_000)
    widget.get_dataset_from_test_mode([red, green])

    itf = widget.data_to_layer_itf
    np.testing.assert_allclose(
        itf.get_coords_from_locs(red), itf.get_coords_from_locs(green), atol=1e-3
    )


def test_channel_separation_is_preserved(make_napari_viewer):
    """A real 200 nm offset between channels must survive rendering."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    red = _channel("red", 10_000, 12_000, 10_000, 12_000)
    green = _channel("green", 10_200, 12_200, 10_000, 12_000)
    widget.get_dataset_from_test_mode([red, green])

    itf = widget.data_to_layer_itf
    # The offset between the two channels is in x, the last column.
    separation = (
        itf.get_coords_from_locs(green)[:, 2] - itf.get_coords_from_locs(red)[:, 2]
    )
    np.testing.assert_allclose(separation, 200.0, atol=1e-2)


def test_loading_a_second_dataset_does_not_move_the_first(make_napari_viewer):
    """P0-06.  This is the regression that silently broke image alignment.

    Dataset B sits closer to the origin than A.  Under the auto-offset that
    shifted A by 7000 nm relative to any already-aligned reference image.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    itf = widget.data_to_layer_itf

    a = _channel("A", 10_000, 12_000, 10_000, 12_000)
    widget.get_dataset_from_test_mode([a])
    before = itf.get_coords_from_locs(a).copy()

    b = _channel("B", 3_000, 5_000, 3_000, 5_000)
    widget.get_dataset_from_test_mode([a, b])
    after = itf.get_coords_from_locs(a)

    np.testing.assert_allclose(before, after, atol=1e-3)


@pytest.mark.parametrize("zdim", [False, True], ids=["2d", "3d"])
def test_render_ranges_and_camera_use_one_axis_convention(make_napari_viewer, zdim):
    """Asymmetric axes expose the old 2-D range and 3-D camera transpositions."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    ds = _asymmetric_channel(zdim=zdim)
    widget.get_dataset_from_test_mode([ds])

    itf = widget.data_to_layer_itf
    assert itf.render_range_x == pytest.approx([10_000, 12_000])
    assert itf.render_range_y == pytest.approx([40_000, 50_000])
    if zdim:
        assert itf.render_range_z == pytest.approx([-500, 500])

    widget.move_camera_center_to_render_range_center()
    # (z, y, x).  This asserted (z, x, y) and so agreed with a centre that had
    # x and y interchanged: napari places index 1 on y, as the check below
    # confirms against napari itself rather than against us.
    centre = widget.viewer.camera.center
    assert centre[0] == pytest.approx(0 if zdim else 1)
    assert centre[1] == pytest.approx(45_000)
    assert centre[2] == pytest.approx(11_000)


def test_napari_reads_a_camera_centre_as_z_y_x(make_napari_viewer):
    """The convention the test above depends on, taken from napari.

    Asked of napari directly, so that the assertion above is anchored to the
    library's behaviour rather than to a comment in this repository -- which
    is how it came to assert the transposition it was written to catch.
    """
    viewer = make_napari_viewer(ndisplay=3)
    viewer.add_points(np.array([[0.0, 0.0, 0.0], [10.0, 1_000.0, 100_000.0]]))
    viewer.reset_view()

    centre = viewer.camera.center
    assert centre[1] == pytest.approx(500.0)  # y
    assert centre[2] == pytest.approx(50_000.0)  # x


# ------------------------------------------------------------- named views
#
# The view buttons set Euler angles chosen when the planner emitted (z, x, y).
# The planner emits (z, y, x) now, so the same angles select a different pair
# of axes: every button showed a view other than the one on its label, and a
# dataset opened side-on.  Directions cannot drift that way -- these assert the
# axis the camera looks down, which is what the label is a claim about.


def _looks_along(camera):
    """The world axis the camera faces, as a signed name in (z, y, x)."""
    direction = np.asarray(camera.view_direction, dtype=float)
    axis = int(np.argmax(np.abs(direction)))
    return f"{'-' if direction[axis] < 0 else '+'}{'zyx'[axis]}"


def _points_up(camera):
    up = np.asarray(camera.up_direction, dtype=float)
    axis = int(np.argmax(np.abs(up)))
    return f"{'-' if up[axis] < 0 else '+'}{'zyx'[axis]}"


@pytest.mark.parametrize(
    "view, along, up",
    [("XY", "+z", "-y"), ("XZ", "-y", "-z"), ("YZ", "+x", "-z")],
)
def test_each_view_button_looks_down_the_axis_it_names(
    make_napari_viewer, view, along, up
):
    """A view of the XY plane is the one looking down z, and so on."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.viewer.dims.ndisplay = 3
    widget.get_dataset_from_test_mode([_asymmetric_channel(zdim=True)])

    widget.change_camera(set_view_to=view)

    assert _looks_along(widget.viewer.camera) == along
    assert _points_up(widget.viewer.camera) == up


def test_a_dataset_opens_looking_at_the_xy_plane(make_napari_viewer):
    """Not side-on: the image is what you see first, the sections on request."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.viewer.dims.ndisplay = 3
    widget.get_dataset_from_test_mode([_asymmetric_channel(zdim=True)])

    assert _looks_along(widget.viewer.camera) == "+z"
    assert _points_up(widget.viewer.camera) == "-y"


def test_the_named_views_are_three_distinct_directions(make_napari_viewer):
    """Two buttons showing the same view is the shape the old bug took."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.viewer.dims.ndisplay = 3
    widget.get_dataset_from_test_mode([_asymmetric_channel(zdim=True)])

    seen = set()
    for view in ("XY", "XZ", "YZ"):
        widget.change_camera(set_view_to=view)
        seen.add(_looks_along(widget.viewer.camera).lstrip("+-"))
    assert seen == {"x", "y", "z"}


def test_the_scale_bar_is_labelled_in_nanometres(make_napari_viewer):
    """Unlabelled, napari's scale bar could only call these coordinates px."""
    widget = napari_storm(napari_viewer=make_napari_viewer())

    assert widget.viewer.scale_bar.unit == "nm"


def test_labelling_the_scale_bar_does_not_move_the_data(make_napari_viewer):
    """The reason this is not `Layer.units`, which napari 0.8 will want.

    Setting units per layer rescales the world: `reset_view` then framed the
    two-spot fixture with most of it off-screen.  Whoever lifts the napari<0.8
    pin will have to move to Layer.units, and this is the check that says
    whether the move is safe.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_asymmetric_channel(zdim=True)])
    widget.viewer.reset_view()
    framed = (widget.viewer.camera.zoom, tuple(widget.viewer.camera.center))

    widget.viewer.scale_bar.unit = "nm"
    widget.viewer.reset_view()

    assert widget.viewer.camera.zoom == pytest.approx(framed[0])
    assert tuple(widget.viewer.camera.center) == pytest.approx(framed[1])


@pytest.mark.parametrize("zdim", [False, True], ids=["2d", "3d"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_range_filter_changes_only_the_selected_axis(make_napari_viewer, zdim, axis):
    """Range sliders must filter the named property in both dimensionalities."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    ds = _asymmetric_channel(zdim=zdim)
    widget.get_dataset_from_test_mode([ds])

    if axis == "x":
        widget.render_config.range_x_percent = np.array([0, 50])
    else:
        widget.render_config.range_y_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_data_range(ds)

    if axis == "x":
        assert ds.x_pos_nm.min() == pytest.approx(10_000)
        assert ds.x_pos_nm.max() == pytest.approx(11_000)
        assert ds.y_pos_nm.min() == pytest.approx(40_000)
        assert ds.y_pos_nm.max() == pytest.approx(50_000)
    else:
        assert ds.y_pos_nm.min() == pytest.approx(40_000)
        assert ds.y_pos_nm.max() == pytest.approx(45_000)
        assert ds.x_pos_nm.min() == pytest.approx(10_000)
        assert ds.x_pos_nm.max() == pytest.approx(12_000)


@pytest.mark.parametrize("zdim", [False, True], ids=["2d", "3d"])
def test_render_range_preview_uses_world_axis_extents(make_napari_viewer, zdim):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_asymmetric_channel(zdim=zdim)])

    coords, _faces = widget.Srender_rangex.get_coords_faces()
    # (z, y, x): the x extent is the last column, the y extent the middle one.
    assert np.unique(coords[:, 2]) == pytest.approx([10_000, 12_000])
    assert np.unique(coords[:, 1]) == pytest.approx([40_000, 50_000])
    expected_z = [-500, 500] if zdim else [1]
    assert np.unique(coords[:, 0]) == pytest.approx(expected_z)


def test_reference_image_frame_matches_localization_frame(make_napari_viewer):
    """An image translated to (z, x, y) nm must land on data at those nm."""
    from napari_storm.pyqt.image_import_dialog import ImageImportResult
    from napari_storm.pyqt.image_layer_controls import _expand_image

    widget = napari_storm(napari_viewer=make_napari_viewer())
    # Asymmetric on purpose: with a square field of view this test passes
    # whichever way round the two lateral axes are.
    ds = _asymmetric_channel()
    widget.get_dataset_from_test_mode([ds])

    result = ImageImportResult(
        file_path="ref.tif",
        img=np.zeros((20, 20), dtype=np.uint8),
        orientation="XY",
        px_xy_nm=100.0,
        px_z_nm=100.0,
        x_off_nm=10_000.0,
        y_off_nm=40_000.0,
        z_off_nm=0.0,
        layer_name="ref",
    )
    _data, _scale, translate = _expand_image(result)

    coords = widget.data_to_layer_itf.get_coords_from_locs(ds)
    # translate is (z, y, x); the image origin must coincide with the data
    # origin now that neither carries a hidden shift.
    assert translate[1] == pytest.approx(coords[:, 1].min(), abs=1)  # y
    assert translate[2] == pytest.approx(coords[:, 2].min(), abs=1)  # x
