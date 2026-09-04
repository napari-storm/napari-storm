"""Driving the exporter from the widget: axis order, scope, and the round trip.

The riskiest part is not the raster -- that has its own golden test -- but the
adapter. `RenderPlanner` returns coordinates as ``(z, x, y)`` and sigmas as
``(z, y, x)``, which is easy to miss and produces a file that looks entirely
plausible with its lateral widths transposed. Every fixture here is
deliberately asymmetric in all three axes so that a swap cannot pass.
"""

import numpy as np
import pytest
import tifffile

from napari_storm._dock_widget import napari_storm
from napari_storm.image_export import (
    SCOPE_EVERYTHING,
    ExportOptions,
    channel_for,
    plan_from_widget,
)
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


def _dataset(name="ds", zdim=False, n=200, anisotropic=False):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    if anisotropic:
        fields += [
            ("sigma_x_pixels", "f4"),
            ("sigma_y_pixels", "f4"),
            ("sigma_z_pixels", "f4"),
        ]
    locs = np.zeros(n, dtype=fields)
    # Distinct spans per axis, so a transposition changes the answer.
    locs["x_pos_nm"] = np.linspace(1_000, 3_000, n)
    locs["y_pos_nm"] = np.linspace(1_000, 7_000, n)
    if zdim:
        locs["z_pos_nm"] = np.linspace(-400, 400, n)
    if anisotropic:
        locs["sigma_x_pixels"] = 20.0
        locs["sigma_y_pixels"] = 60.0
        locs["sigma_z_pixels"] = 100.0
    dataset = LocalizationDataBaseClass(
        np.rec.array(locs), name=name, zdim_present=zdim
    )
    if anisotropic:
        dataset.sigma_present = True
    return dataset


def _widget(make_napari_viewer, **kwargs):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset(**kwargs)])
    return widget, viewer


# ------------------------------------------------------------- axis order


def test_coordinates_are_converted_to_z_y_x(make_napari_viewer):
    """The planner hands back (z, x, y); the exporter needs (z, y, x)."""
    widget, _ = _widget(make_napari_viewer, zdim=True)
    dataset = widget.localization_datasets[0]

    channel = channel_for(widget.data_to_layer_itf, dataset)

    # y spans 1000..7000 and x spans 1000..3000, so the wider column is y.
    y_span = np.ptp(channel.coords_nm[:, 1])
    x_span = np.ptp(channel.coords_nm[:, 2])
    assert y_span == pytest.approx(6000.0, rel=1e-3)
    assert x_span == pytest.approx(2000.0, rel=1e-3)


def test_sigmas_are_left_in_z_y_x_and_not_reordered(make_napari_viewer):
    """Reordering these as well would transpose the lateral widths."""
    widget, _ = _widget(make_napari_viewer, zdim=True, anisotropic=True)
    widget.render_config.gaussian_mode = 1
    dataset = widget.localization_datasets[0]

    channel = channel_for(widget.data_to_layer_itf, dataset)

    # sigma_y is three times sigma_x in the fixture; the ratio must survive.
    sigma_y = channel.sigmas_nm[0, 1]
    sigma_x = channel.sigmas_nm[0, 2]
    assert sigma_y == pytest.approx(3.0 * sigma_x, rel=1e-3)
    assert channel.sigmas_nm[0, 0] > sigma_y  # z is widest of the three


def test_sigmas_come_back_in_nanometres_not_normalized(make_napari_viewer):
    """The planner normalizes against the largest; a raster needs real units."""
    widget, _ = _widget(make_napari_viewer)
    widget.render_config.fixed_sigma_xy_nm = 25.0
    dataset = widget.localization_datasets[0]

    channel = channel_for(widget.data_to_layer_itf, dataset)

    assert channel.sigmas_nm[0, 1] == pytest.approx(25.0, rel=1e-3)
    assert channel.sigmas_nm[0, 2] == pytest.approx(25.0, rel=1e-3)


# ------------------------------------------------------------------ scope


def test_the_default_scope_is_the_current_view_in_2d(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer, zdim=True)

    plan = plan_from_widget(widget, ExportOptions(pixel_size_nm=50.0))

    assert plan.grid.shape[0] == 1  # one plane: a projection, not a slice
    assert plan.grid.is_2d


def test_the_current_view_follows_the_render_range(make_napari_viewer):
    """Shrinking the render range must shrink the exported extent."""
    widget, _ = _widget(make_napari_viewer)
    full = plan_from_widget(widget, ExportOptions(pixel_size_nm=20.0))

    widget.update_render_range("x", (40, 60))
    widget.update_render_range("y", (40, 60))
    narrowed = plan_from_widget(widget, ExportOptions(pixel_size_nm=20.0))

    assert narrowed.grid.shape[2] < full.grid.shape[2]
    assert narrowed.grid.shape[1] < full.grid.shape[1]


def test_a_3d_export_makes_a_stack_over_the_whole_extent(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer, zdim=True)

    plan = plan_from_widget(
        widget,
        ExportOptions(pixel_size_nm=50.0, scope=SCOPE_EVERYTHING, z_step_nm=100.0),
    )

    assert plan.grid.shape[0] > 1
    assert plan.grid.pixel_size_nm[0] == 100.0
    assert plan.grid.pixel_size_nm[1] == 50.0


def test_the_3d_extent_ignores_the_render_range(make_napari_viewer):
    """'Everything' means everything, whatever the sliders say."""
    widget, _ = _widget(make_napari_viewer, zdim=True)
    options = ExportOptions(pixel_size_nm=50.0, scope=SCOPE_EVERYTHING, z_step_nm=100.0)
    before = plan_from_widget(widget, options).grid.shape

    widget.update_render_range("x", (45, 55))
    widget.update_render_range("y", (45, 55))

    assert plan_from_widget(widget, options).grid.shape == before


def test_the_pixel_size_reaches_the_grid(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer)

    fine = plan_from_widget(widget, ExportOptions(pixel_size_nm=5.0))
    coarse = plan_from_widget(widget, ExportOptions(pixel_size_nm=20.0))

    assert fine.grid.pixel_size_nm[1] == 5.0
    assert fine.grid.shape[1] == 4 * coarse.grid.shape[1]


# ------------------------------------------------------------- the filters


def test_the_export_covers_filtered_rather_than_displayed_localizations(
    make_napari_viewer,
):
    """The whole reason the two masks are separate."""
    widget, _ = _widget(make_napari_viewer, n=1000)
    dataset = widget.localization_datasets[0]
    hidden = dataset.table.limit_active_to(100)
    assert hidden > 0

    channel = channel_for(widget.data_to_layer_itf, dataset)

    assert channel.n_localizations == 1000
    assert channel.n_displayed == 100
    assert channel.was_display_limited


def test_a_thinned_view_produces_the_warning_on_the_plan(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer, n=1000)
    widget.localization_datasets[0].table.limit_active_to(100)

    plan = plan_from_widget(widget, ExportOptions(pixel_size_nm=50.0))

    assert [w.kind for w in plan.warnings] == ["display_limited"]
    assert "1,000" in plan.warnings[0].message


# ----------------------------------------------------------- the round trip


def test_exporting_writes_a_readable_calibrated_file(make_napari_viewer, tmp_path):
    widget, _ = _widget(make_napari_viewer)
    path = tmp_path / "recon.ome.tif"

    widget.export_image(
        options=ExportOptions(pixel_size_nm=50.0),
        path=str(path),
        force_synchronous=True,
    )

    assert path.exists()
    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
        data = handle.asarray()
    assert 'PhysicalSizeX="50.0"' in xml
    assert 'PhysicalSizeXUnit="nm"' in xml
    assert data.size > 0
    assert np.any(data)  # something was actually drawn


def test_a_3d_export_round_trips_with_its_z_calibration(make_napari_viewer, tmp_path):
    widget, _ = _widget(make_napari_viewer, zdim=True)
    path = tmp_path / "stack.ome.tif"

    widget.export_image(
        options=ExportOptions(
            pixel_size_nm=100.0, scope=SCOPE_EVERYTHING, z_step_nm=200.0
        ),
        path=str(path),
        force_synchronous=True,
    )

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
        data = handle.asarray()
    assert 'PhysicalSizeZ="200.0"' in xml
    assert 'PhysicalSizeX="100.0"' in xml
    assert data.ndim >= 2


def test_two_datasets_become_two_channels(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset("a"), _dataset("b")])
    path = tmp_path / "two.ome.tif"

    widget.export_image(
        options=ExportOptions(pixel_size_nm=100.0),
        path=str(path),
        force_synchronous=True,
    )

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
        data = handle.asarray()
    assert 'Name="a"' in xml and 'Name="b"' in xml
    assert data.shape[0] == 2


def test_exporting_with_nothing_loaded_reports_rather_than_raising(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    reported = []
    widget._warn_user = reported.append

    result = widget.export_image(options=ExportOptions(), path="unused.ome.tif")

    assert result is None
    assert reported and "nothing loaded" in reported[0]


def test_the_exported_image_is_not_transposed_flipped_or_mirrored(
    make_napari_viewer, tmp_path
):
    """An orientation check that a span or ratio assertion cannot make.

    A transposed, mirrored or flipped image has exactly the same spans, the
    same intensities and the same localization count as a correct one. The
    fixture is a letter F because it has no symmetry at all: the vertical
    stroke marks minimum x, the longest bar marks maximum y, and the middle bar
    is shorter than the top one. Getting any axis backwards moves one of them.
    """
    rng = np.random.default_rng(0)
    strokes = [
        np.column_stack([np.full(300, 1000.0), np.linspace(1000, 5000, 300)]),
        np.column_stack([np.linspace(1000, 4000, 300), np.full(300, 5000.0)]),
        np.column_stack([np.linspace(1000, 3000, 300), np.full(300, 3400.0)]),
    ]
    xy = np.vstack(strokes) + rng.normal(0, 20, (900, 2))
    # Two faint corner points set the extent wider than the F, so no stroke sits
    # on the boundary. A 2-D export is bounded by the render range *exactly* --
    # it is a crop, and a crop cuts the Gaussians it crosses -- which would
    # otherwise dim the top bar and confuse what this test is measuring.
    xy = np.vstack([xy, [[500.0, 500.0], [4500.0, 5500.0]]])
    locs = np.zeros(len(xy), dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"], locs["y_pos_nm"] = xy[:, 0], xy[:, 1]

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode(
        [LocalizationDataBaseClass(np.rec.array(locs), name="F", zdim_present=False)]
    )
    widget.render_config.fixed_sigma_xy_nm = 40.0
    path = tmp_path / "f.ome.tif"

    widget.export_image(
        options=ExportOptions(pixel_size_nm=20.0),
        path=str(path),
        force_synchronous=True,
    )

    with tifffile.TiffFile(path) as handle:
        image = np.squeeze(handle.asarray()).astype(np.float64)

    rows, columns = image.shape
    # Taller than wide: y spans 5000 nm, x spans 4000 nm.
    assert rows > columns

    # Measured by *extent*, not by summed intensity: a bar's integrated
    # intensity is its localization count times the Gaussian volume, so the
    # long top bar and the short middle bar sum to exactly the same number and
    # the brightest row is decided by noise. How far a bar reaches is the thing
    # that actually differs.
    bright = image > 0.2 * image.max()
    row_width = bright.sum(axis=1)
    column_height = bright.sum(axis=0)

    # The vertical stroke is at minimum x, so the tallest column is near 0.
    # A mirror in x, or a transpose, moves it.
    assert int(np.argmax(column_height)) < 0.2 * columns
    # The top bar is the widest row and sits high up. A flip in y moves it to
    # the bottom.
    top_bar = int(np.argmax(row_width))
    assert top_bar > 0.8 * rows
    # The middle bar is the next widest, lower down, and genuinely shorter.
    middle_bar = int(np.argmax(row_width[: int(0.75 * rows)]))
    assert 0.4 * rows < middle_bar < 0.7 * rows
    assert row_width[middle_bar] < row_width[top_bar]


def test_the_exported_image_is_the_raster_of_the_same_model(
    make_napari_viewer, tmp_path
):
    """End to end: the file matches what the rasterizer would produce."""
    from napari_storm.core.raster import rasterize

    widget, _ = _widget(make_napari_viewer, n=20)
    options = ExportOptions(pixel_size_nm=100.0)
    plan = plan_from_widget(widget, options)
    path = tmp_path / "check.ome.tif"

    widget.export_image(options=options, path=str(path), force_synchronous=True)

    with tifffile.TiffFile(path) as handle:
        written = handle.asarray()
    channel = plan.channels[0]
    expected = rasterize(
        channel.coords_nm, channel.sigmas_nm, channel.values, plan.grid
    )
    np.testing.assert_allclose(np.reshape(written, expected.shape), expected, atol=1e-5)


# ---------------------------------------------- reference-image placement


def test_a_reference_image_is_placed_on_the_localization_plane(make_napari_viewer):
    """Flat data and its reference image must share one plane.

    napari's 2-D display shows a single slice, so an image one plane away from
    the localizations is simply not drawn.
    """
    widget, _ = _widget(make_napari_viewer)

    from napari_storm.core.render_planner import FLAT_DATA_Z_NM

    assert widget.data_to_layer_itf.reference_plane_z_nm() == FLAT_DATA_Z_NM


def test_a_reference_image_is_centred_in_a_3d_stack(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer, zdim=True)
    itf = widget.data_to_layer_itf

    low, high = itf.render_range_z
    assert itf.reference_plane_z_nm() == pytest.approx(0.5 * (low + high))


def test_with_nothing_loaded_it_falls_back_to_the_flat_plane(make_napari_viewer):
    """So an image imported first still meets a 2-D dataset imported second."""
    from napari_storm.core.render_planner import FLAT_DATA_Z_NM

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    assert widget.data_to_layer_itf.reference_plane_z_nm() == FLAT_DATA_Z_NM


def test_loading_a_dataset_does_not_move_an_existing_image(make_napari_viewer):
    """The §3.5 invariant, which is why re-centring is a button and not a rule.

    Placement is read once, at import. If loading a 3-D dataset silently
    re-centred every reference image, an alignment the user had made by hand
    would move under them -- the exact defect the auto-offset removal closed,
    and a §7.4 acceptance gate.
    """
    widget, viewer = _widget(make_napari_viewer)
    image = viewer.add_image(
        np.zeros((1, 8, 8), dtype="f4"),
        translate=(widget.data_to_layer_itf.reference_plane_z_nm(), 0, 0),
    )
    placed_at = tuple(image.translate)

    widget.get_dataset_from_test_mode([_dataset("later", zdim=True)])

    assert tuple(image.translate) == placed_at


# ------------------------------------------------- export/import round trip


def test_an_exported_file_reimports_at_its_original_scale_and_position(
    make_napari_viewer, tmp_path
):
    """§7.4: "the sharpest available test of both".

    Export and import are the same coordinate contract in opposite directions,
    so a round trip through this plugin's own writer and reader checks each
    against the other rather than against itself.
    """
    from napari_storm.pyqt.image_import_dialog import (
        _try_read_ome_position_nm,
        _try_read_tiff_pixel_size,
    )

    widget, _ = _widget(make_napari_viewer)
    path = tmp_path / "roundtrip.ome.tif"
    options = ExportOptions(pixel_size_nm=25.0)
    plan = plan_from_widget(widget, options)

    widget.export_image(options=options, path=str(path), force_synchronous=True)

    px_xy, _px_z = _try_read_tiff_pixel_size(str(path))
    x_nm, y_nm, z_nm = _try_read_ome_position_nm(str(path))

    assert px_xy == pytest.approx(25.0)
    # World axes, so these compare against the grid origin directly: no
    # transposition question arises between "OME PositionX" and "world x".
    assert x_nm == pytest.approx(plan.grid.origin_nm[2])
    assert y_nm == pytest.approx(plan.grid.origin_nm[1])
    assert z_nm == pytest.approx(plan.grid.origin_nm[0])


def test_the_reimported_position_is_where_the_localizations_are(
    make_napari_viewer, tmp_path
):
    """Not merely self-consistent: it has to land on the data it came from."""
    widget, _ = _widget(make_napari_viewer)
    itf = widget.data_to_layer_itf
    path = tmp_path / "roundtrip.ome.tif"

    widget.export_image(
        options=ExportOptions(pixel_size_nm=25.0),
        path=str(path),
        force_synchronous=True,
    )

    from napari_storm.pyqt.image_import_dialog import _try_read_ome_position_nm

    x_nm, y_nm, _z = _try_read_ome_position_nm(str(path))

    # Within one pixel of the render range the export was cut from.
    assert x_nm == pytest.approx(itf.render_range_x[0], abs=25.0)
    assert y_nm == pytest.approx(itf.render_range_y[0], abs=25.0)


def test_a_file_without_positions_imports_without_them(tmp_path):
    """Placement metadata is a convenience, never a requirement."""
    import tifffile

    from napari_storm.pyqt.image_import_dialog import _try_read_ome_position_nm

    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((8, 8), dtype=np.uint16))

    assert _try_read_ome_position_nm(str(plain)) is None


# ------------------------------------------------------- image orientation


def test_an_imported_image_is_not_transposed(make_napari_viewer):
    """Reference images arrive in the axis order the file stored them in.

    This used to swap the two lateral axes, to meet localizations that were
    themselves drawn transposed against napari. Both ends are fixed now: the
    planner emits ``(z, y, x)`` and an image file already stores rows = y, so
    the correct thing to do with the file's axes is nothing at all.
    """
    from napari_storm.pyqt.image_import_dialog import ImageImportResult
    from napari_storm.pyqt.image_layer_controls import _expand_image

    # Deliberately not square: 10 rows of y by 40 columns of x.
    image = np.zeros((10, 40), dtype=np.uint8)
    result = ImageImportResult(
        file_path="ref.tif",
        img=image,
        orientation="XY",
        px_xy_nm=100.0,
        px_z_nm=100.0,
        x_off_nm=0.0,
        y_off_nm=0.0,
        z_off_nm=0.0,
        layer_name="ref",
    )

    data, _scale, _translate = _expand_image(result)

    # dim1 is y, which the file stored as its 10 rows; dim2 is x, its 40 columns.
    assert data.shape == (1, 10, 40)


def test_a_3d_reference_stack_keeps_the_axis_order_it_was_stored_in(
    make_napari_viewer,
):
    from napari_storm.pyqt.image_import_dialog import ImageImportResult
    from napari_storm.pyqt.image_layer_controls import _expand_image

    result = ImageImportResult(
        file_path="ref.tif",
        img=np.zeros((7, 10, 40), dtype=np.uint8),  # (z, y, x)
        orientation="3D",
        px_xy_nm=100.0,
        px_z_nm=200.0,
        x_off_nm=0.0,
        y_off_nm=0.0,
        z_off_nm=0.0,
        layer_name="ref",
    )

    data, _scale, _translate = _expand_image(result)

    assert data.shape == (7, 10, 40)  # (z, y, x) in, (z, y, x) out


def test_an_rgb_reference_keeps_its_colour_axis(make_napari_viewer):
    """The colour axis stays last and is never treated as spatial."""
    from napari_storm.pyqt.image_import_dialog import ImageImportResult
    from napari_storm.pyqt.image_layer_controls import _expand_image

    result = ImageImportResult(
        file_path="ref.tif",
        img=np.zeros((10, 40, 3), dtype=np.uint8),
        orientation="XY",
        px_xy_nm=100.0,
        px_z_nm=100.0,
        x_off_nm=0.0,
        y_off_nm=0.0,
        z_off_nm=0.0,
        layer_name="ref",
    )

    data, _scale, _translate = _expand_image(result)

    assert data.shape == (1, 10, 40, 3)


def test_an_exported_image_reimports_the_right_way_round(make_napari_viewer, tmp_path):
    """The round trip that showed the bug: export, re-import, compare extents."""
    from napari_storm.pyqt.image_import_dialog import ImageImportResult
    from napari_storm.pyqt.image_layer_controls import _expand_image

    widget, _ = _widget(make_napari_viewer)  # x spans 2000 nm, y spans 6000 nm
    path = tmp_path / "roundtrip.ome.tif"
    options = ExportOptions(pixel_size_nm=50.0)
    plan = plan_from_widget(widget, options)

    widget.export_image(options=options, path=str(path), force_synchronous=True)

    with tifffile.TiffFile(path) as handle:
        written = np.squeeze(handle.asarray())
    result = ImageImportResult(
        file_path=str(path),
        img=written,
        orientation="XY",
        px_xy_nm=50.0,
        px_z_nm=50.0,
        x_off_nm=plan.grid.origin_nm[2],
        y_off_nm=plan.grid.origin_nm[1],
        z_off_nm=plan.grid.origin_nm[0],
        layer_name="ref",
    )

    data, scale, _translate = _expand_image(result)

    # The reconstruction is taller in y than wide in x, and the re-imported
    # layer must agree: dim1 is y, dim2 is x.
    y_extent = data.shape[1] * scale[1]
    x_extent = data.shape[2] * scale[2]
    assert y_extent > x_extent
