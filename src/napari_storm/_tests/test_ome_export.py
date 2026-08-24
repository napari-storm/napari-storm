"""The exported file, checked by reading it back.

§4.1 asks for three things this pins down: the pixel size is written *with its
unit*, the export never downsamples to fit, and it contains the localizations
the user's **filters** left active rather than the subsample the render budget
imposed on the screen.

Round-trips go through tifffile's reader rather than through our own writer's
idea of what it wrote, so a metadata key that no reader honours fails here.
"""
import numpy as np
import pytest
import tifffile

from napari_storm.core.ome_export import (ExportChannel, ExportWarning,
                                          plan_export, tile_count,
                                          write_ome_tiff)
from napari_storm.core.raster import GaussianGrid


def _channel(name="ch", n=4, span=1000.0, n_displayed=None, colormap=None):
    rng = np.random.default_rng(0)
    coords = np.zeros((n, 3))
    coords[:, 1] = rng.uniform(200, span - 200, n)
    coords[:, 2] = rng.uniform(200, span - 200, n)
    return ExportChannel(
        name=name,
        coords_nm=coords,
        sigmas_nm=np.full((n, 3), 40.0),
        values=np.ones(n),
        colormap=colormap,
        n_displayed=n_displayed,
    )


def _bounds(span=1000.0):
    return ((0.0, 0.0), (0.0, span), (0.0, span))


# ------------------------------------------------------------- calibration


def test_the_pixel_size_and_its_unit_survive_a_round_trip(tmp_path):
    """A pixel size with no declared unit is the defect this feature avoids."""
    plan = plan_export([_channel()], _bounds(), pixel_size_nm=5.0)
    path = tmp_path / "out.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
    assert 'PhysicalSizeX="5.0"' in xml
    assert 'PhysicalSizeXUnit="nm"' in xml
    assert 'PhysicalSizeY="5.0"' in xml
    assert 'PhysicalSizeYUnit="nm"' in xml
    assert 'PhysicalSizeZUnit="nm"' in xml


def test_an_odd_pixel_size_is_written_exactly_as_asked(tmp_path):
    """Not rounded to something convenient for the grid."""
    plan = plan_export([_channel()], _bounds(), pixel_size_nm=3.7)
    path = tmp_path / "out.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        assert 'PhysicalSizeX="3.7"' in handle.ome_metadata


def test_channel_names_and_colormaps_travel_as_metadata(tmp_path):
    """Colour must not be baked into the samples, or the file stops being data."""
    channels = [
        _channel("membrane", colormap="green"),
        _channel("nucleus", colormap="magenta"),
    ]
    plan = plan_export(channels, _bounds(), pixel_size_nm=10.0)
    path = tmp_path / "out.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
        data = handle.asarray()
    assert 'Name="membrane"' in xml
    assert 'Name="nucleus"' in xml
    # Two channels of single-sample data, not one RGB image.
    assert data.shape[0] == 2
    assert data.dtype == np.float32


# ------------------------------------------------------------ never downsample


def test_a_finer_pixel_size_writes_a_bigger_file_not_a_coarser_one(tmp_path):
    coarse = plan_export([_channel()], _bounds(), pixel_size_nm=10.0)
    fine = plan_export([_channel()], _bounds(), pixel_size_nm=2.0)

    write_ome_tiff(tmp_path / "coarse.ome.tif", coarse)
    write_ome_tiff(tmp_path / "fine.ome.tif", fine)

    with tifffile.TiffFile(tmp_path / "coarse.ome.tif") as handle:
        coarse_shape = handle.asarray().shape
    with tifffile.TiffFile(tmp_path / "fine.ome.tif") as handle:
        fine_shape = handle.asarray().shape

    assert fine_shape[-1] == 5 * coarse_shape[-1]
    assert fine_shape[-2] == 5 * coarse_shape[-2]


def test_the_plan_states_the_size_before_anything_is_written():
    """A 40 GB export should be refusable in advance, not discovered."""
    plan = plan_export([_channel(), _channel("b")], _bounds(20_000.0), 1.0)

    assert plan.shape == (2, 1, 20_000, 20_000)
    assert plan.nbytes == 2 * 20_000 * 20_000 * 4
    assert plan.nbytes > 3e9  # the point: knowable up front


def test_tiles_are_streamed_rather_than_the_image_being_built(tmp_path):
    """Peak memory must not scale with the image.

    A 4000 x 4000 plane is 64 MB as one array; written in 1024-pixel tiles the
    largest live array is 4 MB.
    """
    plan = plan_export([_channel(span=8000.0)], _bounds(8000.0), pixel_size_nm=2.0)
    assert plan.grid.shape == (1, 4000, 4000)

    counted = tile_count(plan)
    assert counted == 4 * 4  # ceil(4000 / 1024) squared

    import tracemalloc

    tracemalloc.start()
    tracemalloc.reset_peak()
    write_ome_tiff(tmp_path / "big.ome.tif", plan)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    whole_image = plan.grid.nbytes()
    assert peak < whole_image / 4, (peak, whole_image)


# ------------------------------------------------------- the display warning


def test_a_thinned_view_warns_with_both_counts():
    """§4.1: say what is displayed, what is written, and why they differ."""
    plan = plan_export(
        [_channel("ch", n=1000, n_displayed=250)], _bounds(), pixel_size_nm=10.0
    )

    assert [w.kind for w in plan.warnings] == ["display_limited"]
    message = plan.warnings[0].message
    assert "250" in message and "1,000" in message
    assert "render budget" in message


def test_an_unthinned_view_does_not_warn():
    plan = plan_export(
        [_channel("ch", n=1000, n_displayed=1000)], _bounds(), pixel_size_nm=10.0
    )
    assert plan.warnings == ()


def test_the_warning_is_data_not_a_side_effect():
    """So a dialog, a log and a test can each do what they need with it."""
    plan = plan_export([_channel(n=10, n_displayed=2)], _bounds(), 10.0)
    assert isinstance(plan.warnings[0], ExportWarning)
    assert plan.warnings[0].kind == "display_limited"


def test_an_empty_channel_is_reported_rather_than_silently_blank():
    channel = ExportChannel(
        name="filtered-out",
        coords_nm=np.zeros((0, 3)),
        sigmas_nm=np.zeros((0, 3)),
        values=np.zeros(0),
    )
    plan = plan_export([channel], _bounds(), pixel_size_nm=10.0)

    assert [w.kind for w in plan.warnings] == ["empty_channel"]


def test_an_empty_channel_still_writes_a_readable_plane(tmp_path):
    channel = ExportChannel(
        name="none",
        coords_nm=np.zeros((0, 3)),
        sigmas_nm=np.zeros((0, 3)),
        values=np.zeros(0),
    )
    plan = plan_export([channel], _bounds(), pixel_size_nm=10.0)

    write_ome_tiff(tmp_path / "empty.ome.tif", plan)

    with tifffile.TiffFile(tmp_path / "empty.ome.tif") as handle:
        data = handle.asarray()
    assert data.shape[-2:] == (100, 100)
    assert not np.any(data)


# ------------------------------------------------------------------ content


def test_the_written_pixels_are_the_rasterized_gaussians(tmp_path):
    """The file is the raster, not a rescaled or clipped version of it."""
    from napari_storm.core.raster import rasterize

    channel = _channel(n=3)
    plan = plan_export([channel], _bounds(), pixel_size_nm=10.0)
    path = tmp_path / "out.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        written = handle.asarray()
    expected = rasterize(
        channel.coords_nm, channel.sigmas_nm, channel.values, plan.grid
    )
    np.testing.assert_allclose(np.reshape(written, expected.shape), expected, atol=1e-6)


def test_a_3d_export_writes_one_plane_per_z_step(tmp_path):
    coords = np.array([[-100.0, 500.0, 500.0], [100.0, 500.0, 500.0]])
    channel = ExportChannel(
        name="stack",
        coords_nm=coords,
        sigmas_nm=np.full((2, 3), 50.0),
        values=np.ones(2),
    )
    plan = plan_export(
        [channel],
        ((-200.0, 200.0), (0.0, 1000.0), (0.0, 1000.0)),
        pixel_size_nm=10.0,
        z_step_nm=50.0,
    )

    write_ome_tiff(tmp_path / "stack.ome.tif", plan)

    with tifffile.TiffFile(tmp_path / "stack.ome.tif") as handle:
        data = handle.asarray()
    assert plan.grid.shape[0] == 8
    assert data.shape[-3] == 8
    # The two localizations sit at different depths, so different planes peak.
    peaks = np.reshape(data, plan.grid.shape).max(axis=(1, 2))
    assert peaks.argmax() != 0


def test_z_step_is_written_separately_from_the_xy_pixel_size(tmp_path):
    channel = _channel(n=2)
    plan = plan_export(
        [channel],
        ((-200.0, 200.0), (0.0, 1000.0), (0.0, 1000.0)),
        pixel_size_nm=10.0,
        z_step_nm=75.0,
    )

    write_ome_tiff(tmp_path / "stack.ome.tif", plan)

    with tifffile.TiffFile(tmp_path / "stack.ome.tif") as handle:
        xml = handle.ome_metadata
    assert 'PhysicalSizeZ="75.0"' in xml
    assert 'PhysicalSizeX="10.0"' in xml


# ---------------------------------------------------------------- lifecycle


def test_progress_is_reported_per_tile(tmp_path):
    plan = plan_export([_channel(span=4000.0)], _bounds(4000.0), pixel_size_nm=2.0)
    seen = []

    write_ome_tiff(tmp_path / "out.ome.tif", plan, progress=lambda d, t: seen.append((d, t)))

    assert seen
    assert seen[-1] == (tile_count(plan), tile_count(plan))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_a_cancelled_export_leaves_no_file_behind(tmp_path):
    """A truncated image that looks finished is worse than no image."""
    plan = plan_export([_channel(span=4000.0)], _bounds(4000.0), pixel_size_nm=2.0)
    path = tmp_path / "cancelled.ome.tif"
    calls = []

    def should_cancel():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(InterruptedError):
        write_ome_tiff(path, plan, should_cancel=should_cancel)

    assert not path.exists()


def test_an_export_with_no_channels_is_refused():
    with pytest.raises(ValueError, match="at least one channel"):
        plan_export([], _bounds(), pixel_size_nm=10.0)


# ------------------------------------------------------------ world position


def test_the_world_position_is_written_with_its_unit(tmp_path):
    """Pixel size says how big a pixel is, not where the image is.

    Without a position, re-importing an export lands it at the origin instead
    of at the render range it was cut from -- and §7.4 asks that an exported
    OME-TIFF re-import "at its original scale and position".
    """
    plan = plan_export([_channel()], ((0.0, 0.0), (500.0, 1500.0), (200.0, 1200.0)), 10.0)
    path = tmp_path / "placed.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
    # Pixel centres, so half a pixel in from the requested bounds.
    assert 'PositionX="205.0"' in xml
    assert 'PositionY="505.0"' in xml
    assert 'PositionXUnit="nm"' in xml
    assert 'PositionZUnit="nm"' in xml


def test_the_position_matches_the_grid_origin_exactly(tmp_path):
    plan = plan_export([_channel()], ((0.0, 0.0), (500.0, 1500.0), (200.0, 1200.0)), 10.0)
    path = tmp_path / "placed.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
    assert f'PositionX="{plan.grid.origin_nm[2]}"' in xml
    assert f'PositionY="{plan.grid.origin_nm[1]}"' in xml


def test_each_z_slice_carries_its_own_depth(tmp_path):
    """A stack whose planes all claim one depth would re-import flat."""
    plan = plan_export(
        [_channel()],
        ((-200.0, 200.0), (0.0, 1000.0), (0.0, 1000.0)),
        pixel_size_nm=10.0,
        z_step_nm=100.0,
    )
    path = tmp_path / "stack.ome.tif"

    write_ome_tiff(path, plan)

    with tifffile.TiffFile(path) as handle:
        xml = handle.ome_metadata
    import re

    depths = sorted({float(m) for m in re.findall(r'PositionZ="([-\d.]+)"', xml)})
    assert len(depths) == plan.grid.shape[0]
    assert depths[1] - depths[0] == pytest.approx(100.0)
