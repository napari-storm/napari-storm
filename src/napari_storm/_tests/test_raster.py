"""The exported raster, pinned against arithmetic rather than against a screenshot.

§4.1: "A golden test should pin the rasterizer against an analytically computed
Gaussian sum, not against a screen capture." That is the point of the exporter
existing at all -- it is the reference, and the canvas is the approximation, so
checking it against the canvas would have the dependency backwards.
"""
import numpy as np
import pytest

from napari_storm.core.raster import (SPLAT_SIGMAS, GaussianGrid,
                                      rasterize, rasterize_tiles)


def _analytic(coords_nm, sigmas_nm, values, grid):
    """The same image, written the obvious slow way, with no tiling or culling."""
    nz, ny, nx = grid.shape
    zs = grid.axis_coordinates_nm(0, 0, nz)
    ys = grid.axis_coordinates_nm(1, 0, ny)
    xs = grid.axis_coordinates_nm(2, 0, nx)
    out = np.zeros(grid.shape, dtype=np.float64)
    for k, z in enumerate(zs):
        for centre, sigma, amplitude in zip(coords_nm, sigmas_nm, values):
            if grid.is_2d:
                weight = amplitude
            else:
                if abs(centre[0] - z) > SPLAT_SIGMAS * sigma[0]:
                    continue
                weight = amplitude * np.exp(-0.5 * ((centre[0] - z) / sigma[0]) ** 2)
            dy = (ys - centre[1]) / sigma[1]
            dx = (xs - centre[2]) / sigma[2]
            patch = weight * np.outer(np.exp(-0.5 * dy**2), np.exp(-0.5 * dx**2))
            # The rasterizer truncates at five sigma; the reference must too, or
            # the comparison measures the truncation instead of the arithmetic.
            patch[np.abs(dy) > SPLAT_SIGMAS, :] = 0.0
            patch[:, np.abs(dx) > SPLAT_SIGMAS] = 0.0
            out[k] += patch
    return out


def _one_splat(sigma_nm=50.0, value=1.0):
    coords = np.array([[0.0, 500.0, 500.0]])
    sigmas = np.full((1, 3), sigma_nm)
    return coords, sigmas, np.array([value])


def _grid(pixel_size_nm=10.0, span_nm=1000.0):
    return GaussianGrid.covering(
        ((0.0, 0.0), (0.0, span_nm), (0.0, span_nm)), pixel_size_nm
    )


# ----------------------------------------------------------------- the golden


def test_a_single_splat_matches_the_closed_form_gaussian():
    coords, sigmas, values = _one_splat()
    grid = _grid()

    image = rasterize(coords, sigmas, values, grid)

    np.testing.assert_allclose(image, _analytic(coords, sigmas, values, grid), atol=1e-6)


def test_the_peak_sits_at_the_localization_and_has_the_right_amplitude():
    """Not merely 'something bright appeared'."""
    coords, sigmas, values = _one_splat(sigma_nm=50.0, value=3.0)
    grid = _grid(pixel_size_nm=10.0)

    image = rasterize(coords, sigmas, values, grid)[0]
    peak = np.unravel_index(np.argmax(image), image.shape)

    # Pixel centres are at 5, 15, ... so 500 nm falls on the boundary between
    # index 49 and 50; either is the correct nearest sample.
    assert peak[0] in (49, 50) and peak[1] in (49, 50)
    assert image.max() == pytest.approx(3.0, rel=0.02)


def test_overlapping_splats_add():
    """Additive, like the renderer: two coincident splats are twice as bright."""
    grid = _grid()
    coords = np.array([[0.0, 500.0, 500.0], [0.0, 500.0, 500.0]])
    sigmas = np.full((2, 3), 50.0)

    image = rasterize(coords, sigmas, np.array([1.0, 1.0]), grid)

    assert image.max() == pytest.approx(2.0, rel=0.02)


def test_a_gaussian_falls_off_like_a_gaussian():
    """A disc would pass a peak check and fails this one."""
    sigma_nm = 100.0
    coords, sigmas, values = _one_splat(sigma_nm=sigma_nm)
    grid = _grid(pixel_size_nm=5.0, span_nm=1000.0)

    image = rasterize(coords, sigmas, values, grid)[0]
    ys = grid.axis_coordinates_nm(1, 0, grid.shape[1])
    xs = grid.axis_coordinates_nm(2, 0, grid.shape[2])
    row = int(np.argmin(np.abs(ys - coords[0, 1])))

    # Compared against each pixel's *own* distance from the localization. The
    # peak pixel is half a pixel off centre, so measuring offsets from it
    # instead would compare a Gaussian against a shifted copy of itself.
    for offset_sigmas in (0.5, 1.0, 2.0, 4.0):
        column = int(np.argmin(np.abs(xs - (coords[0, 2] + offset_sigmas * sigma_nm))))
        distance = (xs[column] - coords[0, 2]) / sigma_nm
        vertical = (ys[row] - coords[0, 1]) / sigma_nm
        expected = np.exp(-0.5 * (distance**2 + vertical**2))
        assert image[row, column] == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------------- tiling


@pytest.mark.parametrize("tile_pixels", [7, 16, 64, 4096])
def test_the_tile_size_never_changes_the_result(tile_pixels):
    """Peak memory is a tuning knob; the image is not.

    A splat straddling a tile boundary is the case that breaks a naive
    implementation, so the grid is deliberately not a multiple of the tile.
    """
    rng = np.random.default_rng(0)
    coords = np.zeros((40, 3))
    coords[:, 1] = rng.uniform(0, 1000, 40)
    coords[:, 2] = rng.uniform(0, 1000, 40)
    sigmas = np.full((40, 3), 40.0)
    values = rng.uniform(0.5, 2.0, 40)
    grid = _grid(pixel_size_nm=10.0, span_nm=1000.0)

    reference = rasterize(coords, sigmas, values, grid, tile_pixels=4096)
    tiled = rasterize(coords, sigmas, values, grid, tile_pixels=tile_pixels)

    np.testing.assert_allclose(tiled, reference, atol=1e-5)


def test_tiles_cover_the_grid_exactly_once():
    grid = GaussianGrid.covering(((0.0, 0.0), (0.0, 700.0), (0.0, 500.0)), 10.0)
    coords, sigmas, values = _one_splat()

    covered = np.zeros(grid.shape, dtype=int)
    for z, ys, xs, tile in rasterize_tiles(coords, sigmas, values, grid, 16):
        covered[z, ys, xs] += 1
        assert tile.shape == (ys.stop - ys.start, xs.stop - xs.start)

    assert np.all(covered == 1)


def test_peak_memory_is_set_by_the_tile_not_the_image():
    """The claim that makes 'never downsample' affordable."""
    grid = GaussianGrid.covering(((0.0, 0.0), (0.0, 20_000.0), (0.0, 20_000.0)), 10.0)
    coords, sigmas, values = _one_splat()

    assert grid.shape == (1, 2000, 2000)
    biggest = max(
        tile.nbytes
        for _z, _ys, _xs, tile in rasterize_tiles(coords, sigmas, values, grid, 256)
    )
    assert biggest <= 256 * 256 * 4
    assert biggest < grid.nbytes() / 50


# --------------------------------------------------------------- the grid


def test_the_requested_pixel_size_is_honoured_exactly():
    """Never downsample: the grid grows, the sampling does not coarsen."""
    for pixel_size in (1.0, 2.5, 10.0, 37.0):
        grid = GaussianGrid.covering(
            ((0.0, 0.0), (0.0, 10_000.0), (0.0, 10_000.0)), pixel_size
        )
        assert grid.pixel_size_nm[1] == pixel_size
        assert grid.pixel_size_nm[2] == pixel_size
        assert grid.shape[1] == int(np.ceil(10_000.0 / pixel_size))


def test_a_finer_pixel_size_produces_a_bigger_image_not_a_coarser_one():
    coarse = GaussianGrid.covering(((0.0, 0.0), (0.0, 1000.0), (0.0, 1000.0)), 10.0)
    fine = GaussianGrid.covering(((0.0, 0.0), (0.0, 1000.0), (0.0, 1000.0)), 1.0)

    assert fine.shape[1] == 10 * coarse.shape[1]
    assert fine.pixel_size_nm[1] < coarse.pixel_size_nm[1]


def test_the_grid_covers_the_whole_requested_extent():
    """A truncated export would silently lose data at the far edge."""
    grid = GaussianGrid.covering(((0.0, 0.0), (0.0, 1005.0), (0.0, 1005.0)), 10.0)

    assert grid.shape[1] * grid.pixel_size_nm[1] >= 1005.0
    assert grid.axis_coordinates_nm(1, 0, grid.shape[1])[-1] >= 1005.0 - 10.0


def test_z_sampling_is_independent_of_xy_sampling():
    """SMLM axial and lateral resolution differ; one knob would be wrong."""
    grid = GaussianGrid.covering(
        ((-500.0, 500.0), (0.0, 1000.0), (0.0, 1000.0)),
        pixel_size_nm=10.0,
        z_step_nm=50.0,
    )

    assert grid.pixel_size_nm == (50.0, 10.0, 10.0)
    assert grid.shape == (20, 100, 100)


def test_a_2d_grid_projects_rather_than_slicing():
    """Every localization contributes fully, whatever its z."""
    grid = _grid()
    coords = np.array([[0.0, 500.0, 500.0], [900.0, 500.0, 500.0]])
    sigmas = np.full((2, 3), 50.0)

    image = rasterize(coords, sigmas, np.array([1.0, 1.0]), grid)

    assert grid.shape[0] == 1
    # Both splats land on the same spot, so the projection sums them; a slice
    # through z = 0 would show only one.
    assert image.max() == pytest.approx(2.0, rel=0.02)


def test_a_3d_grid_applies_axial_falloff():
    grid = GaussianGrid.covering(
        ((-200.0, 200.0), (0.0, 1000.0), (0.0, 1000.0)),
        pixel_size_nm=10.0,
        z_step_nm=100.0,
    )
    coords = np.array([[0.0, 500.0, 500.0]])
    sigmas = np.full((1, 3), 100.0)

    image = rasterize(coords, sigmas, np.array([1.0]), grid)

    peaks = image.max(axis=(1, 2))
    assert peaks.argmax() in (1, 2)  # brightest slice is nearest z = 0
    assert peaks[0] < peaks.max()  # and it falls off away from it


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_nonsense_pixel_size_is_refused(bad):
    with pytest.raises(ValueError, match="pixel size"):
        GaussianGrid.covering(((0.0, 0.0), (0.0, 100.0), (0.0, 100.0)), bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_nonsense_z_step_is_refused(bad):
    with pytest.raises(ValueError, match="z step"):
        GaussianGrid.covering(
            ((0.0, 100.0), (0.0, 100.0), (0.0, 100.0)), 10.0, z_step_nm=bad
        )


def test_mismatched_inputs_are_refused_before_any_work():
    grid = _grid()
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        rasterize(np.zeros((4, 2)), np.ones((4, 2)), np.ones(4), grid)
    with pytest.raises(ValueError, match="same shape"):
        rasterize(np.zeros((4, 3)), np.ones((3, 3)), np.ones(4), grid)
    with pytest.raises(ValueError, match=r"\(N,\)"):
        rasterize(np.zeros((4, 3)), np.ones((4, 3)), np.ones(3), grid)
