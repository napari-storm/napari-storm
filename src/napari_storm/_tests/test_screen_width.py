"""The Gaussian on screen has the width the settings ask for.

Every earlier pixel test was blind to the defect this module guards against.
Until 3.1 the shaders drew a Gaussian of 1.25 sigma from a covariance built of
the *square roots* of the sigmas, then wrote the falloff to both colour and
alpha, which additive blending multiplied together: an isotropic splat came out
0.88 sigma wide, a 4:1 anisotropy as roughly 2.8:1, and the billboard backend
drew an anisotropic one as a chevron, because its texture coordinates reached
the fragment shader mirrored on one triangle of each quad.  A Gaussian of the
wrong width is still a symmetric Gaussian, so tests of symmetry and position
passed throughout.

Here the width is *measured*: the second moment of each spot on an offscreen
render, against the same estimator applied to a reference image of the model
-- a sum of Gaussians evaluated at pixel centres, cut where the canvas cuts it
-- drawn at the screen's own scale.  The scale is not assumed either: the two
spots are a known distance apart and the pixel size is read off the render.

The reference is plain NumPy so this runs wherever the GL tests run.  The same
model is held against CrossCorrelate's voxel renderer, and the export raster
against both, in ``test_voxel_baseline.py``.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core.raster import GaussianGrid, rasterize
from napari_storm.core.render_planner import SIGMA_TO_SIZE_FACTOR
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles._napari_compat import instancing_available
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer
from napari_storm.ns_constants import FWHM_TO_SIGMA

#: Where the canvas cuts a Gaussian, in sigmas: half the billboard edge.  The
#: reference is drawn with the same cut so the two are the same model and a
#: second moment reads the same on both.  (The export raster cuts at
#: ``raster.SPLAT_SIGMAS`` instead; that difference is documented there.)
CANVAS_CUT_SIGMAS = SIGMA_TO_SIZE_FACTOR / 2

SPOT_SEPARATION_NM = 6_000.0

#: Half-width, in sigmas, of the window a spot is measured over.  Four sigma
#: holds 99.99% of a Gaussian's mass and far more of its second moment.
WINDOW_SIGMAS = 4.0

BACKENDS = [NapariParticlesRenderer, InstancedRenderer]


def reference_image(
    shape, nm_per_px, origin_nm, centres_nm, sigma_rows_nm, sigma_cols_nm
):
    """The model at pixel centres: unit-peak Gaussians, cut on a square.

    *origin_nm* is the ``(row, column)`` coordinate of pixel ``(0, 0)``'s
    centre and *centres_nm* the ``(row, column)`` of each Gaussian, both in
    nanometres along the image axes.
    """
    rows = origin_nm[0] + nm_per_px * np.arange(shape[0])[:, None]
    cols = origin_nm[1] + nm_per_px * np.arange(shape[1])[None, :]
    image = np.zeros(shape)
    for row_nm, col_nm in centres_nm:
        dr = (rows - row_nm) / sigma_rows_nm
        dc = (cols - col_nm) / sigma_cols_nm
        inside = (np.abs(dr) <= CANVAS_CUT_SIGMAS) & (np.abs(dc) <= CANVAS_CUT_SIGMAS)
        image += np.where(inside, np.exp(-0.5 * (dr**2 + dc**2)), 0.0)
    return image


def _two_spots(sigma_x_nm=None, sigma_y_nm=None, sigma_z_nm=None, zdim=False):
    """Two localizations a known distance apart along x.

    With sigmas given, the dataset carries per-localization uncertainties so
    that variable-Gaussian mode can be switched on and the shader is asked
    for that width per axis.
    """
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    with_sigma = sigma_x_nm is not None
    if with_sigma:
        fields += [("sigma_x_pixels", "f4"), ("sigma_y_pixels", "f4")]
        if zdim:
            fields.append(("sigma_z_pixels", "f4"))
    locs = np.zeros(2, dtype=fields)
    locs["x_pos_nm"] = [-SPOT_SEPARATION_NM / 2, SPOT_SEPARATION_NM / 2]
    if with_sigma:
        locs["sigma_x_pixels"] = sigma_x_nm
        locs["sigma_y_pixels"] = sigma_y_nm
        if zdim:
            locs["sigma_z_pixels"] = sigma_z_nm
    dataset = LocalizationDataBaseClass(
        np.rec.array(locs), name="spots", zdim_present=zdim
    )
    dataset.sigma_present = with_sigma
    return dataset


def _skip_unless_available(backend_class):
    if backend_class is InstancedRenderer and not instancing_available():
        pytest.skip("this session has no GL backend with instancing")


def _open(make_napari_viewer, backend_class, dataset):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer, renderer=backend_class(viewer))
    widget.get_dataset_from_test_mode([dataset])
    return widget, viewer


def _capture(viewer, look_at=None):
    """The red channel of an offscreen render, framed with a margin.

    *look_at*, when given, is called after the view is framed and before the
    zoom margin is applied -- the widget's ``change_camera`` keeps the centre
    and zoom, so it has to run after napari's ``reset_view`` and not before.
    The scale bar is switched off: it is white, and on a dim scene the
    brightest pixel of the frame is its label rather than a spot.
    """
    viewer.scale_bar.visible = False
    viewer.reset_view()
    if look_at is not None:
        look_at()
    viewer.camera.zoom *= 0.5  # margin, so nothing is clipped by the frame
    image = np.asarray(viewer.window._qt_viewer.canvas._scene_canvas.render())
    return image[..., 0].astype(np.float64)


def _spot(image, column_range, half):
    """``(row, column, sigma_rows_px, sigma_columns_px)`` of one spot.

    Centroid and per-axis second moments over a window of *half* pixels
    either side of the brightest pixel inside *column_range*, with the floor
    of the window subtracted so background does not widen the spot.  The
    second moment of a Gaussian is its variance, so this reads the width
    without fitting anything -- provided the window holds the whole spot,
    which is the caller's job.
    """
    part = image[:, column_range]
    row, col = np.unravel_index(np.argmax(part), part.shape)
    col += column_range.start
    assert half < row < image.shape[0] - half, "spot too close to the frame"
    assert half < col < image.shape[1] - half, "spot too close to the frame"
    window = image[row - half : row + half + 1, col - half : col + half + 1]
    weight = np.clip(window - window.min(), 0.0, None)
    gy, gx = np.mgrid[-half : half + 1.0, -half : half + 1.0]
    total = weight.sum()
    cy, cx = (weight * gy).sum() / total, (weight * gx).sum() / total
    var_rows = (weight * (gy - cy) ** 2).sum() / total
    var_cols = (weight * (gx - cx) ** 2).sum() / total
    return row + cy, col + cx, float(np.sqrt(var_rows)), float(np.sqrt(var_cols))


def measure_two_spots(screen, sigma_cols_nm, sigma_rows_nm):
    """Both spots on *screen* and on a reference drawn at the screen's scale.

    The reference is rendered at the pixel size read off the render, anchored
    on the left spot, so the right spot's position checks the scale and the
    widths check the model.  Returns ``(nm_per_px, screen_left, screen_right,
    ref_left, ref_right)`` as :func:`_spot` tuples.
    """
    assert screen.max() > 0, "nothing was drawn"
    middle = screen.shape[1] // 2
    # A first pass with a small window finds the spots; a centroid does not
    # care that the window is tight.  It fixes the scale, and the scale sets
    # the window the widths are then read over.
    left = _spot(screen, slice(0, middle), 20)
    right = _spot(screen, slice(middle, screen.shape[1]), 20)
    nm_per_px = SPOT_SEPARATION_NM / (right[1] - left[1])
    widest = max(sigma_cols_nm, sigma_rows_nm)
    assert 0 < nm_per_px < min(sigma_cols_nm, sigma_rows_nm) / 3, "spots not resolved"
    half = int(np.ceil(WINDOW_SIGMAS * widest / nm_per_px))
    left = _spot(screen, slice(0, middle), half)
    right = _spot(screen, slice(middle, screen.shape[1]), half)

    origin = (-left[0] * nm_per_px, -SPOT_SEPARATION_NM / 2 - left[1] * nm_per_px)
    centres = [(0.0, -SPOT_SEPARATION_NM / 2), (0.0, SPOT_SEPARATION_NM / 2)]
    reference = reference_image(
        screen.shape, nm_per_px, origin, centres, sigma_rows_nm, sigma_cols_nm
    )
    ref_left = _spot(reference, slice(0, middle), half)
    ref_right = _spot(reference, slice(middle, reference.shape[1]), half)
    return nm_per_px, left, right, ref_left, ref_right


def report(nm_per_px, left, right, ref_left, ref_right):
    return (
        f"{nm_per_px:.2f} nm/px; screen sigma (rows, cols) "
        f"{left[2] * nm_per_px:.1f}/{left[3] * nm_per_px:.1f} and "
        f"{right[2] * nm_per_px:.1f}/{right[3] * nm_per_px:.1f} nm; reference "
        f"{ref_left[2] * nm_per_px:.1f}/{ref_left[3] * nm_per_px:.1f} nm"
    )


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_screen_splat_has_the_width_the_settings_ask_for(
    make_napari_viewer, backend_class
):
    """An isotropic Gaussian, 600 nm FWHM: the ratio to the model has to be one.

    It was 0.88 until 3.1, and no test noticed.
    """
    _skip_unless_available(backend_class)
    fwhm_nm = 600.0
    sigma_nm = fwhm_nm / FWHM_TO_SIGMA

    widget, viewer = _open(make_napari_viewer, backend_class, _two_spots())
    widget.Esigma_xy.setText(str(fwhm_nm))
    widget.update_sigma()
    screen = _capture(viewer)
    nm_per_px, left, right, ref_left, ref_right = measure_two_spots(
        screen, sigma_nm, sigma_nm
    )
    measured = report(nm_per_px, left, right, ref_left, ref_right)

    # Position: both spots where the reference has them, to within a pixel.
    assert left[:2] == pytest.approx(ref_left[:2], abs=1.0), measured
    assert right[:2] == pytest.approx(ref_right[:2], abs=1.0), measured
    # The reference reads as the model sigma, less the little the cut costs
    # a second moment; the screen reads the same.
    assert ref_left[2] * nm_per_px == pytest.approx(sigma_nm, rel=0.03), measured
    for spot, ref in ((left, ref_left), (right, ref_right)):
        assert spot[2] / ref[2] == pytest.approx(1.0, rel=0.03), measured
        assert spot[3] / ref[3] == pytest.approx(1.0, rel=0.03), measured


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_an_anisotropic_splat_is_the_ellipse_the_uncertainties_describe(
    make_napari_viewer, backend_class
):
    """Four to one in x against y, from per-localization uncertainties."""
    _skip_unless_available(backend_class)
    sigma_x_nm, sigma_y_nm = 400.0, 100.0

    widget, viewer = _open(
        make_napari_viewer, backend_class, _two_spots(sigma_x_nm, sigma_y_nm)
    )
    widget.render_config.gaussian_mode = 1
    widget.data_to_layer_itf.update_layers()
    screen = _capture(viewer)
    nm_per_px, left, right, ref_left, ref_right = measure_two_spots(
        screen, sigma_x_nm, sigma_y_nm
    )
    measured = report(nm_per_px, left, right, ref_left, ref_right)

    assert left[:2] == pytest.approx(ref_left[:2], abs=1.0), measured
    assert right[:2] == pytest.approx(ref_right[:2], abs=1.0), measured
    # The reference is the ellipse asked for: wide along the columns (x),
    # narrow along the rows (y).
    assert ref_left[3] * nm_per_px == pytest.approx(sigma_x_nm, rel=0.03), measured
    assert ref_left[2] * nm_per_px == pytest.approx(sigma_y_nm, rel=0.03), measured
    # And the screen is the same ellipse, axis by axis.
    for spot, ref in ((left, ref_left), (right, ref_right)):
        assert spot[2] / ref[2] == pytest.approx(1.0, rel=0.03), measured
        assert spot[3] / ref[3] == pytest.approx(1.0, rel=0.03), measured


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_3d_splat_seen_from_the_side_shows_its_axial_width(
    make_napari_viewer, backend_class
):
    """Sigma in z three times sigma in x, viewed along y.

    The covariance the shader projects has to be the marginal of the 3-D one
    in the camera's basis; an XZ view is the case where the projected axis is
    z and the drawn ellipse must be 3:1 the other way round from what the XY
    view shows.  This is what every tilted 3-D view of real data exercises,
    since axial precision is worse than lateral in every SMLM modality.
    """
    _skip_unless_available(backend_class)
    sigma_xy_nm, sigma_z_nm = 100.0, 300.0

    widget, viewer = _open(
        make_napari_viewer,
        backend_class,
        _two_spots(sigma_xy_nm, sigma_xy_nm, sigma_z_nm, zdim=True),
    )
    widget.render_config.gaussian_mode = 1
    widget.data_to_layer_itf.update_layers()
    screen = _capture(viewer, lambda: widget.change_camera(set_view_to="XZ"))
    assert viewer.dims.ndisplay == 3
    # In the XZ view z runs along the rows, x along the columns.
    nm_per_px, left, right, ref_left, ref_right = measure_two_spots(
        screen, sigma_xy_nm, sigma_z_nm
    )
    measured = report(nm_per_px, left, right, ref_left, ref_right)

    assert left[:2] == pytest.approx(ref_left[:2], abs=1.5), measured
    assert right[:2] == pytest.approx(ref_right[:2], abs=1.5), measured
    assert ref_left[2] * nm_per_px == pytest.approx(sigma_z_nm, rel=0.03), measured
    assert ref_left[3] * nm_per_px == pytest.approx(sigma_xy_nm, rel=0.03), measured
    for spot, ref in ((left, ref_left), (right, ref_right)):
        assert spot[2] / ref[2] == pytest.approx(1.0, rel=0.05), measured
        assert spot[3] / ref[3] == pytest.approx(1.0, rel=0.05), measured


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_dense_overlap_saturates_the_canvas_but_not_the_export(
    make_napari_viewer, backend_class
):
    """Isolated splats being right says nothing about the crowded case.

    The canvas ends in an 8-bit framebuffer, so once enough Gaussians overlap
    it clips at white and the sum on screen stops growing with the data.  The
    export raster is float and stays linear.
    """
    _skip_unless_available(backend_class)
    n_dense = 200
    sigma_nm = 300.0 / FWHM_TO_SIGMA

    def scene(n):
        locs = np.zeros(n + 1, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
        # n localizations on top of each other at the left, one alone at the
        # right, so the same frame shows both regimes.
        locs["x_pos_nm"] = [-SPOT_SEPARATION_NM / 2] * n + [SPOT_SEPARATION_NM / 2]
        return LocalizationDataBaseClass(
            np.rec.array(locs), name="dense", zdim_present=False
        )

    widget, viewer = _open(make_napari_viewer, backend_class, scene(n_dense))
    widget.Esigma_xy.setText("300")
    widget.update_sigma()
    screen = _capture(viewer)
    middle = screen.shape[1] // 2
    dense_sum = screen[:, :middle].sum()
    single_sum = screen[:, middle:].sum()
    assert screen[:, :middle].max() == 255.0, "the crowded spot did not clip"
    # Two hundred coincident Gaussians would sum to two hundred times one;
    # the canvas gets nowhere near, because every pixel above white is lost.
    assert dense_sum / single_sum < 0.25 * n_dense

    # The export model: exactly linear.
    grid = GaussianGrid.covering(
        ((0.0, 0.0), (-2_000.0, 2_000.0), (-5_000.0, 5_000.0)), 20.0
    )
    coords = np.zeros((n_dense + 1, 3))
    coords[:, 2] = [-SPOT_SEPARATION_NM / 2] * n_dense + [SPOT_SEPARATION_NM / 2]
    sigmas = np.full((n_dense + 1, 3), sigma_nm)
    export = rasterize(coords, sigmas, np.ones(n_dense + 1), grid)[0]
    half = export.shape[1] // 2
    assert export[:, :half].sum() / export[:, half:].sum() == pytest.approx(
        n_dense, rel=1e-3
    )
