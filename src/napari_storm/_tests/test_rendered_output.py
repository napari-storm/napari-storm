"""What actually reaches the screen, compared between backends.

Everything else in the suite checks state: buffer sizes, layer identity, GL
flags. None of it would have caught the defect this file exists for -- a
Gaussian split along the quad diagonal, reported from a real session and
invisible to every state assertion, because every piece of state was correct.

`viewer.screenshot()` returns black offscreen, which is why the earlier rounds
of this work could not check pixels. The canvas' own
``_scene_canvas.render()`` does work headless, and that is what these use.
"""
import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles._napari_compat import instancing_available
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer


def _two_spots():
    """Two well-separated localizations.

    Separated so each splat is isolated, and spread over a real extent so the
    screen-space cap does not clamp the Gaussian to half the field of view and
    fill the canvas -- which is what a tight cluster with a large FWHM does,
    leaving a splat clipped at the frame edge and not centred in anything.
    """
    locs = np.zeros(2, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = [-3000.0, 3000.0]
    locs["y_pos_nm"] = [0.0, 0.0]
    return LocalizationDataBaseClass(np.rec.array(locs), name="spots", zdim_present=False)


def _render(make_napari_viewer, backend_class, fwhm="600"):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer, renderer=backend_class(viewer))
    widget.get_dataset_from_test_mode([_two_spots()])
    widget.Esigma_xy.setText(fwhm)
    widget.update_sigma()
    viewer.reset_view()
    viewer.camera.zoom *= 0.5  # margin, so nothing is clipped by the frame
    image = np.asarray(viewer.window._qt_viewer.canvas._scene_canvas.render())
    return image[..., 0].astype(float)


HALF = 40

# Offsets of the sampling window, and of each of its three mirrorings.
_GRID_Y, _GRID_X = np.mgrid[-HALF : HALF + 1.0, -HALF : HALF + 1.0]
MIRRORS = {
    "rows": (-_GRID_Y, _GRID_X),
    "columns": (_GRID_Y, -_GRID_X),
    "diagonal": (_GRID_X, _GRID_Y),
}


def _splat_centre(image):
    """The splat's centre, refined from the brightest pixel to the centroid.

    The brightest *pixel* is not the centre.  The Gaussian's centre falls
    wherever it falls between samples, and in this scene it lands almost
    exactly half a pixel off in x.  That matters to the symmetry check below:
    mirroring about a point half a pixel from the true centre shifts the copy
    by a whole pixel, which then reads as asymmetry the renderer never drew.
    It measured 13/255 that way on this machine and 21/255 on CI's software
    rasterizers -- the same image, a different sub-pixel landing.

    Cropping to a window rather than to a threshold bounding box, for the
    original reason: a bounding box silently includes anything else that is
    lit, and collapses onto the frame edge when a splat is clipped, so it
    measures the frame rather than the splat.
    """
    row, col = np.unravel_index(np.argmax(image), image.shape)
    # One past HALF, because the samples below interpolate between neighbours.
    margin = HALF + 1
    assert margin <= row < image.shape[0] - margin, "splat too close to the frame"
    assert margin <= col < image.shape[1] - margin, "splat too close to the frame"

    window = image[row - HALF : row + HALF + 1, col - HALF : col + HALF + 1]
    # Weighted by intensity above a floor, so that the rim and the background
    # do not drag the centroid off the peak.
    weight = np.clip(window - window.max() * 0.05, 0, None)
    return (
        row + (weight * _GRID_Y).sum() / weight.sum(),
        col + (weight * _GRID_X).sum() / weight.sum(),
    )


def _sample(image, ys, xs):
    """Bilinear samples of *image* at fractional coordinates.

    Written out rather than taken from `scipy.ndimage`: SciPy is only a
    transitive dependency here, and this is four lines.
    """
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = ys - y0
    fx = xs - x0
    return (
        image[y0, x0] * (1 - fy) * (1 - fx)
        + image[y0 + 1, x0] * fy * (1 - fx)
        + image[y0, x0 + 1] * (1 - fy) * fx
        + image[y0 + 1, x0 + 1] * fy * fx
    )


def _splat_and_mirrors(image):
    """The splat about its own centre, and the same window mirrored three ways.

    Every mirror is sampled from the same image through the same interpolator
    as the splat itself, so the interpolator's own error lands on both sides
    and cancels.  What is left is the renderer's asymmetry.
    """
    centre_y, centre_x = _splat_centre(image)
    splat = _sample(image, centre_y + _GRID_Y, centre_x + _GRID_X)
    mirrors = {
        name: _sample(image, centre_y + offset_y, centre_x + offset_x)
        for name, (offset_y, offset_x) in MIRRORS.items()
    }
    return splat, mirrors


def _asymmetry(image):
    """How far the splat departs from its mirrorings, worst mirror first."""
    splat, mirrors = _splat_and_mirrors(image)
    return {
        name: np.abs(splat - mirrored).max() / max(splat.max(), 1)
        for name, mirrored in mirrors.items()
    }


@pytest.mark.parametrize(
    "backend_class", [NapariParticlesRenderer, InstancedRenderer]
)
def test_a_splat_is_a_symmetric_gaussian(make_napari_viewer, backend_class):
    """The split showed up as a break in this symmetry, nothing else.

    A Gaussian is symmetric under horizontal *and* vertical mirroring. A quad
    whose two triangles interpolate different texture maps is not: the halves
    either side of the diagonal disagree.
    """
    if backend_class is InstancedRenderer and not instancing_available():
        pytest.skip("this session has no GL backend with instancing")

    image = _render(make_napari_viewer, backend_class)
    splat, _ = _splat_and_mirrors(image)

    # Mirror-symmetric both ways.  The split version is not: its two triangles
    # interpolate different maps, so the halves disagree across the diagonal.
    #
    # 0.03 is measured, not guessed, and the test below is what measures it:
    # aligned this way a clean splat comes to 0.011, one triangle a single
    # pixel out to 0.037, and two pixels out to 0.074.  The 0.08 this
    # assertion carried before the window was centred sat above two of those
    # three -- loose enough to pass the defect the file exists for, and still
    # tight enough to fail a clean image once the rasterizer changed.
    for name, difference in _asymmetry(image).items():
        assert difference < 0.03, (name, difference)

    # Falling away from the centre, not flat and not lumpy.
    profile = splat[HALF, HALF:]
    assert np.all(np.diff(profile) <= 1.0)
    assert profile[-1] < 0.6 * profile[0]


def _split_along_the_diagonal(image, shift):
    """Re-render one of the quad's two triangles *shift* pixels out.

    A stand-in for the reported defect, which was the two triangles
    interpolating texture maps that disagreed.  Displacing one of them is the
    same class of error, and unlike the original it can be dialled down to the
    smallest version worth catching.
    """
    row, col = np.unravel_index(np.argmax(image), image.shape)
    rows, columns = np.indices(image.shape)
    triangle = (rows - row) > (columns - col)

    split = image.copy()
    split[triangle] = np.roll(image, shift, axis=1)[triangle]
    return split


@pytest.mark.parametrize("shift", [1, 2])
def test_the_symmetry_check_would_catch_a_split_splat(make_napari_viewer, shift):
    """Proof the tolerance above is load-bearing rather than merely satisfied.

    Without this, 0.03 is a number that happens to pass today and nothing
    says how much room is left before it stops meaning anything.  Its
    predecessor, 0.08, passed both of these.
    """
    clean = _render(make_napari_viewer, NapariParticlesRenderer)
    assert max(_asymmetry(clean).values()) < 0.03, "the control is not clean"

    split = _split_along_the_diagonal(clean, shift)
    assert max(_asymmetry(split).values()) > 0.03


def test_the_two_gaussian_backends_render_the_same_image(make_napari_viewer):
    """The instanced backend has to be a drop-in, not merely a fast one."""
    if not instancing_available():
        pytest.skip("this session has no GL backend with instancing")

    reference = _render(make_napari_viewer, NapariParticlesRenderer)
    instanced = _render(make_napari_viewer, InstancedRenderer)

    assert reference.shape == instanced.shape
    lit = reference > 10
    assert lit.sum() > 1000, "the reference drew almost nothing"

    # Same footprint, to within antialiasing at the rim.
    overlap = np.logical_and(lit, instanced > 10).sum()
    assert overlap / lit.sum() > 0.98

    # And the same intensities inside it.
    difference = np.abs(reference - instanced)[lit].max() / reference.max()
    assert difference < 0.06, difference
