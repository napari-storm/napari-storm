"""Drawing localizations as discs instead of Gaussians: a point cloud.

The dock widget always draws Gaussians, the reconstruction. A host that embeds
the renderer may want a point cloud instead -- every localization as an opaque
disc, nearer ones hiding farther ones -- which is what a host's own scatter
plot usually draws, and draws wrongly once points overlap in depth. The disc
footprint is that, on the same engine, chosen through `LayerAppearance`.

The rendering tests read pixels from `_scene_canvas.render()`, which works
headless (see `test_rendered_output`).  Its image is in device pixels, so
distances are converted through the canvas' pixel scale -- and, as there, a
disc is found in the image rather than assumed at its centre: for a viewer
that was never shown, the rendered frame is not centred on the camera.
"""

import json

import numpy as np
import pytest

from napari_storm.core import (
    DEFAULT_MIN_SIZE_PX,
    FOOTPRINT_DISC,
    FOOTPRINT_GAUSSIAN,
    DatasetEntry,
    LayerAppearance,
    NullRenderer,
    RenderRequest,
    Scene,
    SceneFormatError,
    load_scene,
    save_scene,
)
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.points_renderer import NapariPointsRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

#: Every backend: the footprint is part of the contract, so all of them keep it.
BACKENDS = [NapariParticlesRenderer, NapariPointsRenderer, InstancedRenderer]

#: The backends that draw the footprint in their own shader.
SHADER_BACKENDS = [NapariParticlesRenderer, InstancedRenderer]

DISC = LayerAppearance(footprint=FOOTPRINT_DISC)
GAUSSIAN = LayerAppearance(footprint=FOOTPRINT_GAUSSIAN)


def _request(zyx, sigma_nm=1000.0, sigmas=(1.0, 1.0, 1.0), colormap="red", name="d"):
    """Localizations at *zyx*, all of width *sigma_nm*, shaped as the planner does.

    The planner normalizes sigmas to the largest and makes the billboard five
    of them across; *sigmas* are those normalized widths, (z, y, x).
    """
    coords = np.atleast_2d(np.asarray(zyx, dtype=np.float32))
    n = len(coords)
    return RenderRequest(
        coords=coords,
        sigmas=np.tile(np.asarray(sigmas, dtype=np.float32), (n, 1)),
        size=5.0 * sigma_nm,
        values=np.ones(n, dtype=np.float32),
        name=name,
        colormap=colormap,
    )


def _render(viewer):
    """The canvas as a uint8 RGB image, and device pixels per canvas pixel."""
    scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
    image = np.asarray(scene_canvas.render())[..., :3]
    return image, float(scene_canvas.pixel_scale)


def _look_at(viewer, zyx, nm_per_px, ndisplay=2):
    viewer.dims.ndisplay = ndisplay
    if ndisplay == 3:
        viewer.camera.angles = (0.0, 0.0, 90.0)
    viewer.camera.center = tuple(zyx) if ndisplay == 3 else tuple(zyx[1:])
    viewer.camera.zoom = 1.0 / nm_per_px


def _lit(image, threshold=128):
    """Pixels where any channel is lit."""
    return image.max(axis=2) >= threshold


def _box(mask):
    """``(centre_row, centre_col, width, height)`` of the lit pixels, device px."""
    rows, cols = np.nonzero(mask)
    assert rows.size, "nothing was drawn"
    return (
        (rows.min() + rows.max()) / 2.0,
        (cols.min() + cols.max()) / 2.0,
        int(cols.max() - cols.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def _at(image, scale, centre, dx_px=0.0, dy_px=0.0):
    """The RGB pixel *dx_px*, *dy_px* canvas pixels from *centre*."""
    row = int(round(centre[0] + dy_px * scale))
    col = int(round(centre[1] + dx_px * scale))
    return image[row, col].astype(int)


# ------------------------------------------------------------- the contract


def test_appearance_refuses_a_footprint_that_does_not_exist():
    with pytest.raises(ValueError, match="footprint"):
        LayerAppearance(footprint="teapot")
    for bad in (-1.0, float("nan"), float("inf"), "large"):
        with pytest.raises(ValueError, match="min_size_px"):
            LayerAppearance(min_size_px=bad)


def test_the_null_renderer_records_the_footprint():
    renderer = NullRenderer()
    renderer.open(1, _request((1.0, 0.0, 0.0)))
    assert renderer.appearance(1).footprint == FOOTPRINT_GAUSSIAN
    assert renderer.appearance(1).min_size_px == DEFAULT_MIN_SIZE_PX

    renderer.set_appearance(1, LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=5))
    renderer.set_appearance(1, LayerAppearance(opacity=0.5))
    appearance = renderer.appearance(1)
    assert appearance.footprint == FOOTPRINT_DISC
    assert appearance.min_size_px == 5
    assert appearance.opacity == 0.5


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_dataset_opens_as_gaussians(make_napari_viewer, backend_class):
    renderer = backend_class(make_napari_viewer())
    renderer.open(1, _request((1.0, 0.0, 0.0)))
    appearance = renderer.appearance(1)
    assert appearance.footprint == FOOTPRINT_GAUSSIAN
    assert appearance.min_size_px == DEFAULT_MIN_SIZE_PX


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_footprint_survives_an_update(make_napari_viewer, backend_class):
    """A filter change must not turn a point cloud back into Gaussians."""
    renderer = backend_class(make_napari_viewer())
    renderer.open(1, _request([(1.0, 0.0, 0.0), (1.0, 50.0, 50.0)]))
    renderer.set_appearance(
        1, LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=6.0)
    )

    renderer.update(1, _request((1.0, 10.0, 10.0)))

    appearance = renderer.appearance(1)
    assert appearance.footprint == FOOTPRINT_DISC
    assert appearance.min_size_px == 6.0
    renderer.set_appearance(1, GAUSSIAN)
    assert renderer.appearance(1).footprint == FOOTPRINT_GAUSSIAN


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_discs_occlude_and_gaussians_add_up(make_napari_viewer, backend_class):
    """The blend state follows the footprint, through napari's own property."""
    renderer = backend_class(make_napari_viewer())
    renderer.open(1, _request((1.0, 0.0, 0.0)))
    assert renderer.layer(1).blending == "additive"
    renderer.set_appearance(1, DISC)
    assert renderer.layer(1).blending == "opaque"
    renderer.set_appearance(1, GAUSSIAN)
    assert renderer.layer(1).blending == "additive"


# ------------------------------------------------------- what reaches the screen


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_disc_is_flat_inside_one_sigma_and_empty_outside(
    make_napari_viewer, backend_class
):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1000.0))
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)  # sigma = 100 canvas px

    gaussian, scale = _render(viewer)
    renderer.set_appearance(1, DISC)
    disc, _ = _render(viewer)
    centre = _box(_lit(disc))[:2]

    for inside in (0.0, 50.0, 90.0):
        assert _at(disc, scale, centre, dx_px=inside)[0] >= 250, inside
        assert _at(disc, scale, centre, dy_px=inside)[0] >= 250, inside
    for outside in (110.0, 150.0):
        assert _at(disc, scale, centre, dx_px=outside).max() == 0, outside
        assert _at(disc, scale, centre, dy_px=-outside).max() == 0, outside
    if backend_class in SHADER_BACKENDS:
        # the same localization as a Gaussian is not flat
        assert _at(gaussian, scale, centre, dx_px=50.0)[0] < 250


def _overlap_colour(make_napari_viewer, backend_class, order, centre_behind):
    """The colour where a nearer red and a farther green disc overlap.

    The discs are placed along napari's own view direction, so nothing here
    assumes which world axis the screen looks down.  With *centre_behind* the
    camera rotates about a point behind both, which moves nothing on screen.
    """
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    for dataset_id, _name in enumerate(order, start=1):
        renderer.open(dataset_id, _request((0.0, 0.0, float(dataset_id))))
        renderer.set_appearance(dataset_id, DISC)
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0.0, 0.0, 90.0)
    towards = -np.asarray(viewer.camera.view_direction)  # scene to camera
    up = np.asarray(viewer.camera.up_direction)
    middle = np.array([100.0, 0.0, 0.0])
    where = {
        "red": middle + 150 * towards + 40 * up,  # nearer
        "green": middle - 150 * towards - 40 * up,  # farther, overlapping
    }
    for dataset_id, name in enumerate(order, start=1):
        renderer.update(dataset_id, _request(where[name], sigma_nm=80.0, colormap=name))
        renderer.set_appearance(dataset_id, LayerAppearance(colormap=name))
    viewer.camera.center = tuple(middle - 600 * towards if centre_behind else middle)
    viewer.camera.zoom = 0.5

    image, scale = _render(viewer)
    # the two discs together are symmetric about their overlap
    return _at(image, scale, _box(_lit(image))[:2])


@pytest.mark.parametrize("backend_class", BACKENDS)
@pytest.mark.parametrize("order", [("red", "green"), ("green", "red")])
def test_the_nearer_disc_wins_whatever_the_draw_order(
    make_napari_viewer, backend_class, order
):
    """Occlusion by depth, which additive Gaussians do not have."""
    r, g, _b = _overlap_colour(make_napari_viewer, backend_class, order, False)
    assert r > 200 and g < 50, (r, g)


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
@pytest.mark.parametrize("order", [("red", "green"), ("green", "red")])
def test_discs_in_front_of_the_rotation_centre_still_occlude(
    make_napari_viewer, backend_class, order
):
    """The splat shaders wrote NDC depth, -1 to 1, where window depth belongs.

    Under a depth test everything nearer than the rotation centre clamped to
    the same depth, and draw order decided which disc showed.
    """
    r, g, _b = _overlap_colour(make_napari_viewer, backend_class, order, True)
    assert r > 200 and g < 50, (r, g)


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
def test_switching_back_to_gaussians_sums_again(make_napari_viewer, backend_class):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((300.0, 0.0, -60.0), sigma_nm=80.0, colormap="red"))
    renderer.open(2, _request((150.0, 0.0, 60.0), sigma_nm=80.0, colormap="green"))
    for dataset_id in (1, 2):
        renderer.set_appearance(dataset_id, DISC)
        renderer.set_appearance(dataset_id, GAUSSIAN)
    _look_at(viewer, (225.0, 0.0, 0.0), nm_per_px=2.0, ndisplay=3)

    image, scale = _render(viewer)
    r, g, _b = _at(image, scale, _box(_lit(image, threshold=20))[:2])
    assert r > 50 and g > 50, (r, g)


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
def test_a_disc_is_never_smaller_on_screen_than_the_floor(
    make_napari_viewer, backend_class
):
    """Zoomed out, a point cloud must not vanish; that is what the floor is for."""
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1.0))
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)  # 2 nm across: 0.2 px

    renderer.set_appearance(
        1, LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=0.0)
    )
    image, scale = _render(viewer)
    # at most a speck: napari's own markers keep an antialiased pixel or two
    assert np.count_nonzero(_lit(image)) <= (2 * scale) ** 2

    renderer.set_appearance(1, LayerAppearance(min_size_px=12.0))
    image, scale = _render(viewer)
    _row, _col, width, height = _box(_lit(image))
    assert abs(width - 12.0 * scale) <= 2 * scale, width
    assert abs(height - 12.0 * scale) <= 2 * scale, height


def test_the_points_backend_floors_discs_through_napari(make_napari_viewer):
    """Approximately: napari's limit floors the marker sprite, not the disc in it."""
    viewer = make_napari_viewer()
    renderer = NapariPointsRenderer(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1.0))
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)

    lit = []
    for floor in (0.0, 12.0, 40.0):
        renderer.set_appearance(
            1, LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=floor)
        )
        assert renderer.layer(1).canvas_size_limits[0] == floor
        lit.append(np.count_nonzero(_lit(_render(viewer)[0])))
    assert lit[0] < lit[1] < lit[2], lit


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
def test_the_floor_leaves_a_disc_that_is_already_large_alone(
    make_napari_viewer, backend_class
):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1000.0))
    renderer.set_appearance(
        1, LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=20.0)
    )
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)  # 200 px across

    image, scale = _render(viewer)
    width = _box(_lit(image))[2]
    assert abs(width - 200.0 * scale) <= 2 * scale, width


@pytest.mark.parametrize("backend_class", [InstancedRenderer])
def test_an_anisotropic_disc_is_the_projected_one_sigma_ellipse(
    make_napari_viewer, backend_class
):
    """sigma_x twice sigma_y draws an ellipse twice as wide as it is tall.

    The instanced backend only: the billboard fallback has never honoured
    anisotropic widths -- its Gaussians come out round too, on `main` -- and a
    disc can only be as anisotropic as the sigmas its shader receives.
    """
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1000.0, sigmas=(1.0, 0.5, 1.0)))
    renderer.set_appearance(1, DISC)
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)

    image, scale = _render(viewer)
    _row, _col, width, height = _box(_lit(image))
    assert abs(width - 200.0 * scale) <= 2 * scale, width
    assert abs(height - 100.0 * scale) <= 2 * scale, height


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
def test_opacity_zero_still_hides_a_disc(make_napari_viewer, backend_class):
    """Discs ignore opacity, except that zero is how a channel is switched off."""
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, _request((1.0, 0.0, 0.0), sigma_nm=1000.0))
    renderer.set_appearance(1, LayerAppearance(footprint=FOOTPRINT_DISC, opacity=0.3))
    _look_at(viewer, (1.0, 0.0, 0.0), nm_per_px=10.0)
    image, scale = _render(viewer)
    assert _at(image, scale, _box(_lit(image))[:2])[0] >= 250

    renderer.set_appearance(1, LayerAppearance(opacity=0.0))
    image, scale = _render(viewer)
    assert image.max() == 0


# --------------------------------------------------------------- persistence


def test_a_scene_keeps_the_footprint(tmp_path):
    path = tmp_path / "scene.json"
    appearance = LayerAppearance(footprint=FOOTPRINT_DISC, min_size_px=4.0)
    save_scene(path, Scene(datasets=(DatasetEntry(name="a", appearance=appearance),)))

    loaded = load_scene(path).datasets[0].appearance
    assert loaded.footprint == FOOTPRINT_DISC
    assert loaded.min_size_px == 4.0


def test_a_scene_without_a_footprint_reads_as_before(tmp_path):
    """Only a footprint that was set is written, and its absence means "leave it"."""
    path = tmp_path / "scene.json"
    save_scene(path, Scene(datasets=(DatasetEntry(name="a"),)))
    raw = json.loads(path.read_text())
    assert "footprint" not in raw["datasets"][0]["appearance"]

    loaded = load_scene(path).datasets[0].appearance
    assert loaded.footprint is None
    assert loaded.min_size_px is None


def test_a_scene_with_an_unknown_footprint_is_refused(tmp_path):
    path = tmp_path / "scene.json"
    save_scene(path, Scene(datasets=(DatasetEntry(name="a", appearance=DISC),)))
    raw = json.loads(path.read_text())
    raw["datasets"][0]["appearance"]["footprint"] = "teapot"
    path.write_text(json.dumps(raw))

    with pytest.raises(SceneFormatError, match="footprint"):
        load_scene(path)
