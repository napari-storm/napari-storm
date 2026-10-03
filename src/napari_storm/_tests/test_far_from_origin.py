"""Splats far from the origin, seen through napari-storm's perspective camera.

Both shader backends draw every localization on a quad that sits at the origin
and is moved into place in the vertex shader, and both used to recover that
quad's corner by inverting the transforms on gl_Position.  Through the
perspective camera the inversion falls apart for a quad far from the camera:
VisPy's near plane is 1/3162 of the eye distance, so the depth row subtracts
two all-but-equal float32 numbers, and the error is multiplied by the camera's
distance from the quad.  Zoomed in, tilted, and looking at data tens of
micrometres from the origin, splats were cut off from changing sides and
flickered at the slightest turn -- or were not drawn at all.

Translation must not change what is drawn: the same scene, with the camera on
it, has to look the same at the origin and 50 um away.  Rendered with
`_scene_canvas.render()`, as the other rendering tests are; the off-centre
frame a never-shown canvas renders is shared by both images, so it cancels.
"""

import numpy as np
import pytest

from napari_storm.core import LayerAppearance, RenderRequest
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

SIGMA_NM = 20 / 2.354
FAR_NM = 5e4


def _cluster():
    """A few localizations around a point, in (z, y, x) nm, 3-D."""
    rng = np.random.default_rng(2)
    local = np.zeros((40, 3), dtype=np.float32)
    local[:, 0] = rng.uniform(-300, 300, len(local))
    local[:, 1:] = rng.uniform(-150, 150, (len(local), 2))
    return local


def _render(make_napari_viewer, backend_class, footprint, offset_nm):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    coords = _cluster() + np.array([0.0, offset_nm, offset_nm], dtype=np.float32)
    renderer.open(
        1,
        RenderRequest(
            coords=coords,
            sigmas=np.ones_like(coords),
            size=5.0 * SIGMA_NM,
            values=np.ones(len(coords), dtype=np.float32),
            name="cluster",
            colormap="gray",
        ),
    )
    renderer.set_appearance(1, LayerAppearance(footprint=footprint))
    viewer.dims.ndisplay = 3
    # As the dock sets it up: a 50 degree perspective camera.
    viewer.camera.perspective = 50
    viewer.camera.center = (0.0, offset_nm, offset_nm)
    viewer.camera.zoom = 8.0
    viewer.camera.angles = (0.0, 75.0, 40.0)
    scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
    return np.asarray(scene_canvas.render())[..., :3].astype(int)


@pytest.mark.parametrize("backend_class", [InstancedRenderer, NapariParticlesRenderer])
@pytest.mark.parametrize("footprint", ["gaussian", "disc", "airy"])
def test_a_scene_far_from_the_origin_looks_as_it_does_at_it(
    make_napari_viewer, backend_class, footprint
):
    near = _render(make_napari_viewer, backend_class, footprint, 0.0)
    far = _render(make_napari_viewer, backend_class, footprint, FAR_NM)
    assert near.max() > 100, "nothing was drawn to compare"
    difference = np.abs(near - far).max(axis=2)
    # Fixed, the two differ by a few pixels on a disc's hard edge and by
    # nothing else.  Before, 17-29 % of what the instanced backend drew
    # differed, and the billboard backend scattered splats across the screen.
    assert difference.mean() < 0.1, difference.mean()
    assert np.count_nonzero(difference > 30) <= 10
