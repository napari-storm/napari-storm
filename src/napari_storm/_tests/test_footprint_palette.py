"""The footprint palette: what the Decorators tab offers and a host can choose.

Every entry has to be drawable on both shader backends, behave as its palette
entry says -- how it blends, whether it follows the uncertainty, whether the
size floor lifts it -- and be reachable from the dock.

Rendering reads pixels from `_scene_canvas.render()`, which works headless.
For a viewer that was never shown the frame is not centred on the camera, so
what was drawn is found in the image rather than assumed at its centre; see
`test_footprint`.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (
    BLEND_OPAQUE,
    FOOTPRINT_GAUSSIAN,
    FOOTPRINTS,
    PALETTE,
    LayerAppearance,
    RenderRequest,
    footprint_named,
)
from napari_storm.core.footprints import (
    BLEND_ADDITIVE,
    GROUP_RECONSTRUCTION,
    GROUP_VISUALISATION,
)
from napari_storm.localization_dataset_types import (
    LocalizationDataBaseClass,
    StormDataClass,
)
from napari_storm.localization_dataset_types.data_formats import storm_data_dtype
from napari_storm.napari_particles.footprint_shaders import LEGACY_NAMES, SHAPES
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

SHADER_BACKENDS = [NapariParticlesRenderer, InstancedRenderer]
UNCERTAINTY = [f.name for f in PALETTE if f.uncertainty]
FIXED_SHAPES = ["cross", "square", "diamond"]


def _request(sigma_nm=1000.0, sigmas=(1.0, 1.0, 1.0)):
    """One localization at the origin, shaped as the planner shapes a request."""
    return RenderRequest(
        coords=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        sigmas=np.array([sigmas], dtype=np.float32),
        size=5.0 * sigma_nm,
        values=np.ones(1, dtype=np.float32),
        name="one",
        colormap="red",
    )


def _drawn(make_napari_viewer, backend_class, name, request, nm_per_px=10.0, **kw):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    renderer.open(1, request)
    renderer.set_appearance(1, LayerAppearance(footprint=name, **kw))
    viewer.dims.ndisplay = 2
    viewer.camera.center = (0.0, 0.0)
    viewer.camera.zoom = 1.0 / nm_per_px
    scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
    image = np.asarray(scene_canvas.render())[..., :3]
    return renderer, image, float(scene_canvas.pixel_scale)


def _lit(image, threshold=40):
    return image.max(axis=2) >= threshold


def _extent(mask):
    """``(width, height)`` of the lit pixels, in device pixels."""
    rows, cols = np.nonzero(mask)
    assert rows.size, "nothing was drawn"
    return int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


# ------------------------------------------------------------ the palette


def test_every_footprint_has_a_shader_and_every_shader_a_footprint():
    assert set(SHAPES) == set(FOOTPRINTS)


def test_the_scientific_gaussian_comes_first_and_alone():
    assert PALETTE[0].name == FOOTPRINT_GAUSSIAN
    assert [f.name for f in PALETTE if f.reconstruction] == [FOOTPRINT_GAUSSIAN]
    assert [f.name for f in PALETTE if f.exportable] == [FOOTPRINT_GAUSSIAN]


def test_names_and_labels_are_unique():
    assert len(set(FOOTPRINTS)) == len(PALETTE)
    assert len({f.label for f in PALETTE}) == len(PALETTE)


def test_every_entry_says_how_it_blends_and_where_it_belongs():
    for footprint in PALETTE:
        assert footprint.blend in (BLEND_ADDITIVE, BLEND_OPAQUE)
        assert footprint.group in (GROUP_RECONSTRUCTION, GROUP_VISUALISATION)
        assert footprint.extent_sigmas > 0
        assert footprint.description


def test_napari_particles_names_still_resolve():
    for legacy, name in LEGACY_NAMES.items():
        assert name in FOOTPRINTS, legacy


def test_an_unknown_footprint_is_named_in_the_error():
    with pytest.raises(ValueError, match="teapot"):
        footprint_named("teapot")


# -------------------------------------------------- what reaches the screen


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
@pytest.mark.parametrize("name", FOOTPRINTS)
def test_every_footprint_draws_and_blends_as_it_says(
    make_napari_viewer, backend_class, name
):
    renderer, image, _scale = _drawn(
        make_napari_viewer, backend_class, name, _request()
    )
    assert np.count_nonzero(_lit(image)) > 100
    expected = "opaque" if footprint_named(name).blend == BLEND_OPAQUE else "additive"
    assert renderer.layer(1).blending == expected


@pytest.mark.parametrize("name", UNCERTAINTY)
def test_uncertainty_footprints_take_the_shape_of_the_sigma_ellipse(
    make_napari_viewer, name
):
    """sigma_x twice sigma_y: twice as wide as tall.

    The instanced backend only: the billboard fallback has never honoured
    anisotropic widths, its Gaussians included.
    """
    request = _request(sigmas=(1.0, 0.5, 1.0))
    _r, image, _s = _drawn(make_napari_viewer, InstancedRenderer, name, request)
    width, height = _extent(_lit(image))
    assert width > 1.4 * height, (width, height)


@pytest.mark.parametrize("name", FIXED_SHAPES)
def test_fixed_shapes_ignore_the_sigma_ellipse(make_napari_viewer, name):
    request = _request(sigmas=(1.0, 0.5, 1.0))
    _r, image, _s = _drawn(make_napari_viewer, InstancedRenderer, name, request)
    width, height = _extent(_lit(image))
    assert abs(width - height) <= 0.1 * width, (width, height)


@pytest.mark.parametrize("backend_class", SHADER_BACKENDS)
@pytest.mark.parametrize("name", FOOTPRINTS)
def test_the_floor_lifts_every_visualisation_but_never_the_gaussian(
    make_napari_viewer, backend_class, name
):
    """Zoomed out, a 1 nm localization is a fifth of a pixel across."""
    _r, image, scale = _drawn(
        make_napari_viewer,
        backend_class,
        name,
        _request(sigma_nm=1.0),
        min_size_px=12.0,
    )
    if footprint_named(name).reconstruction:
        lit = np.count_nonzero(_lit(image))
        assert lit <= 4 * scale * scale, lit
    else:
        # Anything drawn counts, and in device pixels.  Glow is a bright core
        # in a faint halo: its core alone is 4 pixels at scale 1 and 24 at
        # scale 2, so a bright-pixel count passed on Retina displays and
        # failed on every CI runner.  Drawn at all, it covers 800.  Without
        # the floor every footprint draws 0 or 1 pixel.
        lit = np.count_nonzero(_lit(image, threshold=1))
        assert lit >= 20 * scale * scale, lit


# ----------------------------------------------------------------- the dock


def _plain(name="plain", n=200):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 3000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _with_uncertainty(name="storm", n=200):
    locs = np.rec.array(np.zeros(n, dtype=storm_data_dtype))
    locs.x_pos_pixels = np.linspace(0, 50, n)
    locs.y_pos_pixels = np.linspace(0, 30, n)
    rng = np.random.default_rng(0)
    locs.sigma_x_pixels = rng.uniform(0.08, 0.12, n)
    locs.sigma_y_pixels = rng.uniform(0.04, 0.06, n)
    locs.photon_count = rng.uniform(800.0, 1200.0, n)
    return StormDataClass(
        locs=locs,
        name=name,
        pixelsize_nm=100.0,
        zdim_present=False,
        sigma_present=True,
        photon_count_present=True,
    )


def _dock(make_napari_viewer, datasets):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode(datasets)
    return widget


def _choose(widget, name):
    widget.Bfootprint.setCurrentIndex(widget.Bfootprint.findData(name))


def _footprints(widget):
    renderer = widget.data_to_layer_itf.renderer
    return [
        renderer.appearance(d.dataset_id).footprint
        for d in widget.localization_datasets
    ]


def test_the_decorators_tab_offers_the_whole_palette(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    combo = widget.Bfootprint
    offered = [combo.itemData(i) for i in range(combo.count())]
    assert [name for name in offered if name in FOOTPRINTS] == list(FOOTPRINTS)
    assert combo.currentData() == FOOTPRINT_GAUSSIAN


def test_a_style_reaches_every_dataset_and_the_store(make_napari_viewer):
    widget = _dock(make_napari_viewer, [_plain("a"), _plain("b")])
    _choose(widget, "sphere")

    assert _footprints(widget) == ["sphere", "sphere"]
    for dataset in widget.localization_datasets:
        assert widget.dataset_store.state_of(dataset).appearance.footprint == "sphere"

    _choose(widget, FOOTPRINT_GAUSSIAN)
    assert _footprints(widget) == [FOOTPRINT_GAUSSIAN, FOOTPRINT_GAUSSIAN]
    renderer = widget.data_to_layer_itf.renderer
    assert renderer.layer(widget.localization_datasets[0].dataset_id).blending == (
        "additive"
    )


def test_a_dataset_loaded_later_joins_the_chosen_style(make_napari_viewer):
    widget = _dock(make_napari_viewer, [_plain("a")])
    _choose(widget, "cross")
    widget.get_dataset_from_test_mode([_plain("b")])
    assert _footprints(widget) == ["cross"]


def test_the_size_floor_reaches_the_renderer(make_napari_viewer):
    widget = _dock(make_napari_viewer, [_plain("a")])
    _choose(widget, "disc")
    widget.Sfootprint_min_size.setValue(9)
    dataset = widget.localization_datasets[0]
    appearance = widget.data_to_layer_itf.renderer.appearance(dataset.dataset_id)
    assert appearance.min_size_px == 9.0


def test_only_visualisations_offer_a_floor_and_carry_the_export_note(
    make_napari_viewer,
):
    widget = _dock(make_napari_viewer, [_plain("a")])
    assert not widget.Sfootprint_min_size.isEnabled()
    assert widget.Lfootprint_note.isHidden()
    _choose(widget, "ring")
    assert widget.Sfootprint_min_size.isEnabled()
    assert not widget.Lfootprint_note.isHidden()


def test_uncertainty_is_offered_only_where_it_means_something(make_napari_viewer):
    widget = _dock(make_napari_viewer, [_plain("a")])
    _choose(widget, "ring")
    assert not widget.Cfootprint_uncertainty.isEnabled()  # no uncertainty recorded

    widget.get_dataset_from_test_mode([_with_uncertainty()])
    _choose(widget, "cross")
    assert not widget.Cfootprint_uncertainty.isEnabled()  # a fixed shape
    _choose(widget, "ring")
    assert widget.Cfootprint_uncertainty.isEnabled()


def test_the_uncertainty_box_is_the_variable_size_switch(make_napari_viewer):
    widget = _dock(make_napari_viewer, [_with_uncertainty()])
    _choose(widget, "ring")

    widget.Cfootprint_uncertainty.setChecked(True)
    assert widget.Brenderoptions.currentIndex() == 1
    assert widget.render_gaussian_mode == 1

    widget.Brenderoptions.setCurrentIndex(0)
    assert not widget.Cfootprint_uncertainty.isChecked()
    assert widget.render_gaussian_mode == 0


def test_markers_keep_their_colour_when_widths_come_from_the_uncertainty(
    make_napari_viewer,
):
    """In variable-size mode the values are intensity weights, which a Gaussian
    sums.  A marker is not summed, so it is drawn at full value instead -- its
    size and shape already show the uncertainty -- while the Gaussian keeps them.
    """
    widget = _dock(make_napari_viewer, [_with_uncertainty()])
    dataset = widget.localization_datasets[0]

    def drawn_values():
        return widget.data_to_layer_itf.render_state[dataset.dataset_id].values

    _choose(widget, "ring")
    widget.Cfootprint_uncertainty.setChecked(True)
    assert np.all(drawn_values() == 1.0)

    _choose(widget, FOOTPRINT_GAUSSIAN)
    assert np.ptp(drawn_values()) > 0

    _choose(widget, "glow")  # additive: summed like the Gaussian, so weighted
    assert np.ptp(drawn_values()) > 0


def test_a_saved_scene_brings_the_style_back(make_napari_viewer, tmp_path):
    widget = _dock(make_napari_viewer, [_plain("a")])
    _choose(widget, "sphere")
    widget.Sfootprint_min_size.setValue(7)
    path = tmp_path / "scene.json"
    widget.save_scene_to(path)

    _choose(widget, FOOTPRINT_GAUSSIAN)
    widget.Sfootprint_min_size.setValue(2)
    widget.load_scene_from(path)

    assert widget.Bfootprint.currentData() == "sphere"
    assert widget.Sfootprint_min_size.value() == 7
    assert _footprints(widget) == ["sphere"]


def test_the_points_backend_draws_markers_as_napari_symbols(make_napari_viewer):
    """The comparison backend has no shader: an opaque footprint becomes napari's
    nearest marker, and an additive one keeps its splat-like look."""
    from napari_storm.napari_particles.points_renderer import NapariPointsRenderer

    renderer = NapariPointsRenderer(make_napari_viewer())
    renderer.open(1, _request())
    for name, symbol, blending in (
        ("ring", "ring", "opaque"),
        ("cross", "cross", "opaque"),
        ("diamond", "diamond", "opaque"),
        ("glow", "disc", "additive"),
    ):
        renderer.set_appearance(1, LayerAppearance(footprint=name))
        layer = renderer.layer(1)
        drawn = {str(getattr(s, "value", s)) for s in np.atleast_1d(layer.symbol)}
        assert drawn == {symbol}, name
        assert layer.blending == blending, name
