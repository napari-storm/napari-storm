"""What napari computes per mesh vertex stops being per localization.

Instancing decouples the mesh from the data: the mesh is one quad, and every
localization is drawn from it by the shader. Anything napari derives from
*vertices* therefore silently stops describing the localizations. Three defects
reported from real sessions were all this same shape, so they are pinned
together rather than filed apart:

* the Gaussians split along the quad diagonal (per-vertex texture coordinates),
* Z-colour-coding came out one hue (per-vertex colormap lookup),
* the show/hide checkbox and opacity slider did nothing (per-fragment alpha
  overwritten, discarding the layer opacity napari had put there),
* the colormap-range slider and the cutoff/factor boxes did nothing (napari's
  per-vertex contrast window no longer applied once the shader took over the
  colormap lookup).

The first has its own file. The rest are here, all checked in pixels, because
every piece of *state* was correct in each case -- the layer dutifully reported
the opacity and the contrast limits it had been given while drawing something
that ignored both.
"""
import colorsys

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

BACKENDS = [InstancedRenderer, NapariParticlesRenderer]


def _dataset(n=4000, zdim=False, name="ds"):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    locs = np.zeros(n, dtype=fields)
    locs["x_pos_nm"] = np.linspace(0, 8000, n)
    locs["y_pos_nm"] = 4000 + 800 * np.sin(np.linspace(0, 6, n))
    if zdim:
        locs["z_pos_nm"] = np.linspace(-500, 500, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=zdim)


def _widget(make_napari_viewer, backend_class=None, zdim=False):
    viewer = make_napari_viewer()
    kwargs = {}
    if backend_class is not None:
        kwargs["renderer"] = backend_class(viewer)
    widget = napari_storm(napari_viewer=viewer, **kwargs)
    widget.get_dataset_from_test_mode([_dataset(zdim=zdim)])
    return widget, viewer


def _rendered(viewer):
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    return np.asarray(canvas.render())[..., :3].astype(float)


def _brightness(viewer):
    return float(_rendered(viewer).mean())


# ------------------------------------------------------- colour per instance


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_z_colour_coding_spans_the_colormap(make_napari_viewer, backend_class):
    """A rainbow, not a ramp of one hue.

    The instanced mesh is one quad whose four vertices all carry 1.0, so
    napari's per-vertex colormap lookup returned a single colour for every
    localization and only its brightness varied -- reported as "every
    localization was either red or pink".
    """
    widget, viewer = _widget(make_napari_viewer, backend_class, zdim=True)

    widget.Bz_color_coding.setChecked(True)

    image = _rendered(viewer)
    lit = image[image.sum(axis=2) > 40]
    assert len(lit) > 100, "nothing was drawn to measure"

    hues = np.array([colorsys.rgb_to_hsv(*(pixel / 255.0))[0] for pixel in lit[::17]])
    occupied = len(np.unique((hues * 20).astype(int)))
    # A single-hue ramp lands in one or two bins; a colormap sweep spans many.
    assert occupied >= 6, f"only {occupied} hue bins -- this is a ramp, not a rainbow"
    assert np.ptp(hues) > 0.4


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_colormap_change_reaches_the_pixels(make_napari_viewer, backend_class):
    """The shader samples the colormap itself, so it has to be kept in sync."""
    widget, viewer = _widget(make_napari_viewer, backend_class)
    dataset = widget.localization_datasets[0]
    itf = widget.data_to_layer_itf

    itf.set_appearance(dataset, colormap="red")
    red = _rendered(viewer)
    itf.set_appearance(dataset, colormap="green")
    green = _rendered(viewer)

    assert red[..., 0].sum() > red[..., 1].sum()
    assert green[..., 1].sum() > green[..., 0].sum()


# ------------------------------------------------------------------ opacity


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_opacity_zero_hides_the_channel(make_napari_viewer, backend_class):
    """How the per-dataset checkbox hides a channel.

    The instanced fragment shader wrote `gl_FragColor.a = gaussian`, discarding
    the opacity napari had put in alpha -- and under additive blending alpha is
    the only thing scaling what a layer contributes, so opacity did nothing.

    Measured against the *empty* canvas rather than a fraction of the visible
    brightness. The old form asserted ``< 0.5 * visible``, which silently
    assumed the data outweighed the canvas background; it does not on a
    landscape canvas showing a wide, thin dataset, so the threshold tracked how
    much screen the localizations happened to cover rather than whether opacity
    worked. Comparing to the layer hidden is the thing the test means, and it
    holds whatever shape the data is.
    """
    widget, viewer = _widget(make_napari_viewer, backend_class)
    dataset = widget.localization_datasets[0]
    layer = widget.data_to_layer_itf.layer_for(dataset)
    visible = _brightness(viewer)
    layer.visible = False
    empty = _brightness(viewer)
    layer.visible = True
    assert visible > empty, "fixture must draw something to begin with"

    widget.data_to_layer_itf.set_appearance(dataset, opacity=0.0)

    assert _brightness(viewer) == pytest.approx(empty, abs=1e-6)


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_opacity_is_a_continuum_not_a_switch(make_napari_viewer, backend_class):
    """The slider has to work too, not only the checkbox's zero."""
    widget, viewer = _widget(make_napari_viewer, backend_class)
    dataset = widget.localization_datasets[0]
    itf = widget.data_to_layer_itf

    itf.set_appearance(dataset, opacity=1.0)
    full = _brightness(viewer)
    itf.set_appearance(dataset, opacity=0.3)
    dimmed = _brightness(viewer)

    assert dimmed < full


# ---------------------------------------------------------- contrast limits


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_contrast_limits_reach_the_pixels(make_napari_viewer, backend_class):
    """The colormap-range slider and cutoff/factor boxes drive these.

    Sampling the colormap in our own shader meant napari's `node.clim` no
    longer applied to anything, so three visible controls in ChannelControls
    changed the layer state and nothing else.
    """
    widget, viewer = _widget(make_napari_viewer, backend_class, zdim=True)
    widget.Bz_color_coding.setChecked(True)
    dataset = widget.localization_datasets[0]
    before = _rendered(viewer)

    widget.data_to_layer_itf.set_appearance(dataset, contrast_limits=(0.4, 0.6))

    assert float(np.abs(_rendered(viewer) - before).mean()) > 0.05


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_contrast_window_selects_which_values_are_coloured(
    make_napari_viewer, backend_class
):
    """Semantics, not just 'something changed'.

    Values below the window clamp to the bottom of the colormap and values
    above it to the top, so narrowing onto the top half must push the visible
    hues towards the colormap's upper end.
    """
    widget, viewer = _widget(make_napari_viewer, backend_class, zdim=True)
    widget.Bz_color_coding.setChecked(True)
    dataset = widget.localization_datasets[0]

    widget.data_to_layer_itf.set_appearance(dataset, contrast_limits=(0.0, 1.0))
    full = _rendered(viewer)
    widget.data_to_layer_itf.set_appearance(dataset, contrast_limits=(0.75, 1.0))
    upper = _rendered(viewer)

    def _hues(image):
        lit = image[image.sum(axis=2) > 40]
        if len(lit) < 20:
            return np.array([0.0])
        return np.array(
            [colorsys.rgb_to_hsv(*(p / 255.0))[0] for p in lit[::17]]
        )

    # A window over the top quarter spans less of the colour circle than the
    # whole range does.
    assert np.ptp(_hues(upper)) < np.ptp(_hues(full))


def test_a_zero_width_contrast_window_never_reaches_the_shader(make_napari_viewer):
    """napari refuses it upstream, which is why the shader has not divided by zero.

    Written expecting to have to survive a degenerate window -- a range slider
    can produce one mid-drag -- and it turns out napari rejects it before the
    layer accepts it. The floor in `set_contrast_limits` is therefore belt and
    braces rather than load-bearing, and both halves are recorded so the next
    person does not have to rediscover which.
    """
    widget, _viewer = _widget(make_napari_viewer, InstancedRenderer)
    dataset = widget.localization_datasets[0]

    with pytest.raises(ValueError, match="monotonically increasing"):
        widget.data_to_layer_itf.set_appearance(dataset, contrast_limits=(0.5, 0.5))

    # Called directly, the filter still refuses to divide by zero.
    layer = widget.data_to_layer_itf.layer_for(dataset)
    layer._billboard_filter.set_contrast_limits(0.5, 0.5)
    assert layer._billboard_filter.fshader["clim_range"].value > 0


# -------------------------------------------------------------- the controls


def test_the_export_button_is_hidden_until_something_is_loaded(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    assert not widget.Bexport_image.isVisibleTo(widget)

    widget.get_dataset_from_test_mode([_dataset()])

    assert widget.Bexport_image.isVisibleTo(widget)


def test_the_dock_opens_wide_enough_to_read(make_napari_viewer):
    """Otherwise every session starts by dragging the dock wider."""
    widget = napari_storm(napari_viewer=make_napari_viewer())

    assert widget.minimumWidth() >= widget.PREFERRED_WIDTH_PX
    # A minimum, not a fixed width: the dock must still be resizable.
    assert widget.maximumWidth() > widget.PREFERRED_WIDTH_PX


# ------------------------------------------------------- 2-D display mode


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_backends_draw_the_same_in_2d_display_mode(
    make_napari_viewer, backend_class
):
    """napari slices a Surface; the instanced mesh is a stand-in for the data.

    The quad used to sit at z = 0 while `_extent_data` reported the data plane
    at z = 1, so napari's slice never intersected it and the instanced backend
    rendered nothing at all in 2-D. The quad now sits on the data's own plane.

    The plugin does not switch to `ndisplay=2` yet -- that is blocked on the
    reference-image overlay, see `adjust_available_options_to_data_dimension` --
    but the backend must be ready for it, and a user can switch by hand today.
    """
    widget, viewer = _widget(make_napari_viewer, backend_class)
    lit_in_3d = _brightness(viewer)

    viewer.dims.ndisplay = 2

    assert _brightness(viewer) > 0.5 * lit_in_3d


def test_flat_data_has_no_padded_depth(make_napari_viewer):
    """The padding frames whole splats; across a single plane it has nothing to
    frame, and it is what put napari's slice plane where the data is not."""
    widget, viewer = _widget(make_napari_viewer, InstancedRenderer)
    layer = widget.data_to_layer_itf.layer_for(widget.localization_datasets[0])

    extent = np.asarray(layer._extent_data)

    assert extent[0][0] == extent[1][0], "flat data must have zero depth extent"
    # ...while the lateral extent is still padded by the splat radius.
    assert extent[1][1] - extent[0][1] > 0
