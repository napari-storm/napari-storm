"""Splats must keep summing, through every gesture that redraws them.

The defect this pins down was reported from a real session: the first render
looked right, and then any update -- a filter, a switch between fixed and
variable Gaussian -- and the Gaussians stopped adding, each one covering the
last instead of accumulating.

The cause was napari re-applying its own blending preset. napari recomputes
layer order on insert, remove, reorder *and* visibility change, and its `order`
setter calls `_on_blending_change`, which sets
``blend_func=('src_alpha', 'zero', ...)`` for the lowest layer. A destination
factor of `zero` discards whatever is already in the buffer, so each splat
replaces its neighbours rather than summing with them.

Until P0-01 this was repaired by accident: every update destroyed and rebuilt
the layer, and `add_to_viewer` set the state again on the way back in. Updating
in place removed the rebuild, and with it the repair -- so this is a regression
the in-place update introduced, in the *current* renderer as much as the
instanced one, and it is tested on both.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles.points_renderer import NapariPointsRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

#: What true additive blending needs: add the source, keep the destination.
ADDITIVE = ("src_alpha", "one")


def _dataset(name="ds", n=2000, zdim=False):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    locs = np.zeros(n, dtype=fields)
    locs["x_pos_nm"] = np.linspace(10_000, 12_000, n)
    locs["y_pos_nm"] = np.linspace(10_000, 12_000, n)
    if zdim:
        locs["z_pos_nm"] = np.linspace(-300, 300, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=zdim)


def _blend_func(widget, dataset):
    visual = widget.data_to_layer_itf.renderer.layer(dataset.dataset_id)._visual
    return visual._vshare.gl_state.get("blend_func")


def _widget(make_napari_viewer, zdim=False):
    viewer = make_napari_viewer()
    widget = napari_storm(
        napari_viewer=viewer, renderer=NapariParticlesRenderer(viewer)
    )
    widget.get_dataset_from_test_mode([_dataset(zdim=zdim)])
    return widget, widget.localization_datasets[0], viewer


def test_blending_is_additive_when_the_layer_is_created(make_napari_viewer):
    widget, dataset, _ = _widget(make_napari_viewer)
    assert _blend_func(widget, dataset) == ADDITIVE


def test_a_filter_change_does_not_break_blending(make_napari_viewer):
    """The reported case: apply a filter, and the splats stop adding."""
    widget, dataset, _ = _widget(make_napari_viewer)
    widget.render_config.range_x_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_layers()
    assert _blend_func(widget, dataset) == ADDITIVE


def test_switching_gaussian_mode_does_not_break_blending(make_napari_viewer):
    """The other reported case."""
    widget, dataset, _ = _widget(make_napari_viewer)
    widget.Brenderoptions.setCurrentIndex(1)
    assert _blend_func(widget, dataset) == ADDITIVE
    widget.Brenderoptions.setCurrentIndex(0)
    assert _blend_func(widget, dataset) == ADDITIVE


@pytest.mark.parametrize(
    "gesture",
    [
        "sigma",
        "opacity",
        "hide_show",
        "z_colour",
        "another_layer",
        "another_layer_hidden",
        "another_layer_shown",
        "second_dataset",
    ],
)
def test_blending_survives_every_redrawing_gesture(make_napari_viewer, gesture):
    widget, dataset, viewer = _widget(make_napari_viewer, zdim=True)

    if gesture == "sigma":
        widget.Esigma_xy.setText("40")
        widget.update_sigma()
    elif gesture == "opacity":
        widget.channel[0].Slider_opacity.setValue(50)
    elif gesture == "hide_show":
        widget.channel[0].show_channel()
        widget.channel[0].show_channel()
    elif gesture == "z_colour":
        widget.Bz_color_coding.setChecked(True)
    elif gesture == "another_layer":
        viewer.add_image(np.zeros((8, 8), dtype="f4"), name="unrelated")
    elif gesture == "another_layer_hidden":
        # napari connects *every* layer's `visible` event to `_reorder_layers`,
        # and reordering re-applies its blending preset to all of them -- so a
        # layer this plugin does not own can reset our blend by hiding itself.
        other = viewer.add_image(np.zeros((8, 8), dtype="f4"), name="unrelated")
        other.visible = False
    elif gesture == "another_layer_shown":
        other = viewer.add_image(np.zeros((8, 8), dtype="f4"), name="unrelated")
        other.visible = False
        other.visible = True
    elif gesture == "second_dataset":
        widget.open_localization_data_file_and_get_dataset  # noqa: B018
        widget.get_dataset_from_test_mode(
            [_dataset("a", zdim=True), _dataset("b", zdim=True)]
        )
        dataset = widget.localization_datasets[0]

    assert _blend_func(widget, dataset) == ADDITIVE, gesture


def test_blending_holds_while_the_render_range_is_dragged(make_napari_viewer):
    """Mid-drag, not just after release.

    Reported from a real session: the reconstruction "looks odd" for as long as
    a render-range slider is held, and comes back the moment it is let go. The
    release path calls `update_layers`, which re-asserted the blend state and so
    hid the defect from every test that only looked at the committed result.

    What breaks it is the render-range preview box turning itself visible: it is
    a layer, and napari recomputes blending for *all* layers whenever any one of
    them changes visibility.
    """
    from qtpy.QtWidgets import QApplication

    widget, dataset, viewer = _widget(make_napari_viewer)
    slider = widget.Srender_rangex

    slider._dragging = True
    slider._ensure_preview_layer()
    widget._render_range_preview.visible = True
    assert _blend_func(widget, dataset) == ADDITIVE, "preview box shown"

    slider.setValue((20, 80))
    slider._do_preview_update()
    QApplication.processEvents()
    assert _blend_func(widget, dataset) == ADDITIVE, "mid-drag"

    widget._render_range_preview.visible = False
    slider._dragging = False
    assert _blend_func(widget, dataset) == ADDITIVE, "preview box hidden"


def test_the_points_backend_is_left_to_napari(make_napari_viewer):
    """It has no custom blend state, so there is nothing for napari to undo.

    Recorded so the asymmetry is deliberate rather than an oversight: only the
    two Gaussian backends override napari's preset, because only they need
    destination-preserving addition.
    """
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer, renderer=NapariPointsRenderer(viewer))
    widget.get_dataset_from_test_mode([_dataset()])
    layer = widget.data_to_layer_itf.renderer.layer(
        widget.localization_datasets[0].dataset_id
    )
    assert layer.blending == "additive"
