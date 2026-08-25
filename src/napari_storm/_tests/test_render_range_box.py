"""The render-range preview box, and the two things a persistent layer changed.

The box used to be destroyed and recreated on every drag. Commit 3051a72 made it
a single long-lived layer -- "always present, only made visible if necessary" --
which removed the layer churn but also removed two side effects the old code had
been relying on without saying so:

* the colour chosen in the combo box was applied by `add_surface`, so a box that
  is never recreated never picks up a colour changed since;
* toggling its visibility makes napari recompute blending for every layer, which
  is what made the reconstruction look wrong for the duration of a drag. That one
  is covered in `test_additive_blending.py`.
"""
import numpy as np
from qtpy.QtWidgets import QApplication

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles.renderer import NapariParticlesRenderer


def _dataset(name="ds", n=500):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(10_000, 12_000, n)
    locs["y_pos_nm"] = np.linspace(10_000, 12_000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(
        napari_viewer=viewer, renderer=NapariParticlesRenderer(viewer)
    )
    widget.get_dataset_from_test_mode([_dataset()])
    return widget, viewer


def _drag(slider):
    """One drag, from press through release, without a real mouse."""
    slider._dragging = True
    slider._ensure_preview_layer()
    slider._do_preview_update()
    QApplication.processEvents()
    slider._dragging = False


def test_the_box_takes_the_colour_chosen_before_the_first_drag(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer)
    widget.Brender_range_box_color.setCurrentIndex(2)
    chosen = widget.active_render_range_box_color

    _drag(widget.Srender_rangex)

    assert widget._render_range_preview.colormap.name == chosen


def test_the_box_follows_a_colour_changed_after_it_exists(make_napari_viewer):
    """The regression: the layer outlives the drag that created it.

    Recreating the box per drag applied the current colour as a side effect.
    Keeping one layer means the choice has to be pushed onto it deliberately,
    or the box keeps whatever colour it happened to be born with.
    """
    widget, _ = _widget(make_napari_viewer)
    slider = widget.Srender_rangex

    widget.Brender_range_box_color.setCurrentIndex(1)
    _drag(slider)
    first = widget._render_range_preview.colormap.name

    widget.Brender_range_box_color.setCurrentIndex(3)
    chosen = widget.active_render_range_box_color
    assert chosen != first, "fixture needs two different colormaps"

    _drag(slider)

    assert widget._render_range_preview.colormap.name == chosen


def test_the_box_is_reused_rather_than_recreated(make_napari_viewer):
    """Colour is pushed onto the existing layer, not fixed by recreating it."""
    widget, viewer = _widget(make_napari_viewer)
    slider = widget.Srender_rangex

    _drag(slider)
    first = widget._render_range_preview

    widget.Brender_range_box_color.setCurrentIndex(3)
    _drag(slider)

    assert widget._render_range_preview is first
    assert [layer.name for layer in viewer.layers].count("render-range") == 1
