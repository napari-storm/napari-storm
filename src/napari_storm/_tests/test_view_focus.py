"""The rotation centre follows the data, and focus mode fades the rest.

napari rotates a 3-D view about the camera centre, and panning never changes
its depth, so it drifts in front of or behind the structure being looked at.
Starting a rotation now moves it along the line of sight onto the data, and a
double-click makes a structure the centre. Focus mode fades out what lies
further than a radius from that centre, in the shader.
"""

import h5py
import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm._tests.test_summed_contrast import NM_PER_PX, _render, _request
from napari_storm.core import (
    FocusState,
    focus_weight,
    load_scene,
    pivot_on_view_axis,
    point_under_ray,
)
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer

# ----------------------------------------------------------------- geometry


def test_the_pivot_moves_along_the_line_of_sight_onto_the_data():
    on_axis = np.array([[1000.0, 0.0, 0.0], [1010.0, 0.0, 0.0], [990.0, 5.0, 0.0]])
    off_axis = np.array([[-4000.0, 5000.0, 0.0]])  # far to the side: ignored

    moved = pivot_on_view_axis(
        np.vstack([on_axis, off_axis]), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 100.0
    )

    assert moved == pytest.approx((1000.0, 0.0, 0.0))


def test_no_data_near_the_line_of_sight_leaves_the_pivot_alone():
    points = np.array([[0.0, 5000.0, 0.0]])

    assert pivot_on_view_axis(points, (0, 0, 0), (1, 0, 0), 100.0) is None


def test_a_click_finds_the_structure_under_it_at_its_depth():
    structure = np.array(
        [[300.0, 50.0, 20.0], [310.0, 52.0, 18.0], [290.0, 48.0, 22.0]]
    )
    elsewhere = np.array([[0.0, 3000.0, 0.0]])

    found = point_under_ray(
        np.vstack([structure, elsewhere]), (0.0, 50.0, 20.0), (1.0, 0.0, 0.0), 30.0
    )

    assert found == pytest.approx((300.0, 50.0, 20.0))


def test_the_fade_eases_from_full_to_the_outside_opacity():
    focus = FocusState(
        center_world=(0.0, 0.0, 0.0), radius_nm=100.0, outside_opacity=0.2
    )
    points = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 112.5], [0.0, 0.0, 500.0]])

    weights = focus_weight(points, focus)

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.6)  # half way across the edge
    assert weights[2] == pytest.approx(0.2)


def test_a_planar_focus_ignores_depth():
    focus = FocusState(center_world=(0.0, 0.0, 0.0), radius_nm=100.0, planar=True)

    assert focus_weight(np.array([[5000.0, 0.0, 0.0]]), focus) == pytest.approx([1.0])


# --------------------------------------------------------------- the shader


def _two_localizations(make_napari_viewer):
    """One at the focus centre, one 1.5 um away along x, both well in view."""
    viewer = make_napari_viewer()
    renderer = InstancedRenderer(viewer)
    renderer.open(1, _request([-750.0, 750.0]))
    viewer.dims.ndisplay = 2
    viewer.camera.center = (0.0, 0.0)
    viewer.camera.zoom = 0.5 / NM_PER_PX
    return viewer, renderer


def _halves(image, split):
    """The brightest pixel either side of column *split*.

    Where the two localizations land is read off the unfocused image rather
    than assumed: the render is not centred on the camera centre.
    """
    grey = image.max(axis=2)
    return int(grey[:, :split].max()), int(grey[:, split:].max())


def _split_between(image):
    """The column half way across everything lit."""
    cols = np.nonzero(image.max(axis=2) >= 3)[1]
    return int(round(0.5 * (cols.min() + cols.max())))


@pytest.mark.parametrize("outside", [0.0, 0.3])
def test_focus_mode_fades_what_lies_outside_the_radius(make_napari_viewer, outside):
    viewer, renderer = _two_localizations(make_napari_viewer)
    unfocused = _render(viewer)
    split = _split_between(unfocused)
    left, right = _halves(unfocused, split)
    assert left > 20 and left == pytest.approx(right, abs=2)

    renderer.set_focus(
        FocusState(
            center_world=(1.0, 0.0, -750.0),
            radius_nm=400.0,
            outside_opacity=outside,
            planar=True,
        )
    )
    focused_left, focused_right = _halves(_render(viewer), split)

    assert focused_left == pytest.approx(left, abs=2)
    assert focused_right < 0.6 * right
    if outside == 0.0:
        assert focused_right <= 2

    renderer.set_focus(None)
    assert _halves(_render(viewer), split) == pytest.approx((left, right), abs=2)


# --------------------------------------------------------------- the widget

STRUCTURE = (500.0, 40_000.0, 10_000.0)  # (z, y, x)


def _write_3d(path):
    """A tight 3-D cluster, and another 8 um to the side."""
    rng = np.random.default_rng(1)
    z, y, x = STRUCTURE
    cluster = np.column_stack(
        [
            x + rng.uniform(-5, 5, 40),
            y + rng.uniform(-5, 5, 40),
            z + rng.uniform(-5, 5, 40),
        ]
    )
    beside = cluster + np.array([8_000.0, 0.0, -1_500.0])
    xyz = np.vstack([cluster, beside])
    locs = np.rec.array(
        np.zeros(
            len(xyz), dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")]
        )
    )
    locs.x_pos_nm, locs.y_pos_nm, locs.z_pos_nm = xyz.T
    with h5py.File(path, "w") as file:
        stored = file.create_dataset("dataset", data=locs)
        stored.attrs["name"] = "cells3d"
        stored.attrs["zdim_present"] = True
        stored.attrs["dataset_class"] = "LocalizationDataBaseClass"
    return str(path)


@pytest.fixture
def widget(make_napari_viewer, tmp_path):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.reported = []
    widget._warn_user = widget.reported.append
    assert widget.open_localization_data_file_and_get_dataset(
        file_path=_write_3d(tmp_path / "cells3d.ns")
    )
    widget.viewer.dims.ndisplay = 3
    widget.viewer.camera.angles = (10.0, 30.0, 70.0)
    widget.viewer.camera.zoom = 0.2  # px per nm: the cluster beside is far off-axis
    return widget


def test_starting_to_rotate_puts_the_centre_on_the_data_in_the_middle(widget):
    camera = widget.viewer.camera
    direction = np.asarray(camera.view_direction)
    camera.center = tuple(np.asarray(STRUCTURE) - 2_000.0 * direction)

    drag = widget.view_focus._on_drag(widget.viewer, _Event())
    next(drag)  # the press

    assert tuple(camera.center) == pytest.approx(STRUCTURE, abs=10.0)


def test_rotating_with_the_option_off_leaves_the_centre(widget):
    widget.Cauto_pivot.setChecked(False)
    camera = widget.viewer.camera
    away = tuple(np.asarray(STRUCTURE) - 2_000.0 * np.asarray(camera.view_direction))
    camera.center = away

    next(widget.view_focus._on_drag(widget.viewer, _Event()))

    assert tuple(camera.center) == pytest.approx(away)


def test_a_double_click_makes_the_structure_under_it_the_centre(widget):
    camera = widget.viewer.camera
    direction = np.asarray(camera.view_direction)
    camera.center = (0.0, 0.0, 0.0)
    clicked = np.asarray(STRUCTURE) + 1_000.0 * direction  # on its line of sight

    widget.view_focus._on_double_click(
        widget.viewer, _Event(position=tuple(clicked), view_direction=tuple(direction))
    )

    assert tuple(camera.center) == pytest.approx(STRUCTURE, abs=10.0)
    assert camera.zoom == pytest.approx(0.2)  # centred, not zoomed


def test_the_viewer_gets_its_own_double_click_back_when_the_dock_closes(widget):
    callbacks = widget.viewer.mouse_double_click_callbacks
    assert widget.view_focus._on_double_click in callbacks
    assert not any(cb.__name__ == "double_click_to_zoom" for cb in callbacks)

    widget.close_session()

    assert any(
        getattr(cb, "__name__", "") == "double_click_to_zoom" for cb in callbacks
    )
    assert widget.view_focus._on_drag not in widget.viewer.mouse_drag_callbacks


def test_focus_mode_follows_the_camera(widget, monkeypatch):
    sent = []
    monkeypatch.setattr(
        widget.data_to_layer_itf.renderer, "set_focus", sent.append, raising=False
    )
    widget.SBfocus_radius.setValue(750.0)
    widget.SBfocus_outside.setValue(20)

    widget.Cfocus.setChecked(True)
    widget.viewer.camera.center = STRUCTURE

    assert sent[-1] == FocusState(
        center_world=STRUCTURE, radius_nm=750.0, outside_opacity=0.2, planar=False
    )
    widget.Cfocus.setChecked(False)
    assert sent[-1] is None


def test_a_scene_keeps_focus_mode_and_the_rotation_choice(
    widget, make_napari_viewer, tmp_path
):
    widget.Cauto_pivot.setChecked(False)
    widget.SBfocus_radius.setValue(777.0)
    widget.SBfocus_outside.setValue(25)
    widget.Cfocus.setChecked(True)
    path = tmp_path / "focus.json"
    widget.save_scene_to(path)
    assert load_scene(path).view.focus is True

    after = napari_storm(napari_viewer=make_napari_viewer())
    after._warn_user = lambda *_: None
    after.load_scene_from(path)

    assert not after.Cauto_pivot.isChecked()
    assert after.SBfocus_radius.value() == pytest.approx(777.0)
    assert after.SBfocus_outside.value() == 25
    assert after.Cfocus.isChecked() == after.view_focus.supported


# ------------------------------------------------- the dot on the centre


def test_a_grey_dot_marks_the_centre_while_the_camera_moves(widget):
    marker = widget.view_focus.marker
    marker.hide()

    widget.viewer.camera.angles = (15.0, 25.0, 65.0)  # a rotation

    assert marker.shown
    z, y, x = widget.viewer.camera.center
    np.testing.assert_allclose(marker.node._data["a_position"][0], (x, y, z))


def test_the_dot_goes_once_the_camera_is_still(widget, qtbot):
    marker = widget.view_focus.marker
    widget.viewer.camera.zoom = 0.3
    assert marker.shown

    qtbot.waitUntil(lambda: not marker.shown, timeout=3000)


def test_the_dot_is_no_layer(widget):
    marker = widget.view_focus.marker
    widget.viewer.camera.zoom = 0.3

    assert all(layer.name != "rotation centre" for layer in widget.viewer.layers)
    assert marker.node.parent is widget.viewer.window._qt_viewer.canvas.view.scene
    widget.close_session()
    assert marker.node.parent is None


class _Event:
    """What the mouse callbacks read off a napari mouse event."""

    def __init__(self, position=None, view_direction=None, modifiers=()):
        self.button = 1
        self.modifiers = list(modifiers)
        self.position = position
        self.view_direction = view_direction
