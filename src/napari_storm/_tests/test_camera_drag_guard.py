"""Releasing Shift mid-drag must not crash the 3-D camera.

Reported from a real session while changing the camera over 3-D data with a
reference image loaded. It is a VisPy bug, not one of ours, but it is reachable
from an ordinary gesture and it takes the viewer down, so the plugin guards it.

VisPy keeps one `_event_value` per drag and clears it only on mouse *release*.
Shift+drag translates and stores `self.center` -- three world coordinates --
while a plain drag rotates and expects two *screen* coordinates. Let go of Shift
without letting go of the mouse and the rotate branch reads the translate
branch's three-tuple.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.napari_particles._napari_compat import guard_camera_drag_state


class _MouseEvent:
    """vispy reads `event.mouse_event.press_event.pos` as the drag origin."""

    def __init__(self, modifiers=(), press=True):
        self.modifiers = modifiers
        self.pos = np.array([120, 90])
        self.press_event = _MouseEvent(modifiers, press=False) if press else None


class _Event:
    """The parts of a SceneMouseEvent the camera actually reads."""

    def __init__(self, buttons=(1,), modifiers=()):
        self.type = "mouse_move"
        self.buttons = list(buttons)
        self.handled = False
        self.pos = np.array([120, 90])
        self.mouse_event = _MouseEvent(modifiers)
        self.press_event = _MouseEvent(modifiers)


def _camera(viewer):
    return viewer.window._qt_viewer.canvas.view.camera


def test_a_translate_then_rotate_does_not_crash(make_napari_viewer):
    """The reported gesture: pan with Shift, release Shift, keep dragging."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    napari_storm(napari_viewer=viewer)
    camera = _camera(viewer)

    # What Shift+drag leaves behind: the camera centre, three coordinates.
    camera._event_value = camera.center
    assert len(camera._event_value) == 3

    # Shift released, mouse still down: the rotate branch.
    camera.viewbox_mouse_event(_Event(buttons=(1,), modifiers=()))

    # No exception, and the stale value has been replaced by a position.
    assert camera._event_value is None or len(camera._event_value) == 2


def test_the_guard_leaves_a_genuine_translate_alone(make_napari_viewer):
    """Shift still held means the three-tuple is correct and must survive."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    napari_storm(napari_viewer=viewer)
    camera = _camera(viewer)
    from vispy.util import keys

    camera._event_value = camera.center
    camera.viewbox_mouse_event(_Event(buttons=(1,), modifiers=(keys.SHIFT,)))

    assert len(camera._event_value) == 3


def test_the_guard_is_idempotent(make_napari_viewer):
    """Applied on construction and again on every ndisplay change."""
    viewer = make_napari_viewer()
    napari_storm(napari_viewer=viewer)
    camera = _camera(viewer)
    wrapped = camera.viewbox_mouse_event

    guard_camera_drag_state(viewer)

    assert camera.viewbox_mouse_event is wrapped


def test_it_survives_an_ndisplay_change(make_napari_viewer):
    """napari swaps cameras between 2-D and 3-D; the guard has to follow."""
    viewer = make_napari_viewer()
    napari_storm(napari_viewer=viewer)

    viewer.dims.ndisplay = 3
    camera = _camera(viewer)
    camera._event_value = getattr(camera, "center", (0.0, 0.0, 0.0))
    if len(camera._event_value) != 3:
        pytest.skip("this camera does not keep a three-tuple centre")

    camera.viewbox_mouse_event(_Event(buttons=(1,), modifiers=()))

    assert camera._event_value is None or len(camera._event_value) == 2


def test_without_the_guard_the_old_vispy_really_does_crash(
    make_napari_viewer, monkeypatch
):
    """Proof the guard is load-bearing, on the VisPy the report came from.

    Newer VisPy slices `_event_value[:2]` and so cannot raise; the release the
    report came from passes the value whole. That older
    `_update_rotation` is restored here so the reported traceback is reproduced
    exactly, rather than asserted about from a distance.
    """
    from vispy.scene.cameras.arcball import ArcballCamera, _arcball
    from vispy.util.quaternion import Quaternion

    import napari_storm._dock_widget as dock_widget

    monkeypatch.setattr(dock_widget, "guard_camera_drag_state", lambda viewer: None)

    def old_update_rotation(self, event):
        position = event.pos[:2]
        if self._event_value is None:
            self._event_value = position
        size = self._viewbox.size
        self._quaternion = (
            Quaternion(*_arcball(position, size))
            * Quaternion(*_arcball(self._event_value, size))
            * self._quaternion
        )
        self._event_value = position

    monkeypatch.setattr(ArcballCamera, "_update_rotation", old_update_rotation)

    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    napari_storm(napari_viewer=viewer)
    camera = _camera(viewer)
    camera._event_value = camera.center

    with pytest.raises(ValueError, match="too many values"):
        camera.viewbox_mouse_event(_Event(buttons=(1,), modifiers=()))
