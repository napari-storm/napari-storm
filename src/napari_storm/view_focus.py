"""The rotation centre and focus mode, wired to the viewer.

napari rotates a 3-D view about ``camera.center``. Panning moves that point
across the screen but not in depth, so it drifts off the structure being
looked at and every turn swings the structure away. Here:

* **Rotating** (a plain left drag in 3-D) first moves the centre along the
  line of sight onto the localizations in the middle of the view. Only the
  depth changes, so the picture does not jump.
* **Double-clicking** a structure in 3-D makes it the centre and brings it to
  the middle of the view. napari's own double-click zooms in instead; it is
  kept for 2-D.
* **Focus mode** fades out what lies further than a radius from the centre,
  following the camera as it moves.
* **A grey dot** marks the centre while the camera moves -- rotating,
  panning, zooming -- and goes again once it has been still for a moment, as
  Blender shows its orbit centre.

The geometry is in `core.focus`; the fading is done by the renderer.
"""

from __future__ import annotations

import logging

import numpy as np

from .core import (
    DEFAULT_FOCUS_RADIUS_NM,
    DEFAULT_OUTSIDE_OPACITY,
    FocusState,
    pivot_on_view_axis,
    point_under_ray,
)

__all__ = ["ViewFocus"]

#: Localizations looked at when picking a centre: enough to find a
#: structure, few enough to answer within a mouse press.
MAX_POINTS = 400_000

#: The line of sight's reach when rotating starts, as a fraction of the
#: smaller side of the view; tried wider when nothing lies that close.
PIVOT_REACH = (0.05, 0.15, 0.5)

#: How close to a double-click a localization must be, in screen pixels.
CLICK_REACH_PX = (8.0, 25.0)

#: How long the centre's dot stays after the camera stops, in milliseconds.
MARKER_LINGER_MS = 700

_log = logging.getLogger(__name__)


class PivotMarker:
    """A grey dot on the centre of rotation, shown while the camera moves.

    A VisPy marker in the viewer's own scene rather than a napari layer: it is
    an indicator, not data, and has no place in the layer list. It is a fixed
    size on screen, drawn over everything, and hidden again once the camera
    has been still for `MARKER_LINGER_MS`.
    """

    def __init__(self, viewer):
        from qtpy.QtCore import QTimer
        from vispy.scene.visuals import Markers

        class _CentreDot(Markers):
            def _compute_bounds(self, axis, view):
                # An indicator, not data: VisPy measures every node in the
                # scene when the camera is swapped, and an empty marker has no
                # bounds to give -- nor should the dot stretch the view.
                return None

        self.viewer = viewer
        view = viewer.window._qt_viewer.canvas.view
        self.node = _CentreDot(parent=view.scene)
        self.node.set_gl_state("translucent", depth_test=False)
        self.node.order = 1_000_000  # after every layer
        self.node.visible = False
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(MARKER_LINGER_MS)
        self._timer.timeout.connect(self.hide)

    @property
    def shown(self):
        return bool(self.node.visible)

    def flash(self, event=None):
        """Show the dot where the centre is now, and restart the countdown."""
        center = np.asarray(self.viewer.camera.center, dtype=float)
        if self.viewer.dims.ndisplay == 3:
            position = center[::-1]  # VisPy's scene is (x, y, z)
        else:
            position = np.append(center[-2:][::-1], 0.0)
        self.node.set_data(
            pos=position[None, :],
            size=9.0,
            edge_width=1.0,
            face_color=(0.62, 0.62, 0.62, 0.9),
            edge_color=(1.0, 1.0, 1.0, 0.8),
        )
        self.node.visible = True
        self._timer.start()

    def hide(self):
        self.node.visible = False

    def detach(self):
        self._timer.stop()
        self.node.parent = None


class ViewFocus:
    """One dock widget's rotation centre and focus mode."""

    def __init__(self, widget):
        self.widget = widget
        self.viewer = widget.viewer
        self.auto_pivot = True
        self.focus_enabled = False
        self.radius_nm = DEFAULT_FOCUS_RADIUS_NM
        self.outside_opacity = DEFAULT_OUTSIDE_OPACITY

        self.viewer.mouse_drag_callbacks.append(self._on_drag)
        double = self.viewer.mouse_double_click_callbacks
        self._napari_double_click = next(
            (
                cb
                for cb in double
                if getattr(cb, "__name__", "") == "double_click_to_zoom"
            ),
            None,
        )
        if self._napari_double_click is not None:
            double[double.index(self._napari_double_click)] = self._on_double_click
        else:
            double.append(self._on_double_click)
        self.viewer.camera.events.center.connect(self._camera_moved)
        self.viewer.dims.events.ndisplay.connect(self._camera_moved)

        try:
            self.marker = PivotMarker(self.viewer)
        except (AttributeError, ImportError, RuntimeError) as error:
            # No canvas to draw on -- a viewer without a window.
            _log.debug("no centre-of-rotation marker: %s", error)
            self.marker = None
        else:
            for emitter in self._camera_emitters():
                emitter.connect(self.marker.flash)

    def _camera_emitters(self):
        camera = self.viewer.camera.events
        return (camera.center, camera.zoom, camera.angles)

    def detach(self):
        """Give the viewer back its own mouse bindings."""
        if self.marker is not None:
            for emitter in self._camera_emitters():
                try:
                    emitter.disconnect(self.marker.flash)
                except (ValueError, TypeError, RuntimeError):
                    pass
            self.marker.detach()
            self.marker = None
        for callbacks, ours in (
            (self.viewer.mouse_drag_callbacks, self._on_drag),
            (self.viewer.mouse_double_click_callbacks, self._on_double_click),
        ):
            if ours in callbacks:
                index = callbacks.index(ours)
                if ours == self._on_double_click and self._napari_double_click:
                    callbacks[index] = self._napari_double_click
                else:
                    callbacks.pop(index)
        for emitter in (
            self.viewer.camera.events.center,
            self.viewer.dims.events.ndisplay,
        ):
            try:
                emitter.disconnect(self._camera_moved)
            except (ValueError, TypeError, RuntimeError):
                pass

    # ------------------------------------------------------------------
    # The localizations on screen
    # ------------------------------------------------------------------

    def drawn_points(self, limit=MAX_POINTS):
        """World ``(z, y, x)`` of the drawn localizations of shown datasets."""
        widget = self.widget
        itf = widget.data_to_layer_itf
        datasets = []
        for dataset in widget.localization_datasets:
            channel = next(
                (c for c in widget.channel if getattr(c, "dataset", None) is dataset),
                None,
            )
            if channel is not None and not getattr(channel, "show_channel_state", True):
                continue
            datasets.append(dataset)
        total = sum(d.number_of_active_entries() for d in datasets)
        if not total:
            return np.empty((0, 3))
        step = max(1, int(np.ceil(total / limit)))
        parts = []
        for dataset in datasets:
            n = dataset.number_of_active_entries()
            if not n:
                continue
            transform = itf.transform_of(dataset)
            columns = []
            for axis in ("z", "y", "x"):
                if axis == "z" and not dataset.zdim_present:
                    columns.append(np.ones(len(range(0, n, step))))
                    continue
                values = np.asarray(dataset.table.active_coordinate_nm(axis))[::step]
                columns.append(transform.apply_axis(axis, values))
            parts.append(np.column_stack(columns))
        points = np.concatenate(parts) if parts else np.empty((0, 3))
        return points[np.isfinite(points).all(axis=1)]

    def _world_per_pixel(self):
        zoom = float(self.viewer.camera.zoom)
        return 1.0 / zoom if zoom > 0 else 1.0

    def _view_size_px(self):
        size = getattr(self.viewer, "_canvas_size", None)
        if size is None or min(size) <= 0:
            return 600.0
        return float(min(size))

    # ------------------------------------------------------------------
    # Rotation centre
    # ------------------------------------------------------------------

    def pivot_to_view_middle(self):
        """Put the rotation centre on the data in the middle of the view.

        Returns True when it moved. Nothing moves when no localization lies
        anywhere near the line of sight.
        """
        points = self.drawn_points()
        if not len(points):
            return False
        camera = self.viewer.camera
        center = np.asarray(camera.center, dtype=float)
        direction = np.asarray(camera.view_direction, dtype=float)
        for fraction in PIVOT_REACH:
            reach = fraction * self._view_size_px() * self._world_per_pixel()
            moved = pivot_on_view_axis(points, center, direction, reach)
            if moved is not None:
                camera.center = moved
                return True
        return False

    def centre_on(self, position, view_direction):
        """Make the structure under a click the centre, in the middle of the view.

        Returns True when one was found.
        """
        points = self.drawn_points()
        if not len(points):
            return False
        origin = np.asarray(position, dtype=float)[-3:]
        direction = np.asarray(view_direction, dtype=float)[-3:]
        for reach_px in CLICK_REACH_PX:
            target = point_under_ray(
                points, origin, direction, reach_px * self._world_per_pixel()
            )
            if target is not None:
                self.viewer.camera.center = target
                return True
        return False

    def _on_drag(self, viewer, event):
        """A plain left drag in 3-D rotates: centre it on the data first."""
        if (
            self.auto_pivot
            and viewer.dims.ndisplay == 3
            and getattr(event, "button", 1) == 1
            and not set(getattr(event, "modifiers", ())) & {"Shift", "Control", "Alt"}
            and _panning(viewer)
        ):
            self.pivot_to_view_middle()
        yield

    def _on_double_click(self, viewer, event):
        if viewer.dims.ndisplay == 3 and _panning(viewer):
            self.centre_on(event.position, event.view_direction)
        elif self._napari_double_click is not None:
            self._napari_double_click(viewer, event)

    # ------------------------------------------------------------------
    # Focus mode
    # ------------------------------------------------------------------

    @property
    def supported(self):
        renderer = getattr(self.widget.data_to_layer_itf, "renderer", None)
        return bool(getattr(renderer, "supports_focus", False))

    def set_focus(self, enabled=None, radius_nm=None, outside_opacity=None):
        """Change focus mode; anything left None keeps its value."""
        if enabled is not None:
            self.focus_enabled = bool(enabled)
        if radius_nm is not None:
            self.radius_nm = float(radius_nm)
        if outside_opacity is not None:
            self.outside_opacity = float(outside_opacity)
        self.push()

    def state(self):
        """The focus region now, or None while focus mode is off."""
        if not self.focus_enabled:
            return None
        return FocusState(
            center_world=tuple(float(v) for v in self.viewer.camera.center),
            radius_nm=self.radius_nm,
            outside_opacity=self.outside_opacity,
            planar=self.viewer.dims.ndisplay == 2,
        )

    def push(self):
        renderer = getattr(self.widget.data_to_layer_itf, "renderer", None)
        if renderer is not None and hasattr(renderer, "set_focus"):
            renderer.set_focus(self.state())

    def _camera_moved(self, event=None):
        if self.focus_enabled:
            self.push()


def _panning(viewer):
    """Whether a drag means moving the camera, not editing a layer."""
    layer = viewer.layers.selection.active
    return layer is None or getattr(layer, "mode", "pan_zoom") == "pan_zoom"
