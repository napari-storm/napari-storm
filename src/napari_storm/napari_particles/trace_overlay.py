"""Each dataset's traces, drawn as a napari Tracks layer over its localizations.

What to connect, and in which order, is decided host-free in `core.traces`;
this is the part that owns a host layer.  Every napari backend composes one
:class:`TraceOverlay`, so the three draw traces identically.

**Why a Tracks layer.**  It takes exactly ``ID, T, (Z), Y, X`` and draws
connected polylines with per-vertex colour, in 2-D and 3-D, at ~10 ms a frame
for a million vertices, and napari's own layer list, controls and status bar
know what it is.  The alternatives were measured against it: a Shapes layer of
paths builds a mesh per path in Python (0.17 s for 10^5 vertices, so tens of
seconds at the 10^6 a long MINFLUX run reaches) and colours whole shapes, not
vertices, so it cannot colour by time; a line visual of our own would have to
re-implement the clipping, picking and layer integration napari already has.

**What it costs, and how that is paid.**  A Tracks layer is time-resolved: its
data carries a T column and napari adds a time axis to the viewer for it.  Two
things follow, and both are handled here rather than left to the caller:

* *A slider would hide vertices.*  napari draws a track only up to the current
  time point, and fades it before that, so with real timestamps a fresh layer
  shows nothing until the slider is moved to the end -- which is what the
  pyMINFLUX prototype of this did.  Here every vertex sits at t = 0, so the
  time axis has a single step, napari shows no slider for it, and nothing is
  hidden or faded.  Time is shown by colour instead (``trace_color_by="time"``),
  and the order within each trace by the line itself.  The tail is made long so
  that a host's own time axis, should it add one, does not fade the overlay as
  it moves forward.
* *A new axis refits the camera.*  When the viewer's dimensionality changes
  napari re-orders its dims and calls ``fit_to_view``, which re-zooms and
  re-centres.  Adding or removing an overlay must not move the view a user has
  set up, so the camera is put back after either.

The layer's own ids are the trace column's ids whenever they are small enough
to be safe -- napari indexes tracks by id and allocates for every id up to the
largest -- so hovering a trace reports the id the file uses.  Larger ids are
renumbered densely.

Colours are computed here, from the vertices' own values, and handed to the
layer as ``track_colors``; the layer's ``color_by`` stays ``track_id``, which
always exists, so napari never falls back with a warning when the data changes.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ..core.renderer import Changed
from ..core.traces import (
    COLOR_BY_TRACE,
    DEFAULT_TRACE_WIDTH_PX,
    validate_trace_color_by,
    validate_trace_width_px,
)

__all__ = ["TraceOverlay", "TRACE_LAYER_SUFFIX", "tracks_data"]

#: Appended to the dataset's layer name to name its overlay.
TRACE_LAYER_SUFFIX = " traces"

#: The one time point every vertex sits at.  See the module docstring.
TRACE_TIME = 0.0

#: Long enough that a host's own time axis, moved forward, leaves the overlay
#: drawn: napari fades a vertex by (current time - its time) / tail length, so
#: a thousand steps forward costs a thousandth of its alpha.
TAIL_LENGTH = 10**6

#: The largest trace id handed to napari as it is.  napari builds a sparse
#: lookup with one row per id up to the largest, so this caps that table at a
#: few megabytes; MINFLUX ids are well below it.
MAX_LAYER_TRACE_ID = 2**22

#: Colormaps, by what is coloured.  The per-trace hue is cyclic, because the
#: values it gets are spread around [0, 1) and the two ends must not meet as
#: different colours; anything ordered gets a perceptually ordered map.
TRACE_COLORMAP = "hsv"
ORDERED_COLORMAP = "turbo"

#: Changes that move or re-select vertices, and so redraw the overlay.
_TRACE_CHANGES = Changed.SELECTION | Changed.POSITIONS | Changed.TRACES


@dataclass(frozen=True)
class _Style:
    traces: bool = False
    color_by: str = COLOR_BY_TRACE
    width_px: float = DEFAULT_TRACE_WIDTH_PX


def tracks_data(vertices):
    """*vertices* as napari Tracks data, ``(N, 5)`` of ``ID, T, Z, Y, X``.

    Always five columns: flat data keeps its z column, pinned to the plane its
    splats are drawn on, so that the overlay's trailing axes line up with the
    localization layers' ``(z, y, x)`` rather than its time axis landing on z.
    """
    data = np.empty((vertices.n_vertices, 5), dtype=np.float64)
    data[:, 0] = _layer_ids(vertices)
    data[:, 1] = TRACE_TIME
    data[:, 2:] = vertices.coords
    return data


def _layer_ids(vertices):
    ids = vertices.trace_ids
    if len(ids) and ids.min() >= 0 and ids.max() <= MAX_LAYER_TRACE_ID:
        return vertices.vertex_trace_ids
    return vertices.trace_index


def _colours(vertices, color_by):
    """RGBA per vertex for *color_by*, through napari's own colormaps."""
    from napari.utils.colormaps import AVAILABLE_COLORMAPS

    values = vertices.color_values(color_by)
    if color_by == COLOR_BY_TRACE:
        colormap = AVAILABLE_COLORMAPS[TRACE_COLORMAP]
    else:
        colormap = AVAILABLE_COLORMAPS[ORDERED_COLORMAP]
        finite = np.isfinite(values)
        low = float(values[finite].min()) if finite.any() else 0.0
        high = float(values[finite].max()) if finite.any() else 1.0
        span = high - low if high > low else 1.0
        values = np.clip(np.where(finite, (values - low) / span, 0.0), 0.0, 1.0)
    return np.asarray(colormap.map(values), dtype=np.float32)


@contextmanager
def _view_kept(viewer):
    """Undo napari's refit, and keep the user's layer selection.

    Adding or removing the only time-resolved layer changes the viewer's
    dimensionality; napari then re-orders its dims and fits the view to the
    data, and selects the layer it just added.  Neither is what a user who
    ticked a checkbox asked for.
    """
    camera = viewer.camera
    center, zoom, angles = tuple(camera.center), camera.zoom, tuple(camera.angles)
    selection = viewer.layers.selection
    selected = set(selection)
    try:
        yield
    finally:
        camera.center = center
        camera.zoom = zoom
        camera.angles = angles
        survivors = {layer for layer in selected if layer in viewer.layers}
        if set(selection) != survivors:
            selection.clear()
            selection.update(survivors)


class TraceOverlay:
    """One Tracks layer per dataset whose appearance asks for traces.

    Holds, per dataset, the trace vertices from the latest request, the trace
    style from the appearance, and whether the dataset itself is visible and
    how opaque -- the overlay follows both.  A layer exists only while there is
    something to draw: turning traces off, or a filter that leaves no trace
    with two localizations, removes it, so the viewer has no time axis it does
    not need.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self._layers = {}
        self._vertices = {}
        self._styles = {}
        self._names = {}
        self._shown = {}
        self._opacity = {}
        self._closing = set()
        self._warned = set()
        #: callable(dataset_id) -- the user deleted a trace layer in napari.
        #: The overlay is then off for that dataset until asked for again.
        self.on_traces_removed_by_host = None
        viewer.layers.events.removed.connect(self._on_layer_removed)

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------

    def layer(self, dataset_id):
        """The Tracks layer drawing *dataset_id*'s traces, or None."""
        return self._layers.get(dataset_id)

    @property
    def layers(self):
        return list(self._layers.values())

    def draws(self, dataset_id):
        """Whether *dataset_id*'s traces are on screen now."""
        layer = self._layers.get(dataset_id)
        return layer is not None and bool(layer.visible) and layer.opacity > 0

    def appearance_fields(self, dataset_id):
        """The trace fields of `LayerAppearance`, as this overlay holds them."""
        style = self._styles.get(dataset_id, _Style())
        return {
            "traces": style.traces,
            "trace_color_by": style.color_by,
            "trace_width_px": style.width_px,
        }

    def host_bytes(self, dataset_id):
        """The vertices held for *dataset_id* and the arrays handed to napari.

        napari's own indexes over them -- a sorted copy and a k-d tree for
        hovering -- are not counted; they are about the same again.
        """
        total = 0
        vertices = self._vertices.get(dataset_id)
        if vertices is not None:
            for array in (
                vertices.coords,
                vertices.trace_index,
                vertices.time,
                vertices.localization_ids,
            ):
                total += array.nbytes
        layer = self._layers.get(dataset_id)
        if layer is not None:
            total += layer.data.nbytes
            colours = layer.track_colors
            if colours is not None:
                total += colours.nbytes
        return total

    # ------------------------------------------------------------------
    # Following the backend
    # ------------------------------------------------------------------

    def open(self, dataset_id, request):
        """A dataset was opened: hold its vertices, draw nothing yet."""
        self.close(dataset_id)
        self._vertices[dataset_id] = request.traces
        self._styles[dataset_id] = _Style()
        self._names[dataset_id] = request.name
        self._shown[dataset_id] = True
        self._opacity[dataset_id] = 1.0

    def update(self, dataset_id, request):
        """A new request: redraw if it says the traces changed."""
        if dataset_id not in self._styles:
            return
        self._names[dataset_id] = request.name
        if request.changed & _TRACE_CHANGES:
            self._vertices[dataset_id] = request.traces
            self._redraw(dataset_id, data=True)

    def set_shown(self, dataset_id, shown):
        """The dataset was shown or hidden; the overlay goes with it."""
        if dataset_id not in self._styles:
            return
        self._shown[dataset_id] = bool(shown)
        self._apply_visibility(dataset_id)

    def set_appearance(self, dataset_id, appearance):
        """Apply the trace fields of *appearance*, and follow its opacity."""
        if dataset_id not in self._styles:
            return
        if appearance.opacity is not None:
            self._opacity[dataset_id] = float(appearance.opacity)
        if appearance.visible is not None:
            self._shown[dataset_id] = bool(appearance.visible)
        style = self._styles[dataset_id]
        changes = {}
        if appearance.traces is not None:
            changes["traces"] = bool(appearance.traces)
        if appearance.trace_color_by is not None:
            changes["color_by"] = validate_trace_color_by(appearance.trace_color_by)
        if appearance.trace_width_px is not None:
            changes["width_px"] = validate_trace_width_px(appearance.trace_width_px)
        new_style = replace(style, **changes)
        self._styles[dataset_id] = new_style
        if new_style.traces != style.traces:
            self._redraw(dataset_id, data=True)
            return
        layer = self._layers.get(dataset_id)
        if layer is None:
            return
        if new_style.color_by != style.color_by:
            layer.track_colors = self._colours_for(dataset_id)
        if new_style.width_px != style.width_px:
            layer.tail_width = new_style.width_px
        self._apply_visibility(dataset_id)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _wanted(self, dataset_id):
        vertices = self._vertices.get(dataset_id)
        style = self._styles.get(dataset_id)
        return (
            style is not None
            and style.traces
            and vertices is not None
            and vertices.n_segments > 0
        )

    def _colours_for(self, dataset_id):
        vertices = self._vertices[dataset_id]
        color_by = self._styles[dataset_id].color_by
        if color_by not in vertices.color_by_options:
            # Say so once: a host that asks for a column it did not plan gets
            # one hue per trace rather than an exception from deep in a redraw.
            key = (dataset_id, color_by)
            if key not in self._warned:
                self._warned.add(key)
                warnings.warn(
                    f"traces cannot be coloured by {color_by!r}, which was not "
                    f"planned; colouring by trace. Plan with "
                    f"trace_properties=({color_by!r},).",
                    stacklevel=3,
                )
            color_by = COLOR_BY_TRACE
        return _colours(vertices, color_by)

    def _redraw(self, dataset_id, data):
        if not self._wanted(dataset_id):
            self._remove_layer(dataset_id)
            return
        vertices = self._vertices[dataset_id]
        layer = self._layers.get(dataset_id)
        if layer is None:
            self._add_layer(dataset_id, vertices)
            return
        if data:
            layer.data = tracks_data(vertices)
            layer.track_colors = self._colours_for(dataset_id)
        layer.name = self._layer_name(dataset_id)
        self._apply_visibility(dataset_id)

    def _layer_name(self, dataset_id):
        return f"{self._names.get(dataset_id) or 'dataset'}{TRACE_LAYER_SUFFIX}"

    def _add_layer(self, dataset_id, vertices):
        from napari.layers import Tracks

        style = self._styles[dataset_id]
        data = tracks_data(vertices)
        # Built from two vertices and then given the rest: napari builds its
        # track index -- a sort, a k-d tree, a validation loop in Python --
        # once in the constructor and again when the data is set, and at 10^6
        # vertices each build is 0.4 s.  The first two vertices lie inside the
        # data's extent, so nothing in between sees a different scene.
        layer = Tracks(
            data[:2],
            name=self._layer_name(dataset_id),
            tail_width=style.width_px,
            tail_length=TAIL_LENGTH,
            head_length=0,
            # Depth-tested and alpha-blended: a line behind an opaque marker
            # is hidden by it, and a line over a bright Gaussian keeps its own
            # colour rather than adding to the glow until it saturates.
            blending="translucent",
            opacity=self._opacity.get(dataset_id, 1.0),
            visible=self._shown.get(dataset_id, True),
        )
        layer.data = data
        layer.track_colors = self._colours_for(dataset_id)
        with _view_kept(self.viewer):
            self.viewer.add_layer(layer)
        self._layers[dataset_id] = layer
        self._apply_visibility(dataset_id)

    def _remove_layer(self, dataset_id):
        layer = self._layers.pop(dataset_id, None)
        if layer is None:
            return
        self._closing.add(dataset_id)
        try:
            if layer in self.viewer.layers:
                with _view_kept(self.viewer):
                    self.viewer.layers.remove(layer)
        finally:
            self._closing.discard(dataset_id)

    def _apply_visibility(self, dataset_id):
        layer = self._layers.get(dataset_id)
        if layer is None:
            return
        opacity = self._opacity.get(dataset_id, 1.0)
        if layer.opacity != opacity:
            layer.opacity = opacity
        visible = self._shown.get(dataset_id, True)
        if layer.visible != visible:
            layer.visible = visible

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_layer_removed(self, event):
        """A trace layer left the viewer behind our back: traces off for it."""
        layer = getattr(event, "value", None)
        dataset_id = next(
            (key for key, known in self._layers.items() if known is layer), None
        )
        if dataset_id is None or dataset_id in self._closing:
            return
        self._layers.pop(dataset_id, None)
        style = self._styles.get(dataset_id)
        if style is not None:
            # Otherwise the next filter change would put it straight back.
            self._styles[dataset_id] = replace(style, traces=False)
        if self.on_traces_removed_by_host is not None:
            self.on_traces_removed_by_host(dataset_id)

    def close(self, dataset_id):
        """Remove *dataset_id*'s overlay and forget everything held for it."""
        self._remove_layer(dataset_id)
        for store in (
            self._vertices,
            self._styles,
            self._names,
            self._shown,
            self._opacity,
        ):
            store.pop(dataset_id, None)
        self._warned = {key for key in self._warned if key[0] != dataset_id}

    def close_all(self):
        for dataset_id in list(self._styles):
            self.close(dataset_id)

    def detach(self):
        """Stop listening to the viewer.  Call before dropping the backend."""
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layer_removed)
        except (ValueError, TypeError, RuntimeError):
            pass


class TracesMixin:
    """The trace half of a napari backend's public surface.

    A backend that composes a :class:`TraceOverlay` as ``self._traces`` gets
    `draws_traces`, `trace_layer` and the removal callback from here, and
    forwards its own lifecycle calls to the overlay.
    """

    _traces: TraceOverlay

    def draws_traces(self, dataset_id):
        return self._traces.draws(dataset_id)

    def trace_layer(self, dataset_id):
        """The napari Tracks layer drawing *dataset_id*'s traces, or None.

        Backend-specific, like `layer`: for tests and inspection, not for the
        application, which goes through `LayerAppearance`.
        """
        return self._traces.layer(dataset_id)

    @property
    def on_traces_removed_by_host(self):
        """callable(dataset_id): the user deleted a trace layer in napari."""
        return self._traces.on_traces_removed_by_host

    @on_traces_removed_by_host.setter
    def on_traces_removed_by_host(self, callback):
        self._traces.on_traces_removed_by_host = callback
