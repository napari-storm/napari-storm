"""Single place where napari-storm touches napari's private Qt internals.

The billboard renderer needs the VisPy node behind a layer, which napari does
not expose publicly.  Rather than scatter ``viewer.window.qt_viewer...`` chains
through the codebase, every private lookup lives here so that:

* there is exactly one file to update when napari moves something, and
* each lookup has a documented fallback instead of raising deep inside a
  layer-construction call stack.

Verified against napari 0.4.x - 0.7.x.  ``Window.qt_viewer`` emits a
FutureWarning from 0.6 onward and is slated for removal in 0.8, so
``Window._qt_viewer`` is preferred when present.
"""

import warnings
import weakref

__all__ = [
    "NapariInternalsChanged",
    "ADDITIVE_BLEND_STATE",
    "get_qt_viewer",
    "get_layer_visual",
    "get_layer_controls",
    "set_builtin_layer_docks_visible",
    "guard_camera_drag_state",
    "force_additive_blending",
    "release_additive_blending",
]

#: True additive blending: add the source, keep the destination.  Depth testing
#: is off because splats are order-independent once they only ever sum.
ADDITIVE_BLEND_STATE = {
    "blend": True,
    "depth_test": False,
    "blend_func": ("src_alpha", "one"),
}

#: Visual -> its own unwrapped ``set_gl_state``, for visuals we have forced.
_FORCED_BLENDING = weakref.WeakKeyDictionary()


class NapariInternalsChanged(RuntimeError):
    """Raised when a required napari internal is missing on this version."""


def get_qt_viewer(viewer):
    """Return the QtViewer for *viewer*, preferring the non-deprecated path."""
    window = getattr(viewer, "window", None)
    if window is None:
        raise NapariInternalsChanged(
            "viewer has no .window; a napari Viewer is required"
        )
    qt_viewer = getattr(window, "_qt_viewer", None)
    if qt_viewer is not None:
        return qt_viewer
    with warnings.catch_warnings():
        # Public access is deprecated in >=0.6; we already tried the private
        # path above, so the warning carries no information for the user.
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = getattr(window, "qt_viewer", None)
    if qt_viewer is None:
        raise NapariInternalsChanged(
            "cannot reach the QtViewer via Window._qt_viewer or Window.qt_viewer"
        )
    return qt_viewer


def get_layer_visual(viewer, layer):
    """Return the VisPy node rendering *layer*.

    Raises
    ------
    NapariInternalsChanged
        If napari no longer exposes a layer-to-visual mapping.  The renderer
        genuinely cannot work without this, so it is a hard error.
    """
    qt_viewer = get_qt_viewer(viewer)
    mapping = getattr(qt_viewer, "layer_to_visual", None)
    if mapping is None:
        # napari 0.6+ also carries the mapping on the canvas.
        canvas = getattr(qt_viewer, "canvas", None)
        mapping = getattr(canvas, "layer_to_visual", None)
    if mapping is None:
        raise NapariInternalsChanged(
            "no layer_to_visual mapping on QtViewer or its canvas"
        )
    try:
        entry = mapping[layer]
    except KeyError as exc:
        raise NapariInternalsChanged(
            "layer is not present in layer_to_visual; was it added to the viewer?"
        ) from exc
    node = getattr(entry, "node", None)
    return entry if node is None else node


def force_additive_blending(visual):
    """Make *visual*'s blend function non-negotiable.

    napari recomputes the GL state of **every** layer whenever layer order is
    recomputed, and `VispyCanvas` recomputes order on insert, remove, reorder,
    and on *any* layer's visibility change -- it connects
    ``napari_layer.events.visible`` to ``_reorder_layers`` for every layer it
    holds.  For the bottom-most visible layer `_on_blending_change` then picks
    ``blend_func=('src_alpha', 'zero', 'one', 'one')``, discarding the
    destination so the canvas background cannot bleed through.  For a
    reconstruction that is exactly backwards: each splat replaces its
    neighbours instead of summing with them, and the image stops accumulating.

    Re-asserting the state from event handlers was the first attempt, and it is
    the wrong shape: it means enumerating every trigger -- the render-range
    preview box turning itself visible mid-drag was one that got missed -- and
    winning an ordering race against napari's own handler on each of them.
    napari funnels all of it through one call, ``node.set_gl_state``, so
    wrapping that settles the whole class of bug: whatever napari asks for on
    this visual, the blend function it gets back is additive.
    """
    if visual in _FORCED_BLENDING:
        visual.set_gl_state(**ADDITIVE_BLEND_STATE)
        return visual

    original = visual.set_gl_state

    def set_gl_state(*args, **kwargs):
        # napari's other choices (cull_face, blend_equation) are kept; only the
        # blending itself is overridden.
        kwargs.update(ADDITIVE_BLEND_STATE)
        return original(*args, **kwargs)

    # VisPy freezes its visuals, so the original cannot be parked on the object
    # itself -- `set_gl_state` can be rebound only because the name already
    # exists on the class.  A weak-keyed registry keeps the bookkeeping off the
    # visual without pinning it in memory.
    _FORCED_BLENDING[visual] = original
    visual.set_gl_state = set_gl_state
    visual.set_gl_state(**ADDITIVE_BLEND_STATE)
    return visual


def release_additive_blending(visual):
    """Undo :func:`force_additive_blending`, leaving the visual as napari made it."""
    original = _FORCED_BLENDING.pop(visual, None)
    if original is None:
        return
    try:
        visual.set_gl_state = original
    except (AttributeError, TypeError):
        pass


def get_layer_controls(viewer, layer):
    """Return the Qt controls widget for *layer*, or ``None`` if unavailable.

    Unlike the visual, the controls widget is only needed for cosmetic
    conveniences, so a missing widget is reported as ``None`` rather than
    raising.
    """
    try:
        qt_viewer = get_qt_viewer(viewer)
    except NapariInternalsChanged:
        return None
    controls = getattr(qt_viewer, "controls", None)
    widgets = getattr(controls, "widgets", None)
    if widgets is None:
        return None
    try:
        return widgets[layer]
    except (KeyError, TypeError):
        return None


def set_builtin_layer_docks_visible(viewer, visible):
    """Best-effort visibility change for napari's built-in layer docks.

    This is cosmetic and must never prevent the plugin from opening.  Keeping
    the lookup here also prevents optional viewer-chrome behavior from
    scattering private Qt access through the dock widget and reader.
    """
    try:
        qt_viewer = get_qt_viewer(viewer)
    except NapariInternalsChanged:
        return False
    changed = False
    for name in ("dockLayerControls", "dockLayerList"):
        dock = getattr(qt_viewer, name, None)
        if dock is not None:
            dock.setVisible(bool(visible))
            changed = True
    return changed


# ----------------------------------------------------------------------
# Instanced rendering (Level 3 prototype)
# ----------------------------------------------------------------------


def enable_instanced_backend():
    """Select VisPy's ``gl+`` backend, which is the only one with instancing.

    Must run before *any* GL context exists — so before napari builds its
    canvas, which in practice means before ``napari.Viewer()``. The choice is
    process-global: every layer in the viewer, not only ours, then renders
    through PyOpenGL. That cost is real and is recorded in
    ``docs/backend-comparison.md``; this function exists so it is at least made
    deliberately and in one place.

    Returns True if instancing is available afterwards.
    """
    import vispy

    try:
        vispy.use(gl="gl+")
    except Exception:  # noqa: BLE001 - a context already exists, or no PyOpenGL
        pass
    return instancing_available()


def instancing_available():
    """True when the active GL backend can issue instanced draw calls."""
    from vispy.gloo import gl

    return hasattr(gl.current_backend, "glDrawElementsInstanced")


# ----------------------------------------------------------------------
# VisPy camera drag state
# ----------------------------------------------------------------------

#: Marker so a camera is not wrapped twice.
_GUARDED = "_napari_storm_drag_guard"


def guard_camera_drag_state(viewer):
    """Stop a Shift released mid-drag from crashing the 3-D camera.

    VisPy's `Base3DRotationCamera` keeps one `_event_value` for every drag
    gesture and only clears it on mouse *release*:

    * Shift + drag translates, and stores ``self.center`` -- three world
      coordinates;
    * plain drag rotates, and expects two *screen* coordinates.

    Let go of Shift without letting go of the mouse and the rotate branch reads
    the translate branch's three-tuple, so `_arcball` raises
    ``ValueError: too many values to unpack``. It is an ordinary gesture: pan,
    then keep dragging to turn.

    VisPy fixed this in newer releases by slicing to ``[:2]``, which stops the
    exception but still rotates from a garbage previous position for one frame.
    This resets the value instead, so the rotation simply starts from where the
    pointer is -- and it works on the older VisPy releases where the upstream
    fix is absent, which the supported napari range can still resolve.

    Idempotent, and re-applied when napari swaps cameras on an ndisplay change.
    """
    canvas = getattr(get_qt_viewer(viewer), "canvas", None)
    view = getattr(canvas, "view", None)
    camera = getattr(view, "camera", None)
    if camera is None or getattr(camera, _GUARDED, False):
        return camera

    original = camera.viewbox_mouse_event

    def viewbox_mouse_event(event):
        if getattr(event, "type", None) == "mouse_move":
            value = getattr(camera, "_event_value", None)
            try:
                stale = value is not None and len(value) != 2
            except TypeError:
                stale = False
            if stale:
                mouse_event = getattr(event, "mouse_event", None)
                modifiers = getattr(mouse_event, "modifiers", ())
                if 1 in getattr(event, "buttons", ()) and not modifiers:
                    # A translate's centre, about to be read as a position.
                    camera._event_value = None
        return original(event)

    camera.viewbox_mouse_event = viewbox_mouse_event
    setattr(camera, _GUARDED, True)
    return camera
