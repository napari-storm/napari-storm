"""VisPy's settings for instanced drawing, without napari.

What the instanced billboards need from VisPy itself: the ``gl+`` GL backend,
the only one with instanced draw calls, chosen before any GL context exists,
and the additive blend state the Gaussians are summed with.

Moved unchanged from `_napari_compat.py`, which imports them from here, so
every old import keeps working.  The moved docstrings still speak of napari,
the host they were written for.  The functions import VisPy themselves.
"""

#: True additive blending: add the source, keep the destination.  Depth testing
#: is off because splats are order-independent once they only ever sum.
ADDITIVE_BLEND_STATE = {
    "blend": True,
    "depth_test": False,
    "blend_func": ("src_alpha", "one"),
}


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
