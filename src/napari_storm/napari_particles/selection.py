"""Which renderer the plugin uses, and what happens when it cannot.

Level 3 asked for the production backend to be chosen "behind a measured backend
decision". The measurement is in `docs/backend-comparison.md` and it is not
close: the instanced backend costs 28 B/localization against 352, and one update
on the 5M fixture takes 0.16 s against 2.47 s. So it is the default.

It is not, however, unconditional. Instancing needs VisPy's ``gl+`` backend, and
while napari selects that itself -- `napari/_vispy/visuals/markers.py` calls
``use(gl='gl+')`` at import for napari's own instanced Points -- it falls back to
``gl2`` with a warning when PyOpenGL is missing. On ``gl2`` there is no
``glDrawElementsInstanced`` at all and the draw call raises. A viewer that cannot
instance must still be able to show a reconstruction, so the choice is made per
session, from the capability that is actually present, and the fallback says so
rather than failing at the first draw.
"""

from __future__ import annotations

from ._napari_compat import instancing_available

__all__ = ["select_renderer", "renderer_class_for_session"]


def renderer_class_for_session():
    """The renderer class this session can actually use, and why.

    Returns ``(class, reason)``. *reason* is ``None`` when the preferred backend
    is available and a human-readable explanation when it is not.
    """
    from .instanced_renderer import InstancedRenderer
    from .renderer import NapariParticlesRenderer

    if instancing_available():
        return InstancedRenderer, None
    return (
        NapariParticlesRenderer,
        "Instanced rendering is unavailable: VisPy's 'gl+' backend was not "
        "selected, which usually means PyOpenGL is not installed. Falling back "
        "to the original billboard renderer -- the image is the same, but it "
        "uses about 12x more memory per localization and updates more slowly.",
    )


def select_renderer(viewer, notify=True):
    """Build the best renderer this session supports.

    Set *notify* to False to keep the fallback silent, which is what the tests
    that force the fallback want.
    """
    renderer_class, reason = renderer_class_for_session()
    if reason is not None and notify:
        try:
            from napari.utils.notifications import show_warning

            show_warning(reason)
        except Exception:  # noqa: BLE001 - never block rendering on a notification
            pass
    return renderer_class(viewer)
