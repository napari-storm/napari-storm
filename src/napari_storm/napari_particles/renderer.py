"""The napari/VisPy backend: Gaussian billboards drawn as a Surface layer.

One of the two implementations Level 3 compares. Everything specific to
*this* way of drawing localizations lives here — the `Particles` layer, its
shader filter, its viewer registration and its teardown — so that a second
backend can be written against `LocalizationRenderer` without touching the code
that decides what to draw.
"""

from __future__ import annotations

import numpy as np

from ..core.renderer import LayerAppearance, LocalizationRenderer
from .particles import Particles

__all__ = ["NapariParticlesRenderer"]


class NapariParticlesRenderer(LocalizationRenderer):
    """Draws each dataset as one `Particles` layer in a napari viewer."""

    def __init__(self, viewer):
        self.viewer = viewer
        # Layers by dataset id.  Ids are never reused, so a handle cannot be
        # mistaken for a different dataset's after a close.
        self._layers = {}
        # Set while we are removing a layer ourselves, so the viewer's own
        # `removed` event does not recurse back into close().
        self._closing = set()
        #: callable(dataset_id) -- the user deleted one of our layers in napari.
        self.on_layer_removed_by_host = None
        viewer.layers.events.removed.connect(self._on_layer_removed)

    def detach(self):
        """Stop listening to the viewer.  Call before dropping the renderer."""
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layer_removed)
        except (ValueError, TypeError, RuntimeError):
            # Already disconnected, or the viewer is gone.
            pass

    def _on_layer_removed(self, event):
        """A layer left the viewer.  If it was ours, stop pretending it is open.

        napari's layer list is the user's, and they can delete a layer from it
        at any time. Without this the renderer kept a handle to a layer nobody
        could see, and the next update wrote buffers into it.
        """
        layer = getattr(event, "value", None)
        dataset_id = next(
            (key for key, known in self._layers.items() if known is layer), None
        )
        if dataset_id is None or dataset_id in self._closing:
            return
        self._layers.pop(dataset_id, None)
        layer.close()
        if self.on_layer_removed_by_host is not None:
            self.on_layer_removed_by_host(dataset_id)

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------

    def layer(self, dataset_id):
        """The layer drawing *dataset_id*, or None."""
        return self._layers.get(dataset_id)

    def is_open(self, dataset_id):
        return dataset_id in self._layers

    @property
    def layers(self):
        """Every live layer, in no particular order."""
        return list(self._layers.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, dataset_id, request):
        self.close(dataset_id)
        layer = Particles(
            request.coords,
            size=request.size,
            values=request.values,
            sigmas=request.sigmas,
            antialias=request.antialias,
            colormap=request.colormap,
            filter=None,
            name=request.name,
        )
        layer.add_to_viewer(self.viewer)
        # Attaches the Gaussian shader filter.  Deliberately not napari's own
        # `shading`, which accepts only None/'flat'/'smooth' -- see P0-08.
        layer.shader = "gaussian"
        self._layers[dataset_id] = layer
        return layer

    def update(self, dataset_id, request):
        """Rewrite the layer's buffers, keeping the layer and its visual.

        `Particles.update_particle_data` rebuilds billboard geometry from
        whatever coordinates it is given, including a different number of them,
        so a filter change needs new buffers but not a new layer.
        """
        layer = self._layers.get(dataset_id)
        if layer is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        layer.update_particle_data(
            coords=request.coords,
            size=request.size,
            values=request.values,
            sigmas=request.sigmas,
        )
        layer.visible = True
        # Setting visible makes napari recompute layer order, which re-applies
        # its own blending preset over ours.  Re-assert afterwards, or the
        # splats stop summing and the reconstruction becomes whichever Gaussian
        # was drawn last.
        layer._apply_blend_state()
        return layer

    def set_visible(self, dataset_id, visible):
        layer = self._layers.get(dataset_id)
        if layer is not None:
            layer.visible = bool(visible)

    def set_appearance(self, dataset_id, appearance):
        """Apply the non-None fields of *appearance* to the napari layer.

        This is where "a napari Surface layer has a colormap, an opacity and
        contrast limits" is allowed to be known. Before, every channel control
        reached through to the layer object and set those directly, which meant
        no second backend could ever satisfy them.
        """
        layer = self._layers.get(dataset_id)
        if layer is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        if appearance.colormap is not None:
            layer.colormap = appearance.colormap
        if appearance.opacity is not None:
            layer.opacity = float(appearance.opacity)
        if appearance.contrast_limits is not None:
            layer.contrast_limits = list(appearance.contrast_limits)
        if appearance.visible is not None:
            layer.visible = bool(appearance.visible)
        return layer

    def appearance(self, dataset_id):
        layer = self._layers.get(dataset_id)
        if layer is None:
            return None
        return LayerAppearance(
            colormap=layer.colormap,
            opacity=layer.opacity,
            contrast_limits=tuple(layer.contrast_limits),
            visible=layer.visible,
        )

    def value_range(self, dataset_id):
        """napari tracks this for us as the layer's contrast-limits range."""
        layer = self._layers.get(dataset_id)
        return None if layer is None else tuple(layer.contrast_limits_range)

    def close(self, dataset_id):
        """Disconnect the layer's callbacks and filters, then drop it."""
        layer = self._layers.pop(dataset_id, None)
        if layer is None:
            return
        # close() disconnects the three layer-list callbacks and detaches the
        # shader filters; without it each removed layer left live connections
        # behind and callback work grew with every dataset ever opened.
        layer.close()
        self._closing.add(dataset_id)
        try:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        finally:
            self._closing.discard(dataset_id)

    def close_all(self):
        for dataset_id in list(self._layers):
            self.close(dataset_id)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def host_bytes(self, dataset_id):
        """Host-side bytes this backend holds for *dataset_id*.

        Every localization is six vertices with its centre, sigma and value
        repeated across them, which is what the ~352 B/localization figure in
        the baseline report is measuring.
        """
        layer = self._layers.get(dataset_id)
        if layer is None:
            return 0
        total = 0
        for name in (
            "_coords",
            "_centercoords",
            "_sigmas",
            "_size",
            "_texcoords",
            "_view_faces",
            "_view_vertices",
        ):
            array = getattr(layer, name, None)
            if isinstance(array, np.ndarray):
                total += array.nbytes
        data = getattr(layer, "data", None)
        if isinstance(data, tuple):
            total += sum(a.nbytes for a in data if isinstance(a, np.ndarray))
        return total
