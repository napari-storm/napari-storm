"""The instanced backend, as a `LocalizationRenderer` the application can use.

Same contract as the other two, so it can be injected in their place and driven
by the same planner, the same controls and the same benchmark harness.

Requires VisPy's ``gl+`` GL backend. Call
`napari_storm.napari_particles._napari_compat.enable_instanced_backend()`
*before* napari builds a canvas, or construction fails with an explanation
rather than a blank viewer.
"""

from __future__ import annotations

from ..core.renderer import LayerAppearance, LocalizationRenderer
from ._napari_compat import instancing_available
from .instanced_layer import InstancedParticles

__all__ = ["InstancedRenderer"]


class InstancedRenderer(LocalizationRenderer):
    """Draws each dataset as one instanced quad."""

    def __init__(self, viewer):
        if not instancing_available():
            raise RuntimeError(
                "instanced rendering needs VisPy's 'gl+' backend, which must be "
                "selected before any GL context exists. Call "
                "napari_storm.napari_particles._napari_compat."
                "enable_instanced_backend() before creating the viewer, or set "
                "NAPARI_STORM_RENDERER=instanced before importing napari."
            )
        self.viewer = viewer
        self._layers = {}
        self._closing = set()
        self.on_layer_removed_by_host = None
        viewer.layers.events.removed.connect(self._on_layer_removed)

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------

    def layer(self, dataset_id):
        return self._layers.get(dataset_id)

    def is_open(self, dataset_id):
        return dataset_id in self._layers

    @property
    def layers(self):
        return list(self._layers.values())

    def detach(self):
        try:
            self.viewer.layers.events.removed.disconnect(self._on_layer_removed)
        except (ValueError, TypeError, RuntimeError):
            pass

    def _on_layer_removed(self, event):
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
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, dataset_id, request):
        self.close(dataset_id)
        layer = InstancedParticles(
            request.coords,
            size=request.size,
            sigmas=request.sigmas,
            values=request.values,
            colormap=request.colormap,
            name=request.name,
        )
        layer.add_to_viewer(self.viewer)
        self._layers[dataset_id] = layer
        return layer

    def update(self, dataset_id, request):
        layer = self._layers.get(dataset_id)
        if layer is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        layer.update_particle_data(
            coords=request.coords,
            size=request.size,
            sigmas=request.sigmas,
            values=request.values,
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
        layer = self._layers.get(dataset_id)
        return None if layer is None else tuple(layer.contrast_limits_range)

    def close(self, dataset_id):
        layer = self._layers.pop(dataset_id, None)
        if layer is None:
            return
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

    def host_bytes(self, dataset_id):
        layer = self._layers.get(dataset_id)
        return 0 if layer is None else layer.host_bytes()
