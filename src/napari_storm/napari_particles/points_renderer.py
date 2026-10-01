"""The napari-native candidate for the Level 3 backend decision.

The plan requires prototyping "true instanced billboards" against "the most
maintainable napari/VisPy-native alternative" on the same fixtures before
committing to a production backend. This is that alternative: one napari
``Points`` layer per dataset, no custom shader, no private VisPy access, no
vendored renderer to maintain.

What it gives up is stated plainly, because the decision turns on it:

* **No Gaussian falloff.** A point is a disc of uniform colour. The billboard
  backend renders an actual 2-D Gaussian whose covariance follows the camera.
  For an SMLM reconstruction that is not cosmetic -- the Gaussian *is* the
  localization-precision estimate being displayed.
* **No per-localization width.** ``size`` is per-point, so variable-Gaussian
  mode can size each disc by its uncertainty, but the intensity weighting that
  makes a tight localization brighter has no equivalent in a flat disc.

What it gives back is memory and maintenance: it stores a position, a size and
a colour per localization, and nothing else.

Its default look is additive, spherically shaded discs as wide as the
Gaussian billboard -- the closest a Points layer gets to a splat.  Under the
``"disc"`` footprint it draws what the other backends draw there: opaque, flat,
depth-tested discs of radius sigma.  The ``min_disc_px`` floor goes through
napari's own marker limit, which bounds the marker sprite rather than the disc
drawn inside it, so here it is approximate.
"""

from __future__ import annotations

import numpy as np

from ..core.renderer import (
    DEFAULT_MIN_DISC_PX,
    FOOTPRINT_DISC,
    FOOTPRINT_GAUSSIAN,
    LayerAppearance,
    LocalizationRenderer,
)
from .instanced_layer import DISC_QUAD_SCALE

#: napari's own floor and ceiling for marker sizes, in canvas pixels.
_NAPARI_CANVAS_SIZE_LIMITS = (2, 10000)

__all__ = ["NapariPointsRenderer"]


class NapariPointsRenderer(LocalizationRenderer):
    """Draws each dataset as one napari ``Points`` layer."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._layers = {}
        self._footprints = {}  # dataset id -> (footprint, min_disc_px)
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
        self._footprints.pop(dataset_id, None)
        if self.on_layer_removed_by_host is not None:
            self.on_layer_removed_by_host(dataset_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _point_sizes(request, footprint=FOOTPRINT_GAUSSIAN):
        """One disc diameter per localization, from the normalized sigmas.

        The request's ``size`` is the billboard edge for the largest Gaussian
        and its ``sigmas`` are normalized against that same largest one, so the
        product recovers a per-localization width on the same scale the
        billboard backend draws.  A disc is narrower: two sigmas across.
        """
        widest = np.max(request.sigmas[:, 1:], axis=1)
        scale = DISC_QUAD_SCALE if footprint == FOOTPRINT_DISC else 1.0
        return np.asarray(request.size * widest * scale, dtype=np.float32)

    def _apply_footprint(self, dataset_id, previous=None):
        """Put the dataset's footprint on its layer.

        Sizes are rescaled from what the layer holds rather than recomputed, so
        a footprint change needs no request -- appearance never rebuilds data.
        """
        layer = self._layers[dataset_id]
        footprint, min_disc_px = self._footprints[dataset_id]
        if previous is not None and previous != footprint:
            ratio = (
                DISC_QUAD_SCALE if footprint == FOOTPRINT_DISC else 1 / DISC_QUAD_SCALE
            )
            layer.size = np.asarray(layer.size, dtype=np.float64) * ratio
        if footprint == FOOTPRINT_DISC:
            layer.shading = "none"
            layer.blending = "opaque"
            layer.canvas_size_limits = (
                float(min_disc_px),
                _NAPARI_CANVAS_SIZE_LIMITS[1],
            )
        else:
            layer.shading = "spherical"
            layer.blending = "additive"
            layer.canvas_size_limits = _NAPARI_CANVAS_SIZE_LIMITS

    def open(self, dataset_id, request):
        self.close(dataset_id)
        layer = self.viewer.add_points(
            request.coords,
            name=request.name,
            size=self._point_sizes(request),
            features={"value": np.asarray(request.values)},
            face_color="value",
            face_colormap=request.colormap if request.colormap is not None else "gray",
            border_width=0.0,
            blending="additive",
            # A disc is the closest this backend gets to a splat; spherical
            # shading at least gives 3-D data some depth cue.
            shading="spherical",
            out_of_slice_display=True,
        )
        self._layers[dataset_id] = layer
        self._footprints[dataset_id] = (FOOTPRINT_GAUSSIAN, DEFAULT_MIN_DISC_PX)
        return layer

    def update(self, dataset_id, request):
        layer = self._layers.get(dataset_id)
        if layer is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        # Points has no in-place buffer API; assigning data replaces the arrays
        # but keeps the layer, its colormap and its event connections.  That it
        # cannot do better than this is itself a result for the comparison.
        layer.data = request.coords
        layer.features = {"value": np.asarray(request.values)}
        layer.size = self._point_sizes(request, self._footprints[dataset_id][0])
        layer.face_color = "value"
        layer.visible = True
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
            layer.face_colormap = appearance.colormap
            layer.face_color = "value"
        if appearance.opacity is not None:
            layer.opacity = float(appearance.opacity)
        if appearance.contrast_limits is not None:
            layer.face_contrast_limits = tuple(appearance.contrast_limits)
        if appearance.visible is not None:
            layer.visible = bool(appearance.visible)
        if appearance.footprint is not None or appearance.min_disc_px is not None:
            previous, min_disc_px = self._footprints[dataset_id]
            footprint = appearance.footprint or previous
            if appearance.min_disc_px is not None:
                min_disc_px = float(appearance.min_disc_px)
            self._footprints[dataset_id] = (footprint, min_disc_px)
            self._apply_footprint(dataset_id, previous=previous)
        return layer

    def appearance(self, dataset_id):
        layer = self._layers.get(dataset_id)
        if layer is None:
            return None
        footprint, min_disc_px = self._footprints[dataset_id]
        return LayerAppearance(
            colormap=layer.face_colormap,
            opacity=layer.opacity,
            contrast_limits=tuple(layer.face_contrast_limits or (0.0, 1.0)),
            visible=layer.visible,
            footprint=footprint,
            min_disc_px=min_disc_px,
        )

    def value_range(self, dataset_id):
        layer = self._layers.get(dataset_id)
        if layer is None:
            return None
        values = np.asarray(layer.features["value"])
        if values.size == 0:
            return None
        return float(np.min(values)), float(np.max(values))

    def close(self, dataset_id):
        layer = self._layers.pop(dataset_id, None)
        self._footprints.pop(dataset_id, None)
        if layer is None:
            return
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
        """Host-side bytes this backend holds for *dataset_id*."""
        layer = self._layers.get(dataset_id)
        if layer is None:
            return 0
        total = 0
        for array in (
            layer.data,
            np.asarray(layer.size),
            np.asarray(layer.face_color),
            np.asarray(layer.features["value"]),
        ):
            if isinstance(array, np.ndarray):
                total += array.nbytes
        return total
