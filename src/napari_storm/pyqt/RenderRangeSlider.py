import napari
import numpy as np
from qtpy.QtCore import Qt, QTimer
from superqt import QDoubleRangeSlider

from ..DataToLayerInterface import DataToLayerInterface


class RangeSlider2(QDoubleRangeSlider):
    """
    Range slider controlling a render-range along one axis (x/y/z).

    While dragging, a single shared Surface layer ('render-range') owned by the
    parent napari_storm widget is updated to show the *combined* current render-range
    box across all three axes.  The layer is hidden on release and re-used on the
    next drag — no layer churn during interaction.
    """

    def __init__(self, parent=None, type="x"):
        super().__init__(parent)
        self._owner = parent
        self._axis = type  # 'x', 'y', or 'z'

        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 100)
        self.setSingleStep(1)
        self.setValue((0, 100))

        self._dragging = False
        self._pending_preview_update = False
        self.valueChanged.connect(self._on_value_changed)

    # ------------------------------------------------------------------
    # Viewer helper
    # ------------------------------------------------------------------

    def _viewer(self):
        v = getattr(self._owner, "viewer", None) or getattr(self._owner, "_viewer", None)
        return v if v is not None else napari.current_viewer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to full range immediately and commit."""
        self.blockSignals(True)
        self.setValue((0, 100))
        self.blockSignals(False)
        self._apply_range()

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._dragging = True
        self._ensure_preview_layer()
        preview = self._owner._render_range_preview
        if preview is not None:
            preview.visible = True
        self._schedule_preview_update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._dragging = False

        self._apply_range()

        preview = self._owner._render_range_preview
        if preview is not None:
            preview.visible = False

        DataToLayerInterface.update_layers(self._owner.data_to_layer_itf)

    # ------------------------------------------------------------------
    # Slider callback
    # ------------------------------------------------------------------

    def _on_value_changed(self, *args):
        if not self._dragging:
            return
        self._schedule_preview_update()

    # ------------------------------------------------------------------
    # Preview layer handling
    # ------------------------------------------------------------------

    def _schedule_preview_update(self):
        if self._pending_preview_update:
            return
        self._pending_preview_update = True
        QTimer.singleShot(0, self._do_preview_update)

    def _do_preview_update(self):
        self._pending_preview_update = False
        if not self._dragging:
            return

        self._ensure_preview_layer()
        preview = self._owner._render_range_preview
        if preview is None:
            return

        coords, faces = self.get_coords_faces()
        preview.data = (coords, faces)
        preview.opacity = self._owner.render_range_box_opacity
        preview.visible = True

    def _ensure_preview_layer(self):
        """
        Ensure owner._render_range_preview points to a live Surface layer.
        Creates a new one only if it doesn't exist or was removed externally.
        """
        v = self._viewer()
        owner = self._owner

        # Drop stale reference
        if (
            owner._render_range_preview is not None
            and owner._render_range_preview not in v.layers
        ):
            owner._render_range_preview = None

        # Lazy-create
        if owner._render_range_preview is None:
            coords, faces = self.get_coords_faces()
            owner._render_range_preview = v.add_surface(
                (coords, faces),
                opacity=owner.render_range_box_opacity,
                shading="none",
                name="render-range",
                colormap=owner.active_render_range_box_color,
            )
        else:
            # The layer outlives the drag that made it, so the colour picked
            # since then has to be pushed onto it.  Passing the colormap to
            # add_surface() only ever coloured the *first* box: before the layer
            # was made persistent it was destroyed and recreated on every drag,
            # which applied the current choice as a side effect.
            owner._render_range_preview.colormap = owner.active_render_range_box_color

    # ------------------------------------------------------------------
    # Commit render range
    # ------------------------------------------------------------------

    def _apply_range(self):
        lo, hi = self.value()
        if self._axis == "x":
            maxv = self.maximum()
            lo, hi = maxv - hi, maxv - lo
        self._owner.update_render_range(self._axis, (lo, hi))

    # ------------------------------------------------------------------
    # Geometry: full combined render-range box
    # ------------------------------------------------------------------

    def get_coords_faces(self):
        """
        Compute the 8 corner vertices and 12 triangles of the full combined
        render-range box (all three axes) in napari layer space.

        For the currently-dragged axis the *live* slider value is used;
        for the other two axes the already-committed render_config values are used.
        """
        owner = self._owner
        full = owner.data_to_layer_itf

        # Committed percent ranges from render_config
        x_pct = list(owner.render_range_slider_x_percent)
        y_pct = list(owner.render_range_slider_y_percent)
        z_pct = list(owner.render_range_slider_z_percent)

        # Override current axis with live (uncommitted) slider value
        lo_raw, hi_raw = self.value()
        maxv = self.maximum()
        if self._axis == "x":
            # X slider is displayed inverted: high slider position = low data range
            x_pct = [maxv - hi_raw, maxv - lo_raw]
        elif self._axis == "y":
            y_pct = [lo_raw, hi_raw]
        else:
            z_pct = [lo_raw, hi_raw]

        # Napari layer extents, stored as [min, max] in true nanometres.  These
        # used to be read as [1] alone, which held only while the auto-offset
        # forced every axis to start at zero; percentages now map onto the full
        # [min, max] span.
        rrx = full.render_range_x  # dim2 / x extent in both 2-D and 3-D
        rry = full.render_range_y  # dim1 / y extent in both 2-D and 3-D
        rrz = full.render_range_z  # dim0 extent (3-D) / [inf,-inf] (2-D)

        x0, x1 = full.percent_to_absolute(rrx, x_pct)
        y0, y1 = full.percent_to_absolute(rry, y_pct)

        is_3d = bool(owner.zdim) and not np.isinf(rrz[1])
        if is_3d:
            z0, z1 = full.percent_to_absolute(rrz, z_pct)
        else:
            z0 = z1 = 1.0  # flat slab at z=1 for 2-D data

        # 8 vertices ordered as [dim0/z, dim1/y, dim2/x] -- napari's order.
        # Only the two lateral components moved; each row is still the same
        # physical corner in the same position, so `faces` indexes them
        # unchanged.
        coords = np.array(
            [
                [z0, y0, x0],
                [z0, y1, x0],
                [z0, y0, x1],
                [z0, y1, x1],
                [z1, y0, x0],
                [z1, y1, x0],
                [z1, y0, x1],
                [z1, y1, x1],
            ],
            dtype=float,
        )

        # 12 triangles — 2 per face of the box
        faces = np.array(
            [
                [0, 1, 2], [1, 2, 3],   # z = z0 face
                [4, 5, 6], [5, 6, 7],   # z = z1 face
                [0, 2, 4], [2, 4, 6],   # x = x0 face
                [1, 3, 5], [3, 5, 7],   # x = x1 face
                [0, 1, 4], [1, 4, 5],   # y = y0 face
                [2, 3, 6], [3, 6, 7],   # y = y1 face
            ],
            dtype=int,
        )

        return coords, faces
