import logging
"""
Live controls panel for one reference-image napari layer.

Inserted at row 0 of channel_controls_widget_layout (a QFormLayout) when an
image is imported.  Provides live pixel-size and offset spinboxes plus a
"✕ Remove" button.
"""

import numpy as np
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)
from superqt import QDoubleRangeSlider

from .image_import_dialog import ImageImportResult

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orientation expansion
# ---------------------------------------------------------------------------

_THIN = 1.0  # nm — collapsed spatial dimension makes the slab a visual plane

#: Normals in the layer's ``(z, y, x)`` axis order: the plane a reference image
#: lies in is named by the two axes it spans, so its normal is the third.
_PLANE_NORMALS = {
    "XY": (1.0, 0.0, 0.0),   # constant z
    "XZ": (0.0, 1.0, 0.0),   # constant y
    "YZ": (0.0, 0.0, 1.0),   # constant x
}


def _quarter_turn_matrix(axis, quarter_turns=1):
    """Return a 3-D rotation in the layer's ``(z, y, x)`` order.

    The public axis names remain the familiar physical X/Y/Z names while the
    layer itself stores coordinates as Z/Y/X.  Building the rotation in XYZ
    and permuting it avoids silently assigning a button to the wrong axis.
    """
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unknown rotation axis: {axis!r}")

    angle = np.deg2rad(90 * quarter_turns)
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "x":
        rotation_xyz = np.array(
            ((1, 0, 0), (0, c, -s), (0, s, c)), dtype=float
        )
    elif axis == "y":
        rotation_xyz = np.array(
            ((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=float
        )
    else:
        rotation_xyz = np.array(
            ((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=float
        )

    # xyz = xyz_from_zyx @ zyx
    xyz_from_zyx = np.array(((0, 0, 1), (0, 1, 0), (1, 0, 0)))
    rotation_zyx = xyz_from_zyx.T @ rotation_xyz @ xyz_from_zyx
    rotation_zyx[np.isclose(rotation_zyx, 0, atol=1e-12)] = 0
    rotation_zyx[np.isclose(rotation_zyx, 1, atol=1e-12)] = 1
    rotation_zyx[np.isclose(rotation_zyx, -1, atol=1e-12)] = -1
    return rotation_zyx


def _expand_image(result: ImageImportResult):
    """
    Apply orientation expansion and return *(data, scale, translate)* ready
    for napari.add_image().

    Coordinate convention, now napari's own:
        dim0 = z_pos_nm, dim1 = y_pos_nm, dim2 = x_pos_nm

    Orientation mappings
    --------------------
    XY  — en-face (horizontal plane)
        img is stored rows = y, cols = x, as every TIFF and OME file is, which
        *is* the layer order, so only the z axis is added:  (H, W[, C]) →
        (1, H, W[, C])
        scale     = (THIN, px_xy, px_xy)
        translate = (z_off, y_off, x_off)

    XZ  — side view, constant-Y plane
        img[:, np.newaxis, :]  → (H, 1, W)
        rows = z_pos_nm, cols = x_pos_nm, thin slab in Y at y_off
        scale     = (px_z, THIN, px_xy)
        translate = (z_off, y_off, x_off)

    YZ  — side view, constant-X plane
        img[:, :, np.newaxis]  → (H, W, 1)
        rows = z_pos_nm, cols = y_pos_nm, thin slab in X at x_off
        scale     = (px_z, px_xy, THIN)
        translate = (z_off, y_off, x_off)

    3D  — volumetric stack, already in this order
        (D, H, W[, C]), dim1 = y_pos_nm, dim2 = x_pos_nm
        scale     = (px_z, px_xy, px_xy)
        translate = (z_off, y_off, x_off)

    **There is no lateral swap here any more, and its removal is the point.**
    This function used to transpose every en-face and volumetric reference
    image, because localizations were drawn with dim1 = x while an image file
    stores rows = y. That was the wrong end to fix: the localizations were the
    ones disagreeing with napari, so a correct file was bent to meet them and
    the 90-degree rotation controls existed largely to undo it by hand. The
    planner now emits ``(z, y, x)`` and a reference image is placed as it is
    stored. XZ and YZ change too -- their rows were always z, but their
    remaining axis moves with everything else.
    """
    img = result.img
    o = result.orientation
    px_xy = result.px_xy_nm
    px_z = result.px_z_nm
    x_off = result.x_off_nm
    y_off = result.y_off_nm
    z_off = result.z_off_nm

    translate = (z_off, y_off, x_off)

    if o == "XY":
        # Rows are already y and columns already x; only z has to be added.
        data = img[np.newaxis, ...]          # (1, H, W[, C])
        scale = (_THIN, px_xy, px_xy)
    elif o == "XZ":
        data = img[:, np.newaxis, :]         # (H, 1, W)
        scale = (px_z, _THIN, px_xy)
    elif o == "YZ":
        data = img[:, :, np.newaxis]         # (H, W, 1)
        scale = (px_z, px_xy, _THIN)
    elif o == "3D":
        data = img                            # (D, H, W[, C]) already (z, y, x)
        scale = (px_z, px_xy, px_xy)
    else:
        raise ValueError(f"Unknown orientation: {o!r}")

    return data, scale, translate


def _is_rgb_reference(result: ImageImportResult):
    """Resolve RGB using the user-selected spatial orientation.

    A three-dimensional array ending in 3 or 4 is ambiguous: it is planar RGB
    for XY/XZ/YZ, but a scalar volume when the user selected 3D.  An RGB volume
    needs four raw dimensions (Z, X, Y, C).
    """
    expected_ndim = 4 if result.orientation == "3D" else 3
    return result.img.ndim == expected_ndim and result.img.shape[-1] in (3, 4)


def _reference_image_rendering_options(data, orientation, rgb=False):
    """Return stable napari rendering options for a reference image.

    A planar reference expanded to one voxel must be rendered as an embedded
    plane, not ray-marched as a singleton volume.  Depth testing is disabled
    because layer order already places references below localizations, and a
    near-coplanar depth comparison is the other source of camera-motion
    flicker.  True 3-D stacks retain napari's normal volume rendering.
    """
    if orientation == "3D":
        return {}

    spatial_shape = np.asarray(data.shape[:-1] if rgb else data.shape, dtype=float)
    if spatial_shape.shape != (3,):
        raise ValueError("A planar reference image must have three spatial axes")
    position = tuple((spatial_shape - 1) / 2)
    return {
        "depiction": "plane",
        "plane": {
            "position": position,
            "normal": _PLANE_NORMALS[orientation],
            "thickness": 1.0,
        },
        "blending": "translucent_no_depth",
    }


# ---------------------------------------------------------------------------
# Helper: spinbox factory
# ---------------------------------------------------------------------------


def _make_spinbox(lo, hi, decimals, suffix):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSuffix(suffix)
    return sb


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------


class ImageLayerControls(QWidget):
    """
    Controls for one reference-image layer.

    Placed via QFormLayout.insertRow(0, widget) so it spans the full width of
    channel_controls_placeholder and sits above the localisation channel rows.
    Removal is handled by QFormLayout.removeRow so no parent-layout juggling
    is needed in the caller.
    """

    def __init__(self, layer, result: ImageImportResult, dock_widget):
        super().__init__()
        self._layer = layer
        self._result = result
        self._dock_widget = dock_widget
        self._orientation = result.orientation

        self._build_ui()
        self._sync_from_layer()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QFormLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 6, 0, 6)

        # ── Header row: layer name + Remove button ────────────────────
        header = QWidget()
        h_row = QHBoxLayout(header)
        h_row.setContentsMargins(0, 0, 0, 0)

        name_lbl = QLabel(f"<b>{self._layer.name}</b>")
        name_lbl.setStyleSheet("color: rgb(220, 228, 236); font-size: 12px;")

        remove_btn = QPushButton("✕ Remove")
        remove_btn.setFixedHeight(20)
        remove_btn.setStyleSheet(
            "QPushButton { color: rgb(210, 80, 80); font-size: 10px; border: none; }"
            "QPushButton:hover { color: rgb(250, 100, 100); }"
        )
        remove_btn.clicked.connect(self._on_remove)

        h_row.addWidget(name_lbl, 1)
        h_row.addWidget(remove_btn)
        layout.addRow(header)

        # RGB layers do not support napari's scalar contrast or colormap
        # properties, so these controls only exist for grayscale images.
        self._contrast_slider = None
        self._colormap_combo = None
        if not getattr(self._layer, "rgb", False):
            self._colormap_combo = QComboBox()
            self._colormap_combo.addItems(sorted(AVAILABLE_COLORMAPS))
            current_index = self._colormap_combo.findText(self._layer.colormap.name)
            if current_index >= 0:
                self._colormap_combo.setCurrentIndex(current_index)
            self._colormap_combo.setToolTip("Colormap for this reference image")
            layout.addRow("Colormap:", self._colormap_combo)

            self._contrast_slider = QDoubleRangeSlider(Qt.Horizontal)
            contrast_min, contrast_max = self._layer.contrast_limits_range
            contrast_min = float(contrast_min)
            contrast_max = float(contrast_max)
            self._contrast_slider.setRange(contrast_min, contrast_max)
            self._contrast_slider.setValue(tuple(self._layer.contrast_limits))
            span = contrast_max - contrast_min
            self._contrast_slider.setSingleStep(span / 1000 if span > 0 else 0.01)
            self._contrast_slider.setToolTip(
                "Set the lower and upper displayed intensity limits"
            )
            layout.addRow("Contrast:", self._contrast_slider)

        self._opacity_slider = None
        has_alpha_channel = (
            getattr(self._layer, "rgb", False)
            and self._result.img.shape[-1] == 4
        )
        if has_alpha_channel:
            self._opacity_slider = QSlider(Qt.Horizontal)
            self._opacity_slider.setRange(0, 100)
            self._opacity_slider.setValue(round(self._layer.opacity * 100))
            self._opacity_slider.setToolTip(
                "Multiply the image's per-pixel alpha by a uniform opacity"
            )
            layout.addRow("Opacity:", self._opacity_slider)

        # ── Pixel size XY ─────────────────────────────────────────────
        self._px_xy_spin = _make_spinbox(0.1, 100_000.0, 2, " nm")
        self._px_xy_spin.setToolTip("Lateral (XY) pixel size in nanometres")
        layout.addRow("Pixel size XY:", self._px_xy_spin)

        # ── Pixel size Z (hidden for XY orientation) ──────────────────
        self._px_z_lbl = QLabel("Pixel size Z:")
        self._px_z_spin = _make_spinbox(0.1, 100_000.0, 2, " nm")
        self._px_z_spin.setToolTip("Axial (Z) voxel size in nanometres")
        layout.addRow(self._px_z_lbl, self._px_z_spin)

        needs_z = self._orientation in ("3D", "XZ", "YZ")
        self._px_z_lbl.setVisible(needs_z)
        self._px_z_spin.setVisible(needs_z)

        # ── Offsets ───────────────────────────────────────────────────
        x_lbl = "X position:" if self._orientation == "YZ" else "X offset:"
        y_lbl = "Y position:" if self._orientation == "XZ" else "Y offset:"

        self._x_off_spin = _make_spinbox(-100_000.0, 100_000.0, 3, " µm")
        self._y_off_spin = _make_spinbox(-100_000.0, 100_000.0, 3, " µm")
        self._z_off_spin = _make_spinbox(-100_000.0, 100_000.0, 3, " µm")

        self._rotation_buttons = {}
        self._position_rows = {}
        layout.addRow(x_lbl, self._make_position_row("x", self._x_off_spin))
        layout.addRow(y_lbl, self._make_position_row("y", self._y_off_spin))
        layout.addRow("Z offset:", self._make_position_row("z", self._z_off_spin))

        self._centre_btn = QPushButton("Centre on data")
        self._centre_btn.setToolTip(
            "Move this image onto the centre of the localisations' depth "
            "range -- the same placement it was given at import. Loading a "
            "dataset never moves it on its own."
        )
        self._centre_btn.clicked.connect(self.centre_on_data)
        layout.addRow("", self._centre_btn)

        self.setLayout(layout)

    def _make_position_row(self, axis, spinbox):
        """Place an offset spinbox beside its matching rotation buttons."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(spinbox, 1)
        for arrow, turns, direction in (
            ("↶", 1, "counter-clockwise"),
            ("↷", -1, "clockwise"),
        ):
            button = QPushButton(arrow)
            button.setFixedSize(28, 22)
            button.setToolTip(
                f"Rotate 90° {direction} around the {axis.upper()} axis"
            )
            button.clicked.connect(
                lambda _checked=False, a=axis, q=turns: self._rotate_image(a, q)
            )
            row_layout.addWidget(button)
            self._rotation_buttons[(axis, turns)] = button
        self._position_rows[axis] = row
        return row

    # ------------------------------------------------------------------
    # Initialise from current layer state
    # ------------------------------------------------------------------

    def _sync_from_layer(self):
        s = tuple(self._layer.scale)    # (dim0, dim1, dim2)
        t = tuple(self._layer.translate)  # (dim0, dim1, dim2)

        # px_xy: the non-thin spatial axis
        if self._orientation == "YZ":
            px_xy = s[2]  # dim2 = y_pos_nm
        else:
            px_xy = s[1]  # dim1 = x_pos_nm (XY, XZ, 3D)

        # px_z: dim0 = z_pos_nm (may be THIN for XY — irrelevant since hidden)
        px_z = s[0]

        # Offsets — always (z_off_nm, x_off_nm, y_off_nm)
        z_off_nm = t[0]
        x_off_nm = t[1]
        y_off_nm = t[2]

        for spin, val in (
            (self._px_xy_spin, px_xy),
            (self._px_z_spin, px_z),
            (self._x_off_spin, x_off_nm / 1000.0),
            (self._y_off_spin, y_off_nm / 1000.0),
            (self._z_off_spin, z_off_nm / 1000.0),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        if self._contrast_slider is not None:
            self._contrast_slider.blockSignals(True)
            self._contrast_slider.setValue(tuple(self._layer.contrast_limits))
            self._contrast_slider.blockSignals(False)
        if self._colormap_combo is not None:
            current_index = self._colormap_combo.findText(self._layer.colormap.name)
            if current_index >= 0:
                self._colormap_combo.blockSignals(True)
                self._colormap_combo.setCurrentIndex(current_index)
                self._colormap_combo.blockSignals(False)
        if self._opacity_slider is not None:
            self._opacity_slider.blockSignals(True)
            self._opacity_slider.setValue(round(self._layer.opacity * 100))
            self._opacity_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._px_xy_spin.valueChanged.connect(self._on_scale_changed)
        self._px_z_spin.valueChanged.connect(self._on_scale_changed)
        self._x_off_spin.valueChanged.connect(self._on_translate_changed)
        self._y_off_spin.valueChanged.connect(self._on_translate_changed)
        self._z_off_spin.valueChanged.connect(self._on_translate_changed)
        if self._contrast_slider is not None:
            self._contrast_slider.valueChanged.connect(self._on_contrast_changed)
        if self._colormap_combo is not None:
            self._colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        if self._opacity_slider is not None:
            self._opacity_slider.valueChanged.connect(self._on_opacity_changed)

    # ------------------------------------------------------------------
    # Live update callbacks
    # ------------------------------------------------------------------

    def _on_scale_changed(self):
        px_xy = self._px_xy_spin.value()
        px_z = self._px_z_spin.value()

        o = self._orientation
        if o == "XY":
            scale = (_THIN, px_xy, px_xy)
        elif o == "XZ":
            scale = (px_z, px_xy, _THIN)
        elif o == "YZ":
            scale = (px_z, _THIN, px_xy)
        else:  # 3D
            scale = (px_z, px_xy, px_xy)

        try:
            self._layer.scale = scale
        except Exception:
            pass

    def centre_on_data(self):
        """Move this image onto the centre of the localizations' depth range.

        The same rule that placed it at import, offered as an action rather
        than re-applied automatically. Loading a dataset must not move a layer
        the user has already positioned -- that is the §3.5 defect and a §7.4
        acceptance gate -- so the convenience is a button and never a trigger.
        """
        dock = getattr(self, "_dock_widget", None)
        interface = getattr(dock, "data_to_layer_itf", None)
        if interface is None:
            return
        self._z_off_spin.setValue(interface.reference_plane_z_nm() / 1000.0)
        self._on_translate_changed()

    def _on_translate_changed(self):
        # Offsets are always stored as (z_off, x_off, y_off) in napari coords
        z_nm = self._z_off_spin.value() * 1000.0   # µm → nm
        x_nm = self._x_off_spin.value() * 1000.0
        y_nm = self._y_off_spin.value() * 1000.0
        try:
            self._layer.translate = (z_nm, x_nm, y_nm)
        except Exception:
            pass

    def _on_contrast_changed(self, limits):
        try:
            self._layer.contrast_limits = tuple(float(value) for value in limits)
        except Exception as exc:
            _LOG.warning(f"napari-storm: could not update image contrast: {exc}")

    def _on_colormap_changed(self, colormap_name):
        try:
            self._layer.colormap = colormap_name
        except Exception as exc:
            _LOG.warning(f"napari-storm: could not update image colormap: {exc}")

    def _on_opacity_changed(self, percentage):
        try:
            self._layer.opacity = percentage / 100.0
        except Exception as exc:
            _LOG.warning(f"napari-storm: could not update image opacity: {exc}")

    def _rotate_image(self, axis, quarter_turns):
        """Rotate around the image centre without changing its world position."""
        try:
            # A 90-degree out-of-plane transform is singular when napari tries
            # to express it as a 2-D slice.  Rotation is inherently a 3-D
            # operation, so enter 3-D display before applying it.
            self._dock_widget.viewer.dims.ndisplay = 3
            data_shape = np.asarray(self._layer.data.shape, dtype=float)
            if getattr(self._layer, "rgb", False):
                data_shape = data_shape[:-1]
            spatial_shape = data_shape[-self._layer.ndim :]
            centre_data = (spatial_shape - 1) / 2
            centre_before = np.asarray(
                self._layer.data_to_world(tuple(centre_data)), dtype=float
            )

            increment = _quarter_turn_matrix(axis, quarter_turns)
            current = np.asarray(self._layer.rotate, dtype=float)
            updated = increment @ current
            updated[np.isclose(updated, 0, atol=1e-12)] = 0
            updated[np.isclose(updated, 1, atol=1e-12)] = 1
            updated[np.isclose(updated, -1, atol=1e-12)] = -1
            self._layer.rotate = updated

            centre_after = np.asarray(
                self._layer.data_to_world(tuple(centre_data)), dtype=float
            )
            self._layer.translate = (
                np.asarray(self._layer.translate, dtype=float)
                + centre_before
                - centre_after
            )
            self._sync_from_layer()
        except Exception as exc:
            _LOG.warning(f"napari-storm: could not rotate reference image: {exc}")

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def _on_remove(self):
        """Remove the napari layer and detach this widget from the layout."""
        # 1. Remove napari layer
        try:
            self._dock_widget.viewer.layers.remove(self._layer)
        except (KeyError, ValueError):
            pass

        # 2. Remove from dock_widget's tracking list (before Qt deletes self)
        try:
            self._dock_widget.image_layer_controls.remove(self)
        except ValueError:
            pass

        # 3. Remove row from parent QFormLayout (also deletes the widget Qt-side)
        parent = self.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        if parent_layout is not None and hasattr(parent_layout, "removeRow"):
            parent_layout.removeRow(self)
        else:
            self.setParent(None)
            self.deleteLater()
