"""
Dialog for importing a reference image with pixel size, orientation and offset
settings.  Returns an ImageImportResult dataclass consumed by _dock_widget and
ImageLayerControls.
"""

import os
from dataclasses import dataclass

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImageImportResult:
    """All parameters chosen by the user in ImageImportDialog."""

    file_path: str
    img: np.ndarray  # raw array (before orientation expansion)
    layer_name: str
    orientation: str  # 'XY', 'XZ', 'YZ', or '3D'
    px_xy_nm: float
    px_z_nm: float
    # offsets in nm along napari axes: dim0=z, dim1=x_pos_nm, dim2=y_pos_nm
    x_off_nm: float
    y_off_nm: float
    z_off_nm: float


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image(file_path: str) -> np.ndarray:
    """Load image from *file_path* using tifffile or PIL."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import tifffile

            return tifffile.imread(file_path)
        except ImportError:
            pass
    try:
        from PIL import Image as _PIL

        return np.array(_PIL.open(file_path))
    except ImportError:
        pass
    # Last resort — tifffile (will raise ImportError if not installed)
    import tifffile

    return tifffile.imread(file_path)


# ---------------------------------------------------------------------------
# Image type helpers
# ---------------------------------------------------------------------------


def _is_rgb(img: np.ndarray) -> bool:
    """True when the last axis is an RGB/RGBA channel (size 3 or 4)."""
    return img.ndim >= 3 and img.shape[-1] in (3, 4)


def _spatial_ndim(img: np.ndarray) -> int:
    """Number of spatial dimensions (last dim stripped when it is a channel)."""
    return img.ndim - 1 if _is_rgb(img) else img.ndim


def _image_info_text(img: np.ndarray) -> str:
    if _is_rgb(img):
        ch = "RGBA" if img.shape[-1] == 4 else "RGB"
        dims = " × ".join(str(s) for s in img.shape[:-1])
        return f"{dims}, {ch}"
    dims = " × ".join(str(s) for s in img.shape)
    ndim = f"{img.ndim}D"
    return f"{dims}, {ndim} grayscale"


# ---------------------------------------------------------------------------
# TIFF metadata extraction
# ---------------------------------------------------------------------------


def _to_nm(value, unit: str):
    """Convert *value* from *unit* to nanometres; return None if unknown."""
    if value is None:
        return None
    unit = (unit or "µm").strip()
    table = {
        "nm": 1.0,
        "nanometer": 1.0,
        "nanometre": 1.0,
        "µm": 1e3,
        "um": 1e3,
        "micron": 1e3,
        "micrometer": 1e3,
        "micrometre": 1e3,
        "mm": 1e6,
        "millimeter": 1e6,
        "millimetre": 1e6,
        "cm": 1e7,
        "centimeter": 1e7,
        "centimetre": 1e7,
        "m": 1e9,
        "meter": 1e9,
        "metre": 1e9,
    }
    factor = table.get(unit.lower())
    if factor is None:
        return None
    return float(value) * factor


def _res_tag_to_nm(num, den, res_unit_int, imagej_unit=None):
    """
    Convert an XResolution TIFF tag (pixels-per-unit fraction) to nm/pixel.
    *res_unit_int*: 1=unitless, 2=inch, 3=cm.
    *imagej_unit*: optional unit string from ImageJ metadata (overrides).
    """
    if den == 0 or num == 0:
        return None
    ppu = num / den  # pixels per unit
    if imagej_unit:
        nm = _to_nm(1.0 / ppu, imagej_unit)
        if nm is not None:
            return nm
    if res_unit_int == 3:  # cm
        return 1e7 / ppu
    if res_unit_int == 2:  # inch
        return 25_400_000.0 / ppu
    return None


def _try_read_ome_position_nm(file_path: str):
    """Return ``(x_nm, y_nm, z_nm)`` from OME Plane positions, or ``None``.

    The companion to the pixel size: `PhysicalSize` says how big a pixel is and
    `Position` says where the image sits. Only the first plane is read -- a
    stack's remaining planes differ only in depth, and the layer is placed by
    its origin.

    The axis names are world axes, not array axes, so no transposition question
    arises here: OME `PositionX` is a world x whatever order the samples are
    stored in, and this plugin's `x_off_nm` is the same world x.
    """
    try:
        import xml.etree.ElementTree as ET

        import tifffile

        with tifffile.TiffFile(file_path) as handle:
            xml = handle.ome_metadata
        if not xml:
            return None
        root = ET.fromstring(xml)
        namespace = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        prefix = f"{{{namespace}}}" if namespace else ""
        plane = root.find(f".//{prefix}Plane")
        if plane is None:
            return None

        def _axis(name):
            raw = plane.get(f"Position{name}")
            if raw is None:
                return None
            return _to_nm(float(raw), plane.get(f"Position{name}Unit", "µm"))

        x_nm, y_nm, z_nm = _axis("X"), _axis("Y"), _axis("Z")
        if x_nm is None and y_nm is None and z_nm is None:
            return None
        return (x_nm or 0.0, y_nm or 0.0, z_nm or 0.0)
    except Exception:
        # Placement metadata is a convenience; a malformed file must still be
        # importable with the offsets the user types in.
        return None


def _try_read_tiff_pixel_size(file_path: str):
    """
    Return (px_xy_nm, px_z_nm) extracted from TIFF metadata.
    Either value may be None if not found.
    """
    try:
        import tifffile

        with tifffile.TiffFile(file_path) as tif:
            # ── OME-TIFF ────────────────────────────────────────────────
            if tif.ome_metadata:
                try:
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(tif.ome_metadata)
                    ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
                    ns_prefix = f"{{{ns}}}" if ns else ""
                    pixels = root.find(f".//{ns_prefix}Pixels")
                    # Some OME files wrap in a list
                    if pixels is None and len(root):
                        pixels = root[0].find(f".//{ns_prefix}Pixels")
                    if pixels is not None:
                        px_x = float(pixels.get("PhysicalSizeX") or 0)
                        px_z = float(pixels.get("PhysicalSizeZ") or 0)
                        ux = pixels.get("PhysicalSizeXUnit", "µm")
                        uz = pixels.get("PhysicalSizeZUnit", "µm")
                        if px_x > 0:
                            return _to_nm(px_x, ux), (
                                _to_nm(px_z, uz) if px_z > 0 else None
                            )
                except Exception:
                    pass

            # ── ImageJ TIFF ─────────────────────────────────────────────
            if tif.is_imagej:
                meta = tif.imagej_metadata or {}
                spacing = meta.get("spacing")
                ij_unit = meta.get("unit", "µm")
                page = tif.pages[0]
                x_res = page.tags.get("XResolution")
                res_unit = page.tags.get("ResolutionUnit")
                if x_res is not None:
                    val = x_res.value
                    num, den = (
                        (val[0], val[1]) if hasattr(val, "__len__") else (val, 1)
                    )
                    ru = res_unit.value if res_unit else 2
                    nm_per_px = _res_tag_to_nm(num, den, ru, ij_unit)
                    if nm_per_px:
                        px_z_nm = _to_nm(spacing, ij_unit) if spacing else None
                        return nm_per_px, px_z_nm

            # ── Plain TIFF resolution tags ───────────────────────────────
            page = tif.pages[0]
            x_res = page.tags.get("XResolution")
            res_unit = page.tags.get("ResolutionUnit")
            if x_res is not None:
                val = x_res.value
                num, den = (
                    (val[0], val[1]) if hasattr(val, "__len__") else (val, 1)
                )
                ru = res_unit.value if res_unit else 2
                nm_per_px = _res_tag_to_nm(num, den, ru)
                if nm_per_px:
                    return nm_per_px, None

    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Spinbox factory
# ---------------------------------------------------------------------------


def _make_offset_spinbox():
    """Return a QDoubleSpinBox for µm offsets."""
    sb = QDoubleSpinBox()
    sb.setRange(-100_000.0, 100_000.0)
    sb.setDecimals(3)
    sb.setSuffix(" µm")
    sb.setValue(0.0)
    return sb


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class ImageImportDialog(QDialog):
    """
    Modal dialog that lets the user choose pixel size, orientation and
    spatial offset for a reference image before importing it into napari.

    If *file_path* is given the dialog pre-loads the file; otherwise the
    user must click "Browse…".
    """

    _ORIENT_2D = ["XY", "XZ", "YZ"]
    _ORIENT_3D = ["3D"]

    def __init__(self, file_path=None, parent=None, default_z_off_nm=0.0):
        super().__init__(parent)
        self.setWindowTitle("Import Reference Image")
        self.setMinimumWidth(400)

        self._img: np.ndarray = None
        self._file_path: str = None

        outer = QVBoxLayout()
        outer.setSpacing(10)

        # ── File row ─────────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("No file selected…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(browse_btn)
        outer.addLayout(file_row)

        # ── Image info label ─────────────────────────────────────────────
        self._info_lbl = QLabel("–")
        self._info_lbl.setStyleSheet(
            "color: rgb(130, 142, 154); font-size: 11px; font-style: italic;"
        )
        outer.addWidget(self._info_lbl)

        # ── Form ─────────────────────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(6)

        self._px_xy_spin = QDoubleSpinBox()
        self._px_xy_spin.setRange(0.1, 100_000.0)
        self._px_xy_spin.setDecimals(2)
        self._px_xy_spin.setSuffix(" nm")
        self._px_xy_spin.setValue(100.0)
        self._px_xy_spin.setToolTip("Lateral (XY) pixel size in nanometres")
        form.addRow("Pixel size XY:", self._px_xy_spin)

        self._px_z_spin = QDoubleSpinBox()
        self._px_z_spin.setRange(0.1, 100_000.0)
        self._px_z_spin.setDecimals(2)
        self._px_z_spin.setSuffix(" nm")
        self._px_z_spin.setValue(200.0)
        self._px_z_spin.setEnabled(False)
        self._px_z_spin.setToolTip(
            "Axial (Z) voxel size in nanometres — only for 3D stacks and XZ/YZ planes"
        )
        form.addRow("Pixel size Z:", self._px_z_spin)

        self._orient_combo = QComboBox()
        self._orient_combo.addItems(self._ORIENT_2D)
        self._orient_combo.setToolTip(
            "How the 2D image is oriented in 3D space.\n"
            "XY = en-face (z-slab), XZ = side view (constant y), YZ = side view (constant x)"
        )
        self._orient_combo.currentTextChanged.connect(self._on_orient_changed)
        form.addRow("Orientation:", self._orient_combo)

        self._x_off_spin = _make_offset_spinbox()
        self._y_off_spin = _make_offset_spinbox()
        self._z_off_spin = _make_offset_spinbox()
        # Pre-placed on the centre of the localizations' depth range, so a flat
        # dataset and its reference image share one plane and napari's 2-D
        # display can show both.  It is a starting value in an editable field,
        # not a rule: the user can change it here, and nothing moves it later.
        self._z_off_spin.setValue(float(default_z_off_nm) / 1000.0)

        self._x_off_lbl = QLabel("X offset:")
        self._y_off_lbl = QLabel("Y offset:")
        self._z_off_lbl = QLabel("Z offset:")

        form.addRow(self._x_off_lbl, self._x_off_spin)
        form.addRow(self._y_off_lbl, self._y_off_spin)
        form.addRow(self._z_off_lbl, self._z_off_spin)

        outer.addLayout(form)

        # ── Dialog buttons ───────────────────────────────────────────────
        self._btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._btn_box.button(QDialogButtonBox.Ok).setText("Import")
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._btn_box.accepted.connect(self.accept)
        self._btn_box.rejected.connect(self.reject)
        outer.addWidget(self._btn_box)

        self.setLayout(outer)

        if file_path:
            self._load_file(file_path)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Reference Image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            img = _load_image(path)
        except Exception as exc:
            self._info_lbl.setText(f"Error loading file: {exc}")
            return

        self._img = img
        self._file_path = path
        self._file_edit.setText(path)
        self._info_lbl.setText(_image_info_text(img))
        self._btn_box.button(QDialogButtonBox.Ok).setEnabled(True)

        spatial = _spatial_ndim(img)
        is_rgb_img = _is_rgb(img)

        self._orient_combo.blockSignals(True)
        self._orient_combo.clear()
        if spatial >= 3:
            self._orient_combo.addItems(self._ORIENT_3D)
            self._orient_combo.setEnabled(False)
            self._px_z_spin.setEnabled(True)
        elif is_rgb_img:
            # XZ/YZ would corrupt the channel axis — restrict to XY
            self._orient_combo.addItem("XY")
            self._orient_combo.setEnabled(False)
            self._px_z_spin.setEnabled(False)
        else:
            self._orient_combo.addItems(self._ORIENT_2D)
            self._orient_combo.setEnabled(True)
            self._px_z_spin.setEnabled(False)
        self._orient_combo.blockSignals(False)
        self._on_orient_changed(self._orient_combo.currentText())

        # Auto-fill pixel size from TIFF metadata
        if path.lower().endswith((".tif", ".tiff")):
            px_xy, px_z = _try_read_tiff_pixel_size(path)
            if px_xy:
                self._px_xy_spin.setValue(px_xy)
            if px_z:
                self._px_z_spin.setValue(px_z)

            # Where the file says it belongs.  An OME-TIFF this plugin exported
            # carries its own world position, so re-importing one puts it back
            # where it was cut from instead of at the origin.
            position = _try_read_ome_position_nm(path)
            if position is not None:
                x_nm, y_nm, z_nm = position
                self._x_off_spin.setValue(x_nm / 1000.0)
                self._y_off_spin.setValue(y_nm / 1000.0)
                self._z_off_spin.setValue(z_nm / 1000.0)

    # ------------------------------------------------------------------
    # Orientation change
    # ------------------------------------------------------------------

    def _on_orient_changed(self, orientation: str):
        needs_z = orientation in ("XZ", "YZ", "3D")
        self._px_z_spin.setEnabled(needs_z)

        if orientation == "XZ":
            # Constant-Y plane: Y spinbox controls plane position in Y
            self._x_off_lbl.setText("X offset:")
            self._y_off_lbl.setText("Y position:")
            self._z_off_lbl.setText("Z offset:")
        elif orientation == "YZ":
            # Constant-X plane: X spinbox controls plane position in X
            self._x_off_lbl.setText("X position:")
            self._y_off_lbl.setText("Y offset:")
            self._z_off_lbl.setText("Z offset:")
        else:
            self._x_off_lbl.setText("X offset:")
            self._y_off_lbl.setText("Y offset:")
            self._z_off_lbl.setText("Z offset:")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_result(self) -> ImageImportResult:
        """Return ImageImportResult or None if no file was loaded."""
        if self._img is None or self._file_path is None:
            return None
        return ImageImportResult(
            file_path=self._file_path,
            img=self._img,
            layer_name=os.path.basename(self._file_path),
            orientation=self._orient_combo.currentText(),
            px_xy_nm=self._px_xy_spin.value(),
            px_z_nm=self._px_z_spin.value(),
            x_off_nm=self._x_off_spin.value() * 1000.0,  # µm → nm
            y_off_nm=self._y_off_spin.value() * 1000.0,
            z_off_nm=self._z_off_spin.value() * 1000.0,
        )
