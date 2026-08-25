"""Choosing what to export, with the consequences shown before it happens.

The dialog's job is not to collect three numbers. It is to make the two things
§4.1 insists on visible *while the user is still deciding*:

* **What this pixel size costs.** The requested pixel size is honoured exactly,
  which means a wide field at a fine sampling produces a very large file. That
  is the intended behaviour, so the size is shown live rather than discovered
  when the disk fills.
* **That the file will not match the screen**, when the render budget has
  thinned the view. The export contains more localizations than are being
  drawn, and saying so afterwards would be too late to be useful.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox,
                            QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPushButton, QRadioButton, QVBoxLayout)

from ..image_export import (SCOPE_CURRENT_VIEW, SCOPE_EVERYTHING,
                            ExportOptions)

__all__ = ["ExportImageDialog", "format_size", "describe_plan"]


def format_size(nbytes):
    """Human-readable file size.  Exports span kilobytes to hundreds of GB."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def describe_plan(plan):
    """One line of what will be written, for the dialog's summary."""
    _channels, nz, ny, nx = plan.shape
    dimensions = f"{nx:,} x {ny:,}" + (f" x {nz:,}" if nz > 1 else "")
    return (
        f"{dimensions} px, {len(plan.channels)} channel"
        f"{'s' if len(plan.channels) != 1 else ''}, "
        f"{format_size(plan.nbytes)}, "
        f"{plan.n_localizations:,} localizations"
    )


class ExportImageDialog(QDialog):
    """Pixel size, 2-D or 3-D, and where the file goes.

    *plan_for* is ``callable(ExportOptions) -> ExportPlan``, called on every
    change so the summary and warnings describe the current choice rather than
    the one the dialog opened with.
    """

    def __init__(self, plan_for, parent=None, default_pixel_size_nm=10.0):
        super().__init__(parent)
        self.setWindowTitle("Export OME-TIFF")
        self._plan_for = plan_for
        self._plan = None

        form = QFormLayout()

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(3)
        self.pixel_size.setRange(0.001, 100_000.0)
        self.pixel_size.setValue(float(default_pixel_size_nm))
        self.pixel_size.setSuffix(" nm")
        # No "fit to a sensible size" option on purpose: never downsample means
        # the pixel size is the user's, and the file size follows from it.
        self.pixel_size.setToolTip(
            "Exact size of one output pixel. The image is never downsampled to "
            "fit, so a smaller pixel size means a larger file."
        )
        form.addRow("Pixel size:", self.pixel_size)

        self.current_view = QRadioButton("Current view (2-D projection)")
        self.everything = QRadioButton("Everything (3-D stack)")
        self.current_view.setChecked(True)
        scope = QButtonGroup(self)
        scope.addButton(self.current_view)
        scope.addButton(self.everything)
        form.addRow("Export:", self.current_view)
        form.addRow("", self.everything)

        self.z_step = QDoubleSpinBox()
        self.z_step.setDecimals(3)
        self.z_step.setRange(0.001, 100_000.0)
        self.z_step.setValue(50.0)
        self.z_step.setSuffix(" nm")
        self.z_step.setEnabled(False)
        self.z_step.setToolTip(
            "Spacing between slices. Separate from the lateral pixel size "
            "because axial and lateral resolution are not the same."
        )
        form.addRow("Z step:", self.z_step)

        path_row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("reconstruction.ome.tif")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path)
        path_row.addWidget(browse)
        form.addRow("Save to:", path_row)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: palette(link);")
        self.warning.hide()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._save_button = buttons.button(QDialogButtonBox.Save)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.summary)
        layout.addWidget(self.warning)
        layout.addWidget(buttons)

        self.pixel_size.valueChanged.connect(self._refresh)
        self.z_step.valueChanged.connect(self._refresh)
        self.current_view.toggled.connect(self._on_scope_changed)
        self.path.textChanged.connect(self._refresh_button)
        self._on_scope_changed()

    # ------------------------------------------------------------------

    def options(self):
        return ExportOptions(
            pixel_size_nm=float(self.pixel_size.value()),
            scope=SCOPE_CURRENT_VIEW if self.current_view.isChecked() else SCOPE_EVERYTHING,
            z_step_nm=float(self.z_step.value()),
        )

    def output_path(self):
        return self.path.text().strip()

    def plan(self):
        """The plan the summary is describing, or None if it could not be built."""
        return self._plan

    # ------------------------------------------------------------------

    def _on_scope_changed(self, *_):
        self.z_step.setEnabled(self.everything.isChecked())
        self._refresh()

    def _browse(self):
        from qtpy.QtWidgets import QFileDialog

        chosen, _filter = QFileDialog.getSaveFileName(
            self, "Export OME-TIFF", self.path.text(), "OME-TIFF (*.ome.tif)"
        )
        if chosen:
            self.path.setText(chosen)

    def _refresh(self, *_):
        try:
            self._plan = self._plan_for(self.options())
        except Exception as error:  # noqa: BLE001 - shown, not raised at the user
            self._plan = None
            self.summary.setText(f"Cannot export: {error}")
            self.warning.hide()
            self._refresh_button()
            return

        self.summary.setText(describe_plan(self._plan))
        messages = [w.message for w in self._plan.warnings]
        if messages:
            self.warning.setText("\n\n".join(messages))
            self.warning.show()
        else:
            self.warning.hide()
        self._refresh_button()

    def _refresh_button(self, *_):
        self._save_button.setEnabled(
            self._plan is not None and bool(self.output_path())
        )
