import math

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)
from superqt import QDoubleRangeSlider

from .CustomErrors import ParentError, StaticAttributeError


class ChannelControls(QWidget):
    """A QT widget for every channel,
    providing visual controls"""

    # how many log‐units the “max” handle spans (10^-LOG_RANGE … 10^LOG_RANGE)
    LOG_RANGE = 2

    def __init__(
        self,
        parent,
        name,
        channel_index,
        localization_datasets=None,
        data_to_layer_itf=None,
        z_color_encoding_mode=None,
        render_gaussian_mode=None,
    ):
        super().__init__()
        self._parent = parent
        self._name = name
        self._channel_index = channel_index

        # Keep track of show/hide and opacity state
        self.show_channel_state = True
        self.opacity_slider_setting = 0.0
        self.colormap_index = 0

        # Data interfaces
        self.localization_datasets = (
            localization_datasets or parent.localization_datasets
        )
        # Dataset identity is stable while list indices are not: unloading an
        # earlier channel shifts every later index.  Controls therefore retain
        # the dataset they were created for instead of repeatedly indexing the
        # parent's mutable list.
        self.dataset = self.localization_datasets[channel_index]
        self.data_to_layer_itf = data_to_layer_itf or parent.data_to_layer_itf

        # ── Title, Reset, Show/Hide ──
        self.Label = QLabel(name)
        self.Bshow_channel = QCheckBox()
        self.Bshow_channel.setChecked(True)
        self.Bshow_channel.stateChanged.connect(self.show_channel)

        self.Breset = QPushButton("Reset")
        self.Breset.clicked.connect(self.reset)

        self.Bunload = QPushButton("Unload")
        self.Bunload.setToolTip(
            "Remove this localization dataset and its layer from the viewer"
        )
        self.Bunload.clicked.connect(self._unload_dataset)

        # ── Colormap selector ──
        self.Colormap_selector = QComboBox()
        items = [cmap.name for cmap in self.data_to_layer_itf.colormap]
        self.Colormap_selector.addItems(items)
        self.Colormap_selector.setIconSize(QtCore.QSize(32, 32))
        for i, pix in enumerate(self.data_to_layer_itf.colormap_icons):
            self.Colormap_selector.setItemIcon(i, QIcon(pix))
        self.Colormap_selector.setCurrentIndex(channel_index)
        self.Colormap_selector.currentIndexChanged.connect(self.change_color_map)

        # ── Contrast slider & spinboxes ──
        # Cache the original data range once.  A degenerate range is normal --
        # fixed-Gaussian mode gives every localization the value 1.0 -- and
        # dividing by its zero span produced NaN slider positions and a
        # runaway signal loop between the slider and the spin boxes.
        self._orig_min, self._orig_max = self.data_to_layer_itf.value_range_of(
            self.dataset
        )
        if not self._orig_max > self._orig_min:
            self._orig_max = self._orig_min + 1.0
        self._orig_span = self._orig_max - self._orig_min

        # two‐handle slider: left = linear cutoff, right = log‐pos of max
        self.Slider_colormap_range = QDoubleRangeSlider(self)
        self.Slider_colormap_range.setOrientation(Qt.Horizontal)
        self.Slider_colormap_range.setRange(0.0, 1.0)
        self.Slider_colormap_range.setValue((0.0, 0.5))
        self.Slider_colormap_range.valueChanged.connect(
            self._on_contrast_slider_changed
        )

        # cutoff spinbox (absolute units)
        self.cutoff_spin = QDoubleSpinBox(self)
        self.cutoff_spin.setRange(self._orig_min, self._orig_max)
        self.cutoff_spin.setSingleStep((self._orig_max - self._orig_min) / 100.0)
        self.cutoff_spin.setValue(self._orig_min)
        self.cutoff_spin.valueChanged.connect(self._on_cutoff_spin_changed)

        # factor spinbox (log scale)
        min_factor = 10**-self.LOG_RANGE
        max_factor = 10**self.LOG_RANGE
        self.factor_spin = QDoubleSpinBox(self)
        self.factor_spin.setRange(min_factor, max_factor)
        self.factor_spin.setSingleStep(0.1)
        self.factor_spin.setDecimals(3)
        self.factor_spin.setValue(1.0)
        self.factor_spin.valueChanged.connect(self._on_factor_spin_changed)

        # ── Opacity slider ──
        self.Slider_opacity = QSlider(Qt.Horizontal)
        self.Slider_opacity.setRange(0, 100)
        self.Slider_opacity.setValue(100)
        self.Slider_opacity.hide()
        self.Slider_opacity.valueChanged.connect(self.adjust_z_color_encoding_opacity)

        # ── Layout ──
        layout = QGridLayout(self)
        layout.addWidget(self.Label, 0, 0)
        layout.addWidget(self.Breset, 0, 1)
        layout.addWidget(self.Bshow_channel, 0, 2)
        layout.addWidget(self.Bunload, 0, 3)
        layout.addWidget(self.Colormap_selector, 1, 0, 1, 4)
        layout.addWidget(self.Slider_colormap_range, 2, 0, 1, 4)

        spin_row = QWidget(self)
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        spin_row.setLayout(h)
        h.addWidget(QLabel("Cutoff"))
        h.addWidget(self.cutoff_spin)
        h.addStretch()
        h.addWidget(QLabel("×Range"))
        h.addWidget(self.factor_spin)
        layout.addWidget(spin_row, 3, 0, 1, 4)

        layout.addWidget(self.Slider_opacity, 4, 0, 1, 4)

        # Manual alignment (§7.4): numeric entry with a reversible reset.
        # These write a WorldTransform through the store, which announces it;
        # the renderer re-plans on the event.  The measurements are never
        # touched -- a transform moves what is *drawn*.
        self._shift_spins = {}
        shift_row = QWidget()
        shift_layout = QGridLayout(shift_row)
        shift_layout.setContentsMargins(0, 0, 0, 0)
        shift_layout.addWidget(QLabel("Shift [µm]:"), 0, 0)
        for column, axis in enumerate(("x", "y", "z"), start=1):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1e6, 1e6)
            spin.setSingleStep(0.1)
            spin.setPrefix(f"{axis} ")
            spin.setToolTip(
                f"Move this dataset along {axis} without altering the "
                "localizations themselves."
            )
            spin.valueChanged.connect(self._on_shift_changed)
            self._shift_spins[axis] = spin
            shift_layout.addWidget(spin, 0, column)
        self.Breset_shift = QPushButton("Reset")
        self.Breset_shift.setToolTip("Return this dataset to its measured position.")
        self.Breset_shift.clicked.connect(self.reset_shift)
        shift_layout.addWidget(self.Breset_shift, 0, 4)
        layout.addWidget(shift_row, 5, 0, 1, 4)
        layout.setColumnStretch(0, 3)
        self.setLayout(layout)

    @property
    def z_color_encoding_mode(self):
        # proxy to the parent widget's flag
        return self._parent.z_color_encoding_mode

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        self._parent.z_color_encoding_mode = value

    @property
    def render_gaussian_mode(self):
        return self._parent.render_gaussian_mode

    @render_gaussian_mode.setter
    def render_gaussian_mode(self, value):
        raise ParentError("Should be set in Parent not here")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise StaticAttributeError("Name of Channel should not be changed!")

    @property
    def channel_index(self):
        return self._channel_index

    @channel_index.setter
    def channel_index(self, value):
        raise StaticAttributeError("Channel index should not be changed!")

    def adjust_colormap_range(self):
        """Initialize/apply contrast for create_layer() hook."""
        vals = self.Slider_colormap_range.value()
        self._on_contrast_slider_changed(vals)

    def _unload_dataset(self):
        self._parent.unload_dataset(self.dataset)

    def _on_contrast_slider_changed(self, vals):
        """Slider moved: compute/apply cutoff & max, then sync spins."""
        s_cutoff, s_logpos = vals
        data_min = self._orig_min

        cutoff = data_min + s_cutoff * self._orig_span
        logv = -self.LOG_RANGE + s_logpos * (2 * self.LOG_RANGE)
        factor = 10**logv
        maxval = data_min + factor * self._orig_span
        if cutoff > maxval:
            cutoff = maxval

        self.data_to_layer_itf.set_appearance(
            self.dataset, contrast_limits=(cutoff, maxval)
        )

        # sync spinboxes
        self.cutoff_spin.blockSignals(True)
        self.factor_spin.blockSignals(True)
        self.cutoff_spin.setValue(cutoff)
        self.factor_spin.setValue(factor)
        self.cutoff_spin.blockSignals(False)
        self.factor_spin.blockSignals(False)

    def _on_cutoff_spin_changed(self, val):
        """User edited cutoff: update slider & reapply."""
        s_cut = (val - self._orig_min) / self._orig_span
        lo, hi = self.Slider_colormap_range.value()
        self.Slider_colormap_range.blockSignals(True)
        self.Slider_colormap_range.setValue((s_cut, hi))
        self.Slider_colormap_range.blockSignals(False)
        self._on_contrast_slider_changed((s_cut, hi))

    def _on_factor_spin_changed(self, val):
        """User edited factor: update slider & reapply."""
        logv = math.log10(val)
        s_log = (logv + self.LOG_RANGE) / (2 * self.LOG_RANGE)
        lo, hi = self.Slider_colormap_range.value()
        self.Slider_colormap_range.blockSignals(True)
        self.Slider_colormap_range.setValue((lo, s_log))
        self.Slider_colormap_range.blockSignals(False)
        self._on_contrast_slider_changed((lo, s_log))

    def change_color_map(self):
        """Switch between HSV and the selected colormap."""
        idx = self.Colormap_selector.currentIndex()
        self.data_to_layer_itf.set_appearance(
            self.dataset,
            colormap=(
                "hsv"
                if self.z_color_encoding_mode
                else self.data_to_layer_itf.colormap[idx]
            ),
        )

    def adjust_z_color_encoding_opacity(self):
        self.data_to_layer_itf.set_appearance(
            self.dataset, opacity=self.Slider_opacity.value() / 100.0
        )

    def reset(self):
        """Reset to cutoff=0, factor=1, opacity=100%."""
        self.Slider_colormap_range.setValue((0.0, 0.5))
        self.cutoff_spin.setValue(self._orig_min)
        self.factor_spin.setValue(1.0)
        self.Slider_opacity.setValue(100)

    # ------------------------------------------------------------------
    # Manual alignment
    # ------------------------------------------------------------------

    def _store(self):
        return getattr(self._parent, "dataset_store", None)

    def _on_shift_changed(self, *_):
        """Push the spin boxes into the dataset's WorldTransform."""
        from napari_storm.core import WorldTransform

        store = self._store()
        if store is None:
            return
        dataset_id = getattr(self.dataset, "dataset_id", None)
        if dataset_id is None or store.state(dataset_id) is None:
            return
        translation = tuple(
            self._shift_spins[axis].value() * 1000.0 for axis in ("x", "y", "z")
        )
        current = store.state(dataset_id).transform
        store.set_transform(
            dataset_id, WorldTransform(scale=current.scale, translation_nm=translation)
        )

    def reset_shift(self):
        """Back to the measured position, reversibly and in one place."""
        for spin in self._shift_spins.values():
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._on_shift_changed()

    def show_channel(self):
        # This is the checkbox's own stateChanged slot, and it sets the
        # checkbox below.  Called from a user click the write is a no-op, but
        # called programmatically it re-emits stateChanged and re-enters here,
        # toggling forever.
        self.Bshow_channel.blockSignals(True)
        try:
            self._apply_show_channel()
        finally:
            self.Bshow_channel.blockSignals(False)

    def _apply_show_channel(self):
        if self.show_channel_state:
            # hide channel
            self.show_channel_state = False
            self.Bshow_channel.setChecked(False)
            self.data_to_layer_itf.set_appearance(self.dataset, opacity=0.0)
            self.Slider_colormap_range.hide()
            self.cutoff_spin.hide()
            self.factor_spin.hide()
            self.Colormap_selector.hide()
            self.Slider_opacity.hide()
        else:
            # show channel
            self.show_channel_state = True
            self.Bshow_channel.setChecked(True)
            if self.z_color_encoding_mode:
                self.Slider_opacity.show()
                self.Slider_colormap_range.hide()
                self.cutoff_spin.hide()
                self.factor_spin.hide()
                self.Colormap_selector.hide()
            else:
                self.data_to_layer_itf.set_appearance(self.dataset, opacity=1.0)
                self.Slider_colormap_range.show()
                self.cutoff_spin.show()
                self.factor_spin.show()
                self.Colormap_selector.show()
                self.Slider_opacity.hide()
