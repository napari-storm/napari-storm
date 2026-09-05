import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from qtpy import QtCore
from qtpy.QtWidgets import QComboBox, QFormLayout, QPushButton, QSpinBox, QWidget

from .core import DatasetClosed, StoreCleared
from .CustomErrors import ParentError
from .pyqt.filter_slider import RangeSlider3, RangeSlider4


class DataFilterWindow(QWidget):
    """GUI Elements for the data filter widget"""

    def __init__(self, parent):
        super().__init__()

        self.setWindowTitle("Data Filter")

        self.layout = QFormLayout()

        self.parent = parent

        self.Cdatasets = QComboBox()
        self.layout.addRow(self.Cdatasets)

        self.Cparameter = QComboBox()

        self.layout.addRow(self.Cparameter)

        self.CanvasWidget = ParameterHistogrammCanvas(parent=self)
        self.layout.addRow(self.CanvasWidget)

        self.Cfilter_mode = QComboBox()

        self.Bapply_filter = QPushButton()
        self.Bapply_filter.setText("Apply filter to current dataset")

        self.Bapply_filter_to_all = QPushButton()
        self.Bapply_filter_to_all.setText("Apply filter to all datasets")

        self.Breset_filters = QPushButton()
        self.Breset_filters.setText("Reset all filtering")

        self.Sfilter_slider = RangeSlider3(canvas=self.CanvasWidget.Canvas)

        self.Sxrange = RangeSlider4(canvas=self.CanvasWidget.Canvas)

        self.layout.addRow("Adjust graph range:", QWidget())
        self.layout.addRow(self.Sxrange)
        self.layout.addRow("Filter mode:", self.Cfilter_mode)
        self.layout.addRow(self.Sfilter_slider)
        self.layout.addRow(self.Bapply_filter)
        self.layout.addRow(self.Bapply_filter_to_all)
        self.layout.addRow(self.Breset_filters)

        self.SB_nbins = QSpinBox()
        self.SB_nbins.setRange(1, 1000)
        self.SB_nbins.setValue(100)
        self.layout.addRow("Number of bins: ", self.SB_nbins)

        self.setLayout(self.layout)

        self.mouseReleaseEvent = self.reset_render_range_before_filtering

    def reset_render_range_before_filtering(self):
        self.parent.reset_render_range()

    def filter_mode_changed(self):
        self.filter_mode_active_idx = self.Cfilter_mode.currentIndex()

    def clear_entries(self):
        self.Cparameter.clear()
        self.Cdatasets.clear()


class ParameterHistogrammCanvas(QWidget):
    """Multipurpose canvas that is used in data filter widget"""

    def __init__(self, parent):
        super().__init__()
        # Attributes
        self.parent = parent

        # GUI
        self.layout = QFormLayout()

        self.Canvas = MplCanvas()
        self.layout.addRow(self.Canvas)

        self.setLayout(self.layout)

    def draw(self, dataset, parameter, bins, slider_values_decimal=(0, 1)):
        self.Canvas.reinitialise()
        # locs_active is a read-only view of the canonical table, so replace the
        # non-finite entries in a throwaway copy rather than in place.  The old
        # in-place write silently edited the dataset just by drawing a histogram.
        data = np.asarray(getattr(dataset.locs_active, parameter))
        data = np.where(np.isfinite(data), data, 0)
        if data.size == 0:
            # Every localization is filtered out.  There is no distribution to
            # draw, and np.min/np.max on an empty array raises.
            self.Canvas.ax.set_xlabel(parameter)
            self.Canvas.ax.set_ylabel("#")
            self.Canvas.draw()
            return
        self.Canvas.ax.hist(data, bins=bins, facecolor="white")
        self.Canvas.ax.set_xlabel(parameter)
        self.Canvas.ax.set_ylabel("#")
        ylim = self.Canvas.ax.get_ylim()
        self.Canvas.ax.set_ylim(ylim)
        self.set_xrange(data=data, slider_values_decimal=slider_values_decimal)
        self.Canvas.draw()

    def set_xrange(self, data, slider_values_decimal):
        tmp_xrange_data = [np.min(data), np.max(data)]
        tmp_absolute_xrange = tmp_xrange_data[1] - tmp_xrange_data[0]
        xrange = [
            slider_values_decimal[0] * tmp_absolute_xrange + tmp_xrange_data[0],
            slider_values_decimal[1] * tmp_absolute_xrange + tmp_xrange_data[0],
        ]
        self.Canvas.ax.set_xlim(xrange)


class MplCanvas(Canvas):
    """Multipurpose canvas"""

    def __init__(self):
        self.fig = Figure()
        self.fig.set_facecolor("#262930")
        self.ax = self.fig.subplots()
        self.ax.set_facecolor("#262930")
        Canvas.__init__(self, self.fig)
        # Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Canvas.updateGeometry(self)
        self.fig.subplots_adjust(0.2, 0.2, 0.9, 0.9)
        # self.fig.tight_layout()

    def reinitialise(self):
        if self.fig.axes:
            for ax in self.fig.axes:
                ax.cla()
            self.ax = self.fig.gca()
            self.fig.set_facecolor("#262930")
            self.ax.set_facecolor("#262930")
            self.draw()


class DataFilterInterface:
    """Core code of the data filtering functions"""

    def __init__(self, parent, data_filter_window):
        self._parent = parent
        self.dfw = data_filter_window

        self.active_filters = (
            []
        )  # create a dictionary of all active filters including the "normal" render range,

        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_filterable_parameters = []
        self.n_bins = 100
        self.filter_slider_values_decimal = [0, 1]
        self.filter_modes = ["Bandpass", "Bandstop"]
        self.filter_mode_active_idx = 0
        # Removal indices per dataset, keyed by stable dataset id.  This was a
        # list indexed alongside the dataset list, grown with a
        # `while len - 1 <= idx: append` loop and popped by position on unload:
        # one misordered call and a dataset inherited its neighbour's filter.
        self.filter_indices = {}
        self.xrange_slider_values_decimal = (0, 1)

        self.typing_timer_nbins = QtCore.QTimer()
        self.typing_timer_nbins.setSingleShot(True)
        self.typing_timer_nbins.timeout.connect(self.update_nbins)

        self.connect_dfw_with_functions()

    @property
    def list_of_datasets(self):
        return self.parent.localization_datasets

    def on_store_event(self, event):
        """Release filter state when its dataset goes away.

        The combo box is still updated positionally by remove_dataset_entry --
        a QComboBox has no notion of identity -- but the *data* is released by
        id, here, without anyone having to remember to ask.
        """
        if isinstance(event, DatasetClosed):
            self.filter_indices.pop(event.dataset_id, None)
        elif isinstance(event, StoreCleared):
            self.filter_indices.clear()

    def indices_for(self, dataset):
        """Removal indices recorded for *dataset*, or an empty array."""
        return self.filter_indices.get(
            dataset.dataset_id, np.asarray([], dtype=np.int32)
        )

    def _record_indices(self, dataset, indices):
        """Add *indices* to whatever is already filtered out of *dataset*."""
        existing = self.indices_for(dataset)
        indices = np.asarray(indices, dtype=np.int32).ravel()
        if existing.size:
            indices = np.concatenate((existing, indices), dtype=np.int32)
        self.filter_indices[dataset.dataset_id] = np.unique(indices)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError("Cannot change parent of existing Widget")

    def update_nbins(self):
        self.n_bins = self.dfw.SB_nbins.value()
        if self.list_of_datasets:
            self.dfw.CanvasWidget.draw(
                dataset=self.list_of_datasets[self.current_dataset_idx],
                parameter=self.list_of_filterable_parameters[
                    self.current_parameter_idx
                ],
                bins=self.n_bins,
            )

    def connect_dfw_with_functions(self):
        """Connect GUI with functionalities"""
        self.dfw.Cparameter.currentIndexChanged.connect(self.current_parameter_changed)
        self.dfw.Cdatasets.currentIndexChanged.connect(self.current_dataset_changed)
        self.dfw.Sfilter_slider.add_data_filter_itf(self)
        self.dfw.SB_nbins.valueChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_nbins)
        )
        self.dfw.Cfilter_mode.addItems(self.filter_modes)
        self.dfw.Cfilter_mode.currentIndexChanged.connect(self.filter_mode_changed)
        self.dfw.Cfilter_mode.setCurrentIndex(self.filter_mode_active_idx)
        self.dfw.Bapply_filter.clicked.connect(self.apply_filtering)
        self.dfw.Bapply_filter_to_all.clicked.connect(self.apply_filtering_to_all)
        self.dfw.Breset_filters.clicked.connect(self.reset_all_filtering)
        self.dfw.Sxrange.add_data_filter_itf(self)

    def filter_mode_changed(self):
        self.filter_mode_active_idx = self.dfw.Cfilter_mode.currentIndex()

    def _band_indices(self, dataset, reference_dataset):
        """Canonical-row indices of *dataset* excluded by the current band.

        The band is read off *reference_dataset* -- the one whose histogram the
        user is looking at -- because "apply to all" means applying *this* band
        everywhere, not recomputing a different band per dataset.  The two
        near-identical 40-line blocks this replaces differed only in that.
        """
        parameter = self.list_of_filterable_parameters[self.current_parameter_idx]
        reference = getattr(reference_dataset.locs_active, parameter)
        span = np.max(reference) - np.min(reference)
        low = self.filter_slider_values_decimal[0] * span + np.min(reference)
        high = self.filter_slider_values_decimal[1] * span + np.min(reference)

        values = getattr(dataset.locs_all, parameter)
        mode = self.filter_modes[self.filter_mode_active_idx]
        if mode == "Bandpass":
            return np.where((values > high) | (values < low))[0]
        if mode == "Bandstop":
            return np.where((values < high) & (values > low))[0]
        return None

    def apply_filtering(self, idx=None, update_layers=True):
        """Record which datapoints the current band filters out."""
        if isinstance(idx, bool) or idx is None:
            idx = self.current_dataset_idx
        dataset = self.list_of_datasets[idx]
        indices = self._band_indices(dataset, dataset)
        if indices is None:
            return
        self._record_indices(dataset, indices)
        if update_layers:
            self.parent.data_to_layer_itf.set_render_range_and_offset()
            # Only this dataset's selection changed.  Announcing it redraws
            # this dataset; it used to rebuild every layer in the viewer.
            self.parent.dataset_store.notify_mask_changed(dataset.dataset_id)
            self.current_parameter_changed()

    def apply_filtering_to_all(self):
        """like apply_filtering but apply the same band to all open datasets"""
        reference = self.list_of_datasets[self.current_dataset_idx]
        for dataset in self.list_of_datasets:
            indices = self._band_indices(dataset, reference)
            if indices is None:
                return
            self._record_indices(dataset, indices)
        self.parent.data_to_layer_itf.set_render_range_and_offset()
        self.parent.data_to_layer_itf.update_layers()
        self.current_parameter_changed()

    def reset_all_filtering(self):
        """Drop every recorded filter and restore each dataset's full selection."""
        self.filter_indices.clear()
        for dataset in self.list_of_datasets:
            dataset.reset_filters()
        self.current_parameter_changed()
        self.parent.data_to_layer_itf.update_layers()

    def clear_entries(self):
        """Reset GUI and filters"""
        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_filterable_parameters = []
        self.active_filters = []
        self.filter_indices.clear()
        self.dfw.clear_entries()

    def remove_dataset_entry(self, dataset_index):
        """Remove one dataset while keeping the remaining filter state aligned."""
        # filter_indices is released by id from on_store_event; only the
        # positional widget state is this method's business.
        if 0 <= dataset_index < len(self.active_filters):
            self.active_filters.pop(dataset_index)

        combo = self.dfw.Cdatasets
        combo.blockSignals(True)
        if 0 <= dataset_index < combo.count():
            combo.removeItem(dataset_index)
        self.n_datasets = combo.count()
        self.current_dataset_idx = min(dataset_index, self.n_datasets - 1)
        if self.n_datasets:
            combo.setCurrentIndex(self.current_dataset_idx)
        combo.blockSignals(False)

        if not self.n_datasets:
            self.clear_entries()
            return
        self.adjust_available_parameters_to_dataset_type()
        self.current_parameter_changed()
        self.reset_slider_positions()

    def add_dataset_entry(self, dataset_name):
        """Tell data filter itf that a new dataset was imported"""
        self.n_datasets += 1
        self.current_dataset_idx = self.n_datasets - 1
        self.active_filters.append({})
        self.dfw.Cdatasets.addItem(dataset_name)
        self.dfw.Cdatasets.setCurrentIndex(self.current_dataset_idx)

    def current_dataset_changed(self):
        self.current_dataset_idx = self.dfw.Cdatasets.currentIndex()
        self.adjust_available_parameters_to_dataset_type()
        self.current_parameter_changed()
        self.reset_slider_positions()

    def reset_slider_positions(self):
        self.dfw.Sfilter_slider.reset()
        self.dfw.Sxrange.reset()
        self.filter_slider_values_decimal = (0, 1)

    def current_parameter_changed(self, reset=True):
        self.current_parameter_idx = self.dfw.Cparameter.currentIndex()
        if self.list_of_datasets:
            self.dfw.CanvasWidget.draw(
                dataset=self.list_of_datasets[self.current_dataset_idx],
                parameter=self.list_of_filterable_parameters[
                    self.current_parameter_idx
                ],
                bins=self.n_bins,
                slider_values_decimal=self.xrange_slider_values_decimal,
            )
        if reset:
            self.reset_slider_positions()

    def adjust_available_parameters_to_dataset_type(self):
        """Depending on Dataset type set the available filterable parameters"""
        self.list_of_filterable_parameters = []
        if self.list_of_datasets:
            for param in self.list_of_datasets[self.current_dataset_idx].locs_dtype:
                self.list_of_filterable_parameters.append(param[0])
            self.dfw.Cparameter.clear()
            for param in self.list_of_filterable_parameters:
                self.dfw.Cparameter.addItem(param)

    def _start_typing_timer(self, timer):
        timer.start(500)
