import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .core import (
    DEFAULT_DENSITY_RADIUS_NM,
    DENSITY_COLUMN,
    DatasetClosed,
    StoreCleared,
    drop_edits,
    edits_of,
    local_density,
    record_edit,
    split_point,
)
from .CustomErrors import ParentError
from .pyqt.filter_slider import RangeSlider3, RangeSlider4


def _band_fields(edit):
    """The four fields that define a band, as plain values."""
    return {
        "parameter": str(edit["parameter"]),
        "low": float(edit["low"]),
        "high": float(edit["high"]),
        "mode": str(edit["mode"]),
    }


def column_values(dataset, parameter):
    """*parameter* over every canonical row: a measured or a derived column."""
    return np.asarray(dataset.table.column(parameter))


def active_values(dataset, parameter):
    """*parameter* over the rows drawn now, as the histogram shows them."""
    values = column_values(dataset, parameter)
    mask = dataset.active_mask
    return values if mask is None else values[mask]


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

        self.Bsplit = QPushButton()
        self.Bsplit.setText("Split current dataset by last filter")
        self.Bsplit.setToolTip(
            "Apply a filter first. This replaces the current dataset with two "
            "channels: what the last filter kept ('in', what you see now) and "
            "what it removed ('out'), each with its own colour. Localizations "
            "removed by earlier filters go to neither."
        )

        self.Sfilter_slider = RangeSlider3(canvas=self.CanvasWidget.Canvas)

        self.Sxrange = RangeSlider4(canvas=self.CanvasWidget.Canvas)

        self.layout.addRow("Adjust graph range:", QWidget())
        self.layout.addRow(self.Sxrange)
        self.layout.addRow("Filter mode:", self.Cfilter_mode)
        self.layout.addRow(self.Sfilter_slider)
        self.layout.addRow(self.Bapply_filter)
        self.layout.addRow(self.Bapply_filter_to_all)
        self.layout.addRow(self.Bsplit)
        self.layout.addRow(self.Breset_filters)

        self.SB_nbins = QSpinBox()
        self.SB_nbins.setRange(1, 1000)
        self.SB_nbins.setValue(100)
        self.layout.addRow("Number of bins: ", self.SB_nbins)

        self.layout.addRow(QLabel("<b>Local density</b>"))
        self.SBdensity_radius = QDoubleSpinBox()
        self.SBdensity_radius.setRange(0.1, 100000.0)
        self.SBdensity_radius.setDecimals(1)
        self.SBdensity_radius.setSuffix(" nm")
        self.SBdensity_radius.setValue(DEFAULT_DENSITY_RADIUS_NM)
        self.SBdensity_radius.setToolTip(
            "Each localization counts the other localizations within this "
            "radius: a sphere for 3-D data, a circle for 2-D."
        )
        self.layout.addRow("Search radius:", self.SBdensity_radius)
        self.Bdensity = QPushButton()
        self.Bdensity.setText("Calculate local density")
        self.Bdensity.setToolTip(
            "Adds a 'local_density' parameter to every dataset: how many "
            "localizations lie within the search radius of each one. Only "
            "localizations that pass the filters applied so far count as "
            "neighbours. Filter on it here, split by it, or colour by it."
        )
        self.layout.addRow(self.Bdensity)

        self.setLayout(self.layout)
        # A click on the tab's background used to be wired to reset the render
        # range -- crop and camera -- with a handler that took no event, so it
        # raised a TypeError instead. Neither is wanted: a click is not a reset.

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
        data = active_values(dataset, parameter)
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
        # What each filter removed, in the order applied, per dataset id: a
        # split separates what the last one removed from what is still kept.
        self.filter_steps = {}
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
            self.filter_steps.pop(event.dataset_id, None)
        elif isinstance(event, StoreCleared):
            self.filter_indices.clear()
            self.filter_steps.clear()

    def indices_for(self, dataset):
        """Removal indices recorded for *dataset*, or an empty array."""
        return self.filter_indices.get(
            dataset.dataset_id, np.asarray([], dtype=np.int32)
        )

    def _record_indices(self, dataset, indices):
        """Add *indices* to whatever is already filtered out of *dataset*."""
        existing = self.indices_for(dataset)
        indices = np.asarray(indices, dtype=np.int32).ravel()
        self.filter_steps.setdefault(dataset.dataset_id, []).append(indices.copy())
        if existing.size:
            indices = np.concatenate((existing, indices), dtype=np.int32)
        self.filter_indices[dataset.dataset_id] = np.unique(indices)

    def filter_steps_for(self, dataset):
        """What each filter applied to *dataset* removed, oldest first."""
        return list(self.filter_steps.get(dataset.dataset_id, ()))

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
        self.dfw.Bsplit.clicked.connect(self.split_current_dataset)
        self.dfw.Bdensity.clicked.connect(lambda: self.calculate_local_density())
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
        return self.band_indices(dataset, **self._current_band(reference_dataset))

    def current_values(self):
        """The shown parameter over the current dataset's drawn rows."""
        return active_values(
            self.list_of_datasets[self.current_dataset_idx],
            self.list_of_filterable_parameters[self.current_parameter_idx],
        )

    def _current_band(self, reference_dataset):
        """The band the sliders select, in the parameter's own units."""
        parameter = self.list_of_filterable_parameters[self.current_parameter_idx]
        reference = active_values(reference_dataset, parameter)
        span = np.max(reference) - np.min(reference)
        low = self.filter_slider_values_decimal[0] * span + np.min(reference)
        high = self.filter_slider_values_decimal[1] * span + np.min(reference)
        return {
            "parameter": parameter,
            "low": float(low),
            "high": float(high),
            "mode": self.filter_modes[self.filter_mode_active_idx],
        }

    @staticmethod
    def band_indices(dataset, parameter, low, high, mode):
        """Canonical-row indices of *dataset* outside (Bandpass) or inside
        (Bandstop) the band, or None for a mode this does not know."""
        values = column_values(dataset, parameter)
        if mode == "Bandpass":
            return np.where((values > high) | (values < low))[0]
        if mode == "Bandstop":
            return np.where((values < high) & (values > low))[0]
        return None

    def replay_band(self, dataset, edit):
        """Apply a band recorded by a scene, without touching the sliders."""
        indices = self.band_indices(
            dataset, edit["parameter"], edit["low"], edit["high"], edit["mode"]
        )
        if indices is None:
            raise ValueError(f"unknown filter mode {edit['mode']!r}")
        self._record_indices(dataset, indices)
        record_edit(dataset, {"kind": "filter", **_band_fields(edit)})

    def apply_filtering(self, idx=None, update_layers=True):
        """Record which datapoints the current band filters out."""
        if isinstance(idx, bool) or idx is None:
            idx = self.current_dataset_idx
        dataset = self.list_of_datasets[idx]
        band = self._current_band(dataset)
        indices = self.band_indices(dataset, **band)
        if indices is None:
            return
        self._record_indices(dataset, indices)
        record_edit(dataset, {"kind": "filter", **band})
        if update_layers:
            # No re-measuring of the extent: it is taken from every
            # localization, filtered or not, so a filter cannot change it --
            # and re-measuring re-centred the camera, losing the user's view.
            # Only this dataset's selection changed.  Announcing it redraws
            # this dataset; it used to rebuild every layer in the viewer.
            self.parent.dataset_store.notify_mask_changed(dataset.dataset_id)
            self.current_parameter_changed()

    def apply_filtering_to_all(self):
        """like apply_filtering but apply the same band to all open datasets"""
        reference = self.list_of_datasets[self.current_dataset_idx]
        band = self._current_band(reference)
        for dataset in self.list_of_datasets:
            if not dataset.table.has_field(band["parameter"]):
                # e.g. local density, on a dataset opened after it was computed
                continue
            indices = self.band_indices(dataset, **band)
            if indices is None:
                return
            self._record_indices(dataset, indices)
            record_edit(dataset, {"kind": "filter", **band})
        # update_layers keeps the camera where it is; see apply_filtering.
        self.parent.data_to_layer_itf.update_layers()
        self.current_parameter_changed()

    def reset_all_filtering(self):
        """Drop every recorded filter and restore each dataset's full selection.

        A local density counted only the localizations the filters kept, so
        it is counted again, over everything, with the same radius. What a
        split baked into a part's rows stays: the part has no others.
        """
        self.filter_indices.clear()
        self.filter_steps.clear()
        for dataset in self.list_of_datasets:
            dataset.reset_filters()
            drop_edits(dataset, "filter")
            log = edits_of(dataset)
            radii = [
                edit["radius_nm"]
                for edit in log[split_point(log) :]
                if edit.get("kind") == "density"
            ]
            if radii:
                drop_edits(dataset, "density")
                self.compute_density(dataset, radii[-1])
        self.current_parameter_changed()
        self.parent.data_to_layer_itf.update_layers()

    # ------------------------------------------------------------------
    # Local density
    # ------------------------------------------------------------------

    def density_neighbours(self, dataset):
        """The localizations that count as neighbours: those the filters keep.

        The render range is not a filter here -- a crop would make every
        localization near its edge look sparse.
        """
        neighbours = np.ones(dataset.number_of_entries(), dtype=bool)
        neighbours[self.indices_for(dataset)] = False
        return neighbours

    @staticmethod
    def density_coordinates(dataset):
        axes = ["x", "y"]
        if dataset.zdim_present and dataset.table.has_axis("z"):
            axes.append("z")
        return np.column_stack(
            [np.asarray(dataset.all_coordinate_nm(axis)) for axis in axes]
        )

    def compute_density(self, dataset, radius_nm):
        """Write *dataset*'s local density column, and record that it was."""
        counts = local_density(
            self.density_coordinates(dataset),
            radius_nm,
            neighbours=self.density_neighbours(dataset),
        )
        dataset.table.set_side_column(DENSITY_COLUMN, counts)
        record_edit(dataset, {"kind": "density", "radius_nm": float(radius_nm)})

    def replay_density(self, dataset, edit):
        """Count again what a scene recorded, without touching the controls."""
        self.compute_density(dataset, float(edit["radius_nm"]))

    def calculate_local_density(self, radius_nm=None):
        """Give every open dataset a local density column, and show it."""
        if not self.list_of_datasets:
            return
        radius = self.dfw.SBdensity_radius.value() if radius_nm is None else radius_nm
        QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for dataset in self.list_of_datasets:
                self.compute_density(dataset, radius)
        finally:
            QApplication.restoreOverrideCursor()
        self.show_parameter(DENSITY_COLUMN)
        self.parent._sync_color_coding_controls()
        if self.parent.render_config.color_by == DENSITY_COLUMN:
            self.parent.data_to_layer_itf.update_layers()
            self.parent._refresh_color_bar()

    def show_parameter(self, parameter):
        """Offer the current dataset's parameters again, with *parameter* shown."""
        self.adjust_available_parameters_to_dataset_type()
        if parameter in self.list_of_filterable_parameters:
            index = self.list_of_filterable_parameters.index(parameter)
            if self.dfw.Cparameter.currentIndex() == index:
                self.current_parameter_changed()
            else:
                self.dfw.Cparameter.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_current_dataset(self):
        """Replace the current dataset with what the last filter kept and removed."""
        if not self.list_of_datasets:
            return None
        from .dataset_split import split_by_last_filter

        dataset = self.list_of_datasets[self.current_dataset_idx]
        try:
            return split_by_last_filter(self.parent, dataset)
        except ValueError as error:
            QMessageBox.warning(self.dfw, "Cannot split dataset", str(error))
            return None

    def clear_entries(self):
        """Reset GUI and filters"""
        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_filterable_parameters = []
        self.active_filters = []
        self.filter_indices.clear()
        self.filter_steps.clear()
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
            dataset = self.list_of_datasets[self.current_dataset_idx]
            for param in dataset.locs_dtype:
                self.list_of_filterable_parameters.append(param[0])
            # Derived columns -- local density, group ids -- after the measured.
            for name in dataset.table.side_column_names:
                if name not in self.list_of_filterable_parameters:
                    self.list_of_filterable_parameters.append(name)
            self.dfw.Cparameter.clear()
            for param in self.list_of_filterable_parameters:
                self.dfw.Cparameter.addItem(param)

    def _start_typing_timer(self, timer):
        timer.start(500)
