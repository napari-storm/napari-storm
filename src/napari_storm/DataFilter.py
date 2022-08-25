import numpy as np
from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QSpinBox, QPushButton
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from .CustomErrors import *
from .pyqt.filter_slider import RangeSlider3, RangeSlider4


class DataFilterWindow(QWidget):

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
        data = getattr(dataset.locs_active, parameter)
        data[~ np.isfinite(data)] = 0
        self.Canvas.ax.hist(data, bins=bins, facecolor='white')
        self.Canvas.ax.set_xlabel(parameter)
        self.Canvas.ax.set_ylabel('#')
        ylim = self.Canvas.ax.get_ylim()
        self.Canvas.ax.set_ylim(ylim)
        self.set_xrange(data=data, slider_values_decimal=slider_values_decimal)
        self.Canvas.draw()

    def set_xrange(self, data, slider_values_decimal):
        tmp_xrange_data = [np.min(data), np.max(data)]
        tmp_absolute_xrange = tmp_xrange_data[1] - tmp_xrange_data[0]
        xrange = [slider_values_decimal[0] * tmp_absolute_xrange + tmp_xrange_data[0],
                  slider_values_decimal[1] * tmp_absolute_xrange + tmp_xrange_data[0]]
        self.Canvas.ax.set_xlim(xrange)


class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        self.fig.set_facecolor('#262930')
        self.ax = self.fig.subplots()
        self.ax.set_facecolor('#262930')
        Canvas.__init__(self, self.fig)
        # Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Canvas.updateGeometry(self)
        self.fig.subplots_adjust(0.2, 0.2, .9, .9)
        # self.fig.tight_layout()

    def reinitialise(self):
        if self.fig.axes:
            for ax in self.fig.axes:
                ax.cla()
            self.ax = self.fig.gca()
            self.fig.set_facecolor('#262930')
            self.ax.set_facecolor('#262930')
            self.draw()


class DataFilterInterface:
    def __init__(self, parent, data_filter_window):
        self._parent = parent
        self.dfw = data_filter_window

        self.active_filters = []  # create a dictionary of all active filters including the "normal" render range,

        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_filterable_parameters = []
        self.n_bins = 100
        self.filter_slider_values_decimal = [0, 1]
        self.filter_modes = ["Bandpass", "Bandstop"]
        self.filter_mode_active_idx = 0
        self.filter_idx_list = []
        self.xrange_slider_values_decimal = (0, 1)

        self.typing_timer_nbins = QtCore.QTimer()
        self.typing_timer_nbins.setSingleShot(True)
        self.typing_timer_nbins.timeout.connect(self.update_nbins)

        self.connect_dfw_with_functions()

    @property
    def list_of_datasets(self):
        return self.parent.localization_datasets

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def update_nbins(self):
        self.n_bins = self.dfw.SB_nbins.value()
        if self.list_of_datasets:
            self.dfw.CanvasWidget.draw(dataset=self.list_of_datasets[self.current_dataset_idx],
                                       parameter=self.list_of_filterable_parameters[self.current_parameter_idx],
                                       bins=self.n_bins)

    def connect_dfw_with_functions(self):
        self.dfw.Cparameter.currentIndexChanged.connect(self.current_parameter_changed)
        self.dfw.Cdatasets.currentIndexChanged.connect(self.current_dataset_changed)
        self.dfw.Sfilter_slider.add_data_filter_itf(self)
        self.dfw.SB_nbins.valueChanged.connect(lambda: self._start_typing_timer(self.typing_timer_nbins))
        self.dfw.Cfilter_mode.addItems(self.filter_modes)
        self.dfw.Cfilter_mode.currentIndexChanged.connect(self.filter_mode_changed)
        self.dfw.Cfilter_mode.setCurrentIndex(self.filter_mode_active_idx)
        self.dfw.Bapply_filter.clicked.connect(self.apply_filtering)
        self.dfw.Bapply_filter_to_all.clicked.connect(self.apply_filtering_to_all)
        self.dfw.Breset_filters.clicked.connect(self.reset_all_filtering)
        self.dfw.Sxrange.add_data_filter_itf(self)

    def filter_mode_changed(self):
        self.filter_mode_active_idx = self.dfw.Cfilter_mode.currentIndex()

    def apply_filtering(self, idx=None, update_layers=True):
        if isinstance(idx, bool):
            idx = self.current_dataset_idx
        while len(self.filter_idx_list) - 1 <= idx:
            self.filter_idx_list.append(np.asarray([], dtype=np.int32))
        tmp_dataset = self.list_of_datasets[idx]
        tmp_parameter_values_all = getattr(tmp_dataset.locs_all,
                                           self.list_of_filterable_parameters[self.current_parameter_idx])
        tmp_parameter_values_active = getattr(tmp_dataset.locs_active,
                                              self.list_of_filterable_parameters[self.current_parameter_idx])
        tmp_parameter_values_range_active = np.max(tmp_parameter_values_active) - np.min(tmp_parameter_values_active)
        if self.filter_modes[self.filter_mode_active_idx] == "Bandpass":
            tmp_indices = np.where(
                (tmp_parameter_values_all >
                 (self.filter_slider_values_decimal[1] * tmp_parameter_values_range_active +
                  np.min(tmp_parameter_values_active))) |
                (tmp_parameter_values_all <
                 ((tmp_parameter_values_range_active * self.filter_slider_values_decimal[0])
                  + np.min(tmp_parameter_values_active))))

        elif self.filter_modes[self.filter_mode_active_idx] == "Bandstop":
            tmp_indices = np.where(
                (tmp_parameter_values_all <
                 (self.filter_slider_values_decimal[1] * tmp_parameter_values_range_active
                  + np.min(tmp_parameter_values_active))) &
                (tmp_parameter_values_all >
                 ((tmp_parameter_values_range_active * self.filter_slider_values_decimal[0])
                  + np.min(tmp_parameter_values_active))))
        else:
            return

        if self.filter_idx_list[idx].size == 0:
            self.filter_idx_list[idx] = np.asarray(tmp_indices, dtype=np.int32)
        else:
            self.filter_idx_list[idx] = np.concatenate((self.filter_idx_list[idx], tmp_indices[0]), dtype=np.int32)

        self.filter_idx_list[idx] = np.unique(self.filter_idx_list[idx])
        if update_layers:
            self.parent.data_to_layer_itf.update_layers()
            self.current_parameter_changed()

    def apply_filtering_to_all(self):
        while len(self.filter_idx_list) - 1 <= len(self.list_of_datasets):
            self.filter_idx_list.append(np.asarray([], dtype=np.int32))
        tmp_dataset = self.list_of_datasets[self.current_dataset_idx]
        tmp_parameter_values_active = getattr(tmp_dataset.locs_active,
                                              self.list_of_filterable_parameters[self.current_parameter_idx])
        tmp_parameter_values_range_active = np.max(tmp_parameter_values_active) - np.min(tmp_parameter_values_active)
        for i in range(len(self.list_of_datasets)):
            tmp_dataset = self.list_of_datasets[i]
            tmp_parameter_values_all = getattr(tmp_dataset.locs_all,
                                               self.list_of_filterable_parameters[self.current_parameter_idx])
            if self.filter_modes[self.filter_mode_active_idx] == "Bandpass":
                tmp_indices = np.where(
                    (tmp_parameter_values_all >
                     (self.filter_slider_values_decimal[1] * tmp_parameter_values_range_active +
                      np.min(tmp_parameter_values_active))) |
                    (tmp_parameter_values_all <
                     ((tmp_parameter_values_range_active * self.filter_slider_values_decimal[0])
                      + np.min(tmp_parameter_values_active))))

            elif self.filter_modes[self.filter_mode_active_idx] == "Bandstop":
                tmp_indices = np.where(
                    (tmp_parameter_values_all <
                     (self.filter_slider_values_decimal[1] * tmp_parameter_values_range_active
                      + np.min(tmp_parameter_values_active))) &
                    (tmp_parameter_values_all >
                     ((tmp_parameter_values_range_active * self.filter_slider_values_decimal[0])
                      + np.min(tmp_parameter_values_active))))
            else:
                return
            if self.filter_idx_list[i].size == 0:
                self.filter_idx_list[i] = np.asarray(tmp_indices, dtype=np.int32)
            else:
                self.filter_idx_list[i] = np.concatenate((self.filter_idx_list[i], tmp_indices[0]), dtype=np.int32)

            self.filter_idx_list[i] = np.unique(self.filter_idx_list[i])
        self.parent.data_to_layer_itf.update_layers()
        self.current_parameter_changed()

    def reset_all_filtering(self):
        print("reset in progress")
        for idx in range(len(self.filter_idx_list)):
            self.filter_idx_list[idx] = np.asarray([], dtype=np.int32)
        for dataset in self.list_of_datasets:
            dataset.reset_filters()
        self.current_parameter_changed()
        self.parent.data_to_layer_itf.update_layers()

    def clear_entries(self):
        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_filterable_parameters = []
        self.dfw.clear_entries()

    def add_dataset_entry(self, dataset_name):
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
            self.dfw.CanvasWidget.draw(dataset=self.list_of_datasets[self.current_dataset_idx],
                                       parameter=self.list_of_filterable_parameters[self.current_parameter_idx],
                                       bins=self.n_bins, slider_values_decimal=self.xrange_slider_values_decimal)
        if reset:
            self.reset_slider_positions()

    def adjust_available_parameters_to_dataset_type(self):
        self.list_of_filterable_parameters = []
        if self.list_of_datasets:
            for param in self.list_of_datasets[self.current_dataset_idx].locs_dtype:
                self.list_of_filterable_parameters.append(param[0])
            self.dfw.Cparameter.clear()
            for param in self.list_of_filterable_parameters:
                self.dfw.Cparameter.addItem(param)

    def _start_typing_timer(self, timer):
        timer.start(500)
