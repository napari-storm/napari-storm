from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QSpinBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from .CustomErrors import *
from .pyqt.filter_slider import RangeSlider3


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

        self.Sfilter_slider = RangeSlider3(canvas=self.CanvasWidget.Canvas)
        self.layout.addRow("Filter Data:", QWidget())
        self.layout.addRow(self.Sfilter_slider)

        self.SB_nbins = QSpinBox()
        self.SB_nbins.setRange(1, 1000)
        self.SB_nbins.setValue(100)

        self.layout.addRow("Number of bins: ", self.SB_nbins)

        self.setLayout(self.layout)

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

    def draw(self, dataset, parameter):
        self.Canvas.reinitialise()
        self.Canvas.ax.hist(getattr(dataset.locs, parameter), bins=100, facecolor='white')
        self.Canvas.ax.set_xlabel(parameter)
        ylim = self.Canvas.ax.get_ylim()
        self.Canvas.ax.set_ylim(ylim)
        self.Canvas.draw()


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

        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_datasets = []
        self.list_of_filterable_parameters = []

        self.connect_dfw_with_functions()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def connect_dfw_with_functions(self):
        self.dfw.Cparameter.currentIndexChanged.connect(self.current_parameter_changed)
        self.dfw.Cdatasets.currentIndexChanged.connect(self.current_dataset_changed)
        self.dfw.Sfilter_slider.add_data_filter_itf(self)

    def clear_entries(self):
        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_datasets = []
        self.list_of_filterable_parameters = []
        self.dfw.clear_entries()

    def add_dataset_entry(self, dataset_name):
        self.n_datasets += 1
        self.current_dataset_idx = self.n_datasets
        self.list_of_datasets = self.parent.localization_datasets
        self.dfw.Cdatasets.addItem(dataset_name)

    def current_dataset_changed(self):
        self.current_dataset_idx = self.dfw.Cdatasets.currentIndex()
        self.adjust_available_parameters_to_dataset_type()
        self.current_parameter_changed()

    def current_parameter_changed(self):
        self.current_parameter_idx = self.dfw.Cparameter.currentIndex()
        if self.list_of_datasets:
            self.dfw.CanvasWidget.draw(dataset=self.list_of_datasets[self.current_dataset_idx],
                                       parameter=self.list_of_filterable_parameters[self.current_parameter_idx])

    def adjust_available_parameters_to_dataset_type(self):
        self.list_of_filterable_parameters = []
        if self.list_of_datasets:
            for param in self.list_of_datasets[self.current_dataset_idx].locs_dtype:
                self.list_of_filterable_parameters.append(param[0])
            self.dfw.Cparameter.clear()
            for param in self.list_of_filterable_parameters:
                self.dfw.Cparameter.addItem(param)
