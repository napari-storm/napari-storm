from .localization_dataset_types import *

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QSpinBox, QPushButton, QLineEdit
from .CustomErrors import *


class DataAdjustmentWindow(QWidget):
    """GUI Elements for the data adjustment widget"""
    def __init__(self, parent):
        super().__init__()

        self.math_mode_active_idx = 0

        self.setWindowTitle("Data Adjustment")

        self.layout = QFormLayout()

        self.parent = parent

        self.Cdatasets = QComboBox()
        self.layout.addRow(self.Cdatasets)

        self.Cparameter = QComboBox()

        self.layout.addRow(self.Cparameter)

        self.Cmath_mode = QComboBox()

        self.Evalue = QLineEdit()

        self.Bapply_filter = QPushButton()
        self.Bapply_filter.setText("Apply adjustment to current dataset")

        self.Bsave_as_ns = QPushButton()
        self.Bsave_as_ns.setText("Export current dataset as .ns")

        self.layout.addRow("Math mode:", self.Cmath_mode)
        self.layout.addRow("Value:", self.Evalue)
        self.layout.addRow(self.Bapply_filter)
        self.layout.addRow(self.Bsave_as_ns)

        self.setLayout(self.layout)

    def filter_mode_changed(self):
        self.math_mode_active_idx = self.Cmath_mode.currentIndex()

    def clear_entries(self):
        self.Cparameter.clear()
        self.Cdatasets.clear()


class DataAdjustmentInterface:
    """Core code of the data adjustment functions"""
    def __init__(self, parent, data_adjustment_window):
        self._parent = parent
        self.daw = data_adjustment_window

        self.list_of_adjustable_parameters = []

        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.math_modes = ["add offset", "rescale"]
        self.math_mode_active_idx = 0
        self.value = 0

        self.qtimer = QTimer()
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(self.value_changed)

        self.connect_daw_with_functions()

    @property
    def list_of_datasets(self):
        return self.parent.list_of_datasets

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def value_changed(self):
        tmp_value = self.daw.Evalue.text()
        try:
            self.value = float(tmp_value)
        except ValueError:
            self.daw.Evalue.clear()

    def connect_daw_with_functions(self):
        """Connect GUI with functionalities"""
        self.daw.Cparameter.currentIndexChanged.connect(self.current_parameter_changed)
        self.daw.Cdatasets.currentIndexChanged.connect(self.current_dataset_changed)
        self.daw.Evalue.textChanged.connect(lambda: self._start_typing_timer(self.qtimer))
        self.daw.Cmath_mode.addItems(self.math_modes)
        self.daw.Cmath_mode.currentIndexChanged.connect(self.math_mode_changed)
        self.daw.Cmath_mode.setCurrentIndex(self.math_mode_active_idx)
        self.daw.Bapply_filter.clicked.connect(self.apply_adjustement)
        self.daw.Bsave_as_ns.clicked.connect(self.save_current_dataset_as_ns)

    def math_mode_changed(self):
        self.math_mode_active_idx = self.daw.Cmath_mode.currentIndex()

    def apply_adjustement(self, idx=None, update_layers=True):
        """Apply adjustment to dataset of index idx and possibly update the layers"""
        if isinstance(idx, bool):
            idx = self.current_dataset_idx
        if self.math_modes[self.math_mode_active_idx] == self.math_modes[0]:  # add offset
            setattr(self.list_of_datasets[idx].locs_all,
                    self.list_of_adjustable_parameters[self.current_parameter_idx],
                    getattr(self.list_of_datasets[idx].locs_all,
                            self.list_of_adjustable_parameters[self.current_parameter_idx]) + self.value)
        elif self.math_modes[self.math_mode_active_idx] == self.math_modes[1]:  # rescale
            setattr(self.list_of_datasets[idx].locs_all,
                    self.list_of_adjustable_parameters[self.current_parameter_idx],
                    getattr(self.list_of_datasets[idx].locs_all,
                            self.list_of_adjustable_parameters[self.current_parameter_idx]) * self.value)
        if update_layers:
            self.parent.data_to_layer_itf.set_render_range_and_offset()
            self.parent.dataset_itf.hard_refresh(update_data_range=True)
            
    def clear_entries(self):
        """Reset GUI and adjustments"""
        self.n_datasets = 0
        self.current_dataset_idx = 0
        self.current_parameter_idx = 0
        self.list_of_adjustable_parameters = []
        self.daw.clear_entries()

    def add_dataset_entry(self, dataset_name):
        """Tell data filter itf that a new dataset was imported"""
        self.n_datasets += 1
        self.current_dataset_idx = self.n_datasets - 1
        self.daw.Cdatasets.addItem(dataset_name)
        self.daw.Cdatasets.setCurrentIndex(self.current_dataset_idx)

    def current_dataset_changed(self):
        self.current_dataset_idx = self.daw.Cdatasets.currentIndex()
        self.adjust_available_parameters_to_dataset_type()
        self.current_parameter_changed()

    def current_parameter_changed(self, reset=True):
        self.current_parameter_idx = self.daw.Cparameter.currentIndex()

    def adjust_available_parameters_to_dataset_type(self):
        """Depending on Dataset type set the available filterable parameters"""
        self.list_of_adjustable_parameters = []
        if self.list_of_datasets:
            for param in self.list_of_datasets[self.current_dataset_idx].locs_dtype:
                self.list_of_adjustable_parameters.append(param[0])
            self.daw.Cparameter.clear()
            for param in self.list_of_adjustable_parameters:
                self.daw.Cparameter.addItem(param)

    def save_current_dataset_as_ns(self, filename=None):
        tmp_dataset = self.list_of_datasets[self.current_dataset_idx]
        if not filename:
            filename = QFileDialog.getSaveFileName()[0]
        with h5py.File(filename, 'a') as f:
            try:
                dset = f.create_dataset("dataset", data=tmp_dataset.locs_all)
            except ValueError:
                dset["dataset"] = tmp_dataset.locs_all
            dset.attrs["name"] = tmp_dataset.name
            dset.attrs["zdim_present"] = tmp_dataset.zdim_present
            dset.attrs["dataset_class"] = tmp_dataset.__class__.__name__
            if isinstance(tmp_dataset, StormDataClass):
                dset.attrs["pixelsize_nm"] = tmp_dataset.pixelsize_nm
                dset.attrs["sigma_present"] = tmp_dataset.sigma_present
                dset.attrs["photon_count_present"] = tmp_dataset.photon_count_present
        os.rename(filename, filename.split('.')[0]+".ns")


    def _start_typing_timer(self, timer):
        timer.start(500)
