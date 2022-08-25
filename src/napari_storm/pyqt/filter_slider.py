import napari
import numpy as np
from PyQt5.QtCore import Qt
from superqt import QDoubleRangeSlider


class RangeSlider3(QDoubleRangeSlider):
    def __init__(self, canvas, data_filter_itf=None, parent=None):
        self._parent = parent
        super().__init__()
        self.canvas = canvas
        self.ax = canvas.ax
        self.data_filter_itf = data_filter_itf
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 1000)
        self.Range = 1000
        self.range = None
        self.setSingleStep(1)
        self.setValue((0, 1000))
        self.type = type
        self.backup = self.value()
        self.valueChanged.connect(self.virtual_feedback)

    def reset(self):
        self.setValue((0, 1000))

    def parent(self):
        return self._parent

    def add_data_filter_itf(self, dfitf):
        self.data_filter_itf = dfitf

    def set_value_decimal(self, values):
        self.setValue(tuple(np.asarray(self.Range*np.asarray(values), dtype=int)))

    def mouseReleaseEvent(self, ev):
        self.data_filter_itf.filter_slider_values_decimal = np.asarray(self.value())/self.Range

    def virtual_feedback(self):
        if self.ax.lines:
            self.ax.lines[0].remove()
        tmp_max = np.max(getattr(self.data_filter_itf.list_of_datasets[
                                     self.data_filter_itf.current_dataset_idx].locs_active,
                                 self.data_filter_itf.list_of_filterable_parameters[
                                     self.data_filter_itf.current_parameter_idx]))
        tmp_min = np.min(getattr(self.data_filter_itf.list_of_datasets[
                                     self.data_filter_itf.current_dataset_idx].locs_active,
                                 self.data_filter_itf.list_of_filterable_parameters[
                                     self.data_filter_itf.current_parameter_idx]))
        x1 = (tmp_max - tmp_min) * self.value()[0] / self.Range + tmp_min
        x2 = (tmp_max - tmp_min) * self.value()[1] / self.Range + tmp_min
        ylim = self.ax.get_ylim()
        self.ax.plot([x1, x1, x2, x2], [0, ylim[1] + 1, ylim[1] + 1, 0], 'r')
        self.ax.set_ylim(ylim)
        self.canvas.draw()


class RangeSlider4(QDoubleRangeSlider):
    def __init__(self, canvas, data_filter_itf=None, parent=None):
        self._parent = parent
        super().__init__()
        self.canvas = canvas
        self.ax = canvas.ax
        self.data_filter_itf = data_filter_itf
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 1000)
        self.Range = 1000
        self.range = None
        self.setSingleStep(1)
        self.setValue((0, 1000))
        self.type = type
        self.backup = self.value()

    def reset(self):
        self.setValue((0, 1000))

    def parent(self):
        return self._parent

    def add_data_filter_itf(self, dfitf):
        self.data_filter_itf = dfitf

    def mouseReleaseEvent(self, ev):
        self.data_filter_itf.xrange_slider_values_decimal = np.asarray(self.value())/self.Range
        self.data_filter_itf.current_parameter_changed(reset=False)
        self.data_filter_itf.dfw.Sfilter_slider.virtual_feedback()



