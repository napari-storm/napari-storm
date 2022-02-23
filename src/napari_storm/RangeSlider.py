from typing import Tuple

from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtWidgets import QStyleOptionSlider, QStyle
from superqt import QDoubleRangeSlider
import numpy as np
import napari
from superqt.sliders._generic_range_slider import SC_BAR
from superqt.sliders._generic_slider import SC_HANDLE

from ._dock_widget import update_layers


class RangeSlider(QDoubleRangeSlider):
    def __init__(self, parent=None, type='x'):
        self._parent = parent
        super().__init__(parent)
        self.setOrientation(Qt.Horizontal)
        self.setRange(0,100)
        self.Range=100
        self.range = None
        self.setSingleStep(1)
        self.setValue((10,90))
        self.type = type
        self.backup=self.value()
        self.created_feedback_layer=False


    def getRange(self):
        values=np.asarray(self.value())/90
        for i in range(len(values)):
            if values[i]>1:
                values[i]=np.exp((values[i]-1)*40)
            elif values[i]<0.11:
                values[i]=np.log(values[i]+0.000001)
        return values

    def parent(self):
        return self._parent






