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


class RangeSlider2(QDoubleRangeSlider):
    def __init__(self, parent=None, type='x'):
        self._parent = parent
        super().__init__(parent)
        self.setOrientation(Qt.Horizontal)
        self.setRange(0,100)
        self.Range=100
        self.range = None
        self.setSingleStep(1)
        self.setValue((1,100))
        self.type = type
        self.backup=self.value()
        self.created_feedback_layer=False
        self.valueChanged.connect(self.virtual_feedback)


    def getRange(self):
        return self.value()

    def parent(self):
        return self._parent



    """
        def mousePressEvent(self, event):
        if self.value()[0] != self.backup[0]:
            self.create_visual_feedback(0)
        else:
            self.create_visual_feedback(1)

    def mouseMoveEvent(self, event):
        if self.value()[0] != self.backup[0]:
            self.update_visual_feedback(0)
        else:
            self.update_visual_feedback(1)
        self.backup=self.value()

    def mouseReleaseEvent(self, event):
        self.remove_visual_feedback()
    """

    def mouseReleaseEvent(self, event):
        self.backup=self.value()
        self.remove_visual_feedback()

    def virtual_feedback(self):
        if self.created_feedback_layer:
            if self.value()[0] != self.backup[0]:
                self.update_visual_feedback(1)
            elif self.value()[1] != self.backup[1]:
                self.update_visual_feedback(2)
            else:
                pass
        else:
            self.created_feedback_layer=True
            self.create_visual_feedback(0)

    def create_visual_feedback(self,slider):
        v = napari.current_viewer()
        coords,faces=self.get_coords_faces(slider,create=True)
        self.range=v.add_surface((coords,faces),opacity=0,shading='smooth',name='render-range')

    def update_visual_feedback(self,slider):
        v = napari.current_viewer()
        coords,faces=self.get_coords_faces(slider_type=slider)
        #print(coords)
        v.layers['render-range'].data=(coords,faces)
        v.layers['render-range'].opacity=.25
        self.range.coords=coords

    def remove_visual_feedback(self):
        v = napari.current_viewer()
        try:
            v.layers.remove(self.range)
        except:
            print("something went wrong while removing feedback layer")
        self.created_feedback_layer=False
        update_layers(self.parent())

    def calc_backup(self):
        if self.parent().list_of_datasets[-1].zdim:
            x_backup = []
            y_backup = []
            z_backup = []
            for data in self.parent().list_of_datasets:
                x_backup.append(np.max(data.locs_backup.x))
                y_backup.append(np.max(data.locs_backup.y))
                z_backup.append(np.max(data.locs_backup.z))
            self.x_backup = np.max(x_backup)
            self.y_backup = np.max(y_backup)
            self.z_backup = np.max(z_backup)
        else:
            x_backup = []
            y_backup = []
            for data in self.parent().list_of_datasets:
                x_backup.append(np.max(data.locs_backup.x))
                y_backup.append(np.max(data.locs_backup.y))
            self.x_backup = np.max(x_backup)
            self.y_backup = np.max(y_backup)

    def get_coords_faces(self, slider_type, create=False):
        if self.parent().list_of_datasets[-1].zdim:
            #print("in",self.type)
            x = []
            y = []
            z = []
            if create:
                self.calc_backup()
            for data in self.parent().list_of_datasets:
                x.append(np.max(data.locs.x))
                y.append(np.max(data.locs.y))
                z.append(np.max(data.locs.z))
                x.append(np.min(data.locs.x))
                y.append(np.min(data.locs.y))
                z.append(np.min(data.locs.z))
            tmp = x
            x = y
            y = tmp
            coords=[]
            slider_1=self.value()[0]/self.Range
            slider_2=self.value()[1]/self.Range
            if self.type == 'x':
                x.append(self.x_backup)
                for i in [np.max(z), np.min(z)]:
                    for j in [np.max(y), np.min(y)]:
                        for k in [slider_1*np.max(x), slider_2 * np.max(x)]:
                            coords.append([i, j, k])
            elif self.type == 'y':
                y.append(self.y_backup)
                for i in [np.max(z), np.min(z)]:
                    for j in [slider_1*np.max(y), slider_2 * np.max(y)]:
                        for k in [np.max(x), np.min(x)]:
                            coords.append([i, j, k])
            else:
                z.append(self.z_backup)
                for i in [slider_1*np.max(z), slider_2 * np.max(z)]:
                    for j in [np.max(y), np.min(y)]:
                        for k in [np.max(x), np.min(x)]:
                            coords.append([i, j, k])
            faces = [[1, 2, 5], [2, 5, 6], [3, 4, 7], [4, 7, 8], [1, 3, 7], [1, 7, 5], [5, 6, 8],
                                     [5, 7, 8], [2, 6, 8], [2, 4, 8], [1, 2, 4], [1, 3, 4]]
            return np.reshape(np.asarray(coords), (8, 3)) * self.parent().list_of_datasets[
                -1].pixelsize, np.asarray(faces) - 1
        else:
            x = []
            y = []
            if create:
                self.calc_backup()
            for data in self.parent().list_of_datasets:
                x.append(np.max(data.locs.x))
                y.append(np.max(data.locs.y))
                x.append(np.min(data.locs.x))
                y.append(np.min(data.locs.y))
            tmp = x
            x = y
            y = tmp
            coords = []
            slider_1 = self.value()[0] / self.Range
            slider_2 = self.value()[1] / self.Range
            if self.type == 'x':
                x.append(self.x_backup)
                for j in [np.max(y), np.min(y)]:
                    for k in [slider_1*np.max(x), slider_2 * np.max(x)]:
                        coords.append([j, k])
            elif self.type == 'y':
                y.append(self.y_backup)
                for j in [slider_1*np.max(y), slider_2 * np.max(y)]:
                    for k in [np.max(x), np.min(x)]:
                        coords.append([j, k])
            faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
            return np.reshape(np.asarray(coords), (4, 2)) * self.parent().list_of_datasets[-1].pixelsize, np.asarray(
                faces) - 1



