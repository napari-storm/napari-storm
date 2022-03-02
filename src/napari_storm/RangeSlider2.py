import napari
import numpy as np
from PyQt5.QtCore import Qt
from superqt import QDoubleRangeSlider

from .DataToLayerInterface import DataToLayerInterface


class RangeSlider2(QDoubleRangeSlider):
    def __init__(self, parent=None, type="x"):
        self._parent = parent
        super().__init__(parent)
        self.setOrientation(Qt.Horizontal)
        self.setRange(0, 100)
        self.Range = 100
        self.range = None
        self.setSingleStep(1)
        self.setValue((0, 100))
        self.type = type
        self.backup = self.value()
        self.created_feedback_layer = False
        self.valueChanged.connect(self.virtual_feedback)
        self.reset_in_progress = False

    def reset(self):
        self.reset_in_progress = True
        self.setValue((0, 100))
        self.reset_in_progress = False
        self.parent().update_render_range(self.type, self.getRange())

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
        self.backup = self.value()
        self.parent().update_render_range(self.type, self.getRange())
        self.remove_visual_feedback()

    def virtual_feedback(self):
        if not self.reset_in_progress:
            if self.created_feedback_layer:
                if self.value()[0] != self.backup[0]:
                    self.update_visual_feedback(1)
                elif self.value()[1] != self.backup[1]:
                    self.update_visual_feedback(2)
                else:
                    pass
            else:
                self.created_feedback_layer = True
                self.create_visual_feedback(0)

    def create_visual_feedback(self, slider):
        v = napari.current_viewer()
        coords, faces = self.get_coords_faces(slider, create=True)
        self.range = v.add_surface(
            (coords, faces), opacity=0, shading="smooth", name="render-range"
        )

    def update_visual_feedback(self, slider):
        v = napari.current_viewer()
        coords, faces = self.get_coords_faces(slider_type=slider)
        # print(coords)
        v.layers["render-range"].data = (coords, faces)
        v.layers["render-range"].opacity = 0.25
        self.range.coords = coords

    def remove_visual_feedback(self):
        v = napari.current_viewer()
        try:
            v.layers.remove(self.range)
        except:
            print("something went wrong while removing feedback layer")
        self.created_feedback_layer = False
        DataToLayerInterface.update_layers(self.parent().data_to_layer_itf)

    def calc_backup(self):
        if self.parent().localization_datasets[-1].zdim_present:
            x_backup = []
            y_backup = []
            z_backup = []
            for data in self.parent().localization_datasets:
                x_backup.append(np.max(data.locs_all.x_pos_pixels)-data.offset_pixels[0])
                y_backup.append(np.max(data.locs_all.y_pos_pixels)-data.offset_pixels[1])
                z_backup.append(np.max(data.locs_all.z_pos_pixels)-data.offset_pixels[2])
            self.x_backup = np.max(x_backup)
            self.y_backup = np.max(y_backup)
            self.z_backup = np.max(z_backup)
        else:
            x_backup = []
            y_backup = []
            for data in self.parent().localization_datasets:
                x_backup.append(np.max(data.locs_all.x_pos_pixels)-data.offset_pixels[0])
                y_backup.append(np.max(data.locs_all.y_pos_pixels)-data.offset_pixels[1])
            self.x_backup = np.max(x_backup)
            self.y_backup = np.max(y_backup)

    def get_coords_faces(self, slider_type, create=False):
        if self.parent().localization_datasets[-1].zdim_present:
            # print("in",self.type)
            x = []
            y = []
            z = []
            if create:
                self.calc_backup()
            for data in self.parent().localization_datasets:
                x.append(np.max(data.locs.x_pos_pixels))
                y.append(np.max(data.locs.y_pos_pixels))
                z.append(np.max(data.locs.z_pos_pixels))
                x.append(np.min(data.locs.x_pos_pixels))
                y.append(np.min(data.locs.y_pos_pixels))
                z.append(np.min(data.locs.z_pos_pixels))
            tmp = x
            x = y
            y = tmp
            coords = []
            slider_1 = self.value()[0] / self.Range
            slider_2 = self.value()[1] / self.Range
            if self.type == "x":
                x.append(self.x_backup)
                for i in [np.max(z), np.min(z)]:
                    for j in [np.max(y), np.min(y)]:
                        for k in [slider_1 * np.max(x), slider_2 * np.max(x)]:
                            coords.append([i, j, k])
            elif self.type == "y":
                y.append(self.y_backup)
                for i in [np.max(z), np.min(z)]:
                    for j in [slider_1 * np.max(y), slider_2 * np.max(y)]:
                        for k in [np.max(x), np.min(x)]:
                            coords.append([i, j, k])
            else:
                z.append(self.z_backup)
                for i in [slider_1 * np.max(z), slider_2 * np.max(z)]:
                    for j in [np.max(y), np.min(y)]:
                        for k in [np.max(x), np.min(x)]:
                            coords.append([i, j, k])
            faces = [
                [1, 2, 5],
                [2, 5, 6],
                [3, 4, 7],
                [4, 7, 8],
                [1, 3, 7],
                [1, 7, 5],
                [5, 6, 8],
                [5, 7, 8],
                [2, 6, 8],
                [2, 4, 8],
                [1, 2, 4],
                [1, 3, 4],
            ]
            return (
                np.reshape(np.asarray(coords)+self.parent().localization_datasets[-1].offset_pixels[::-1], (8, 3))
                * self.parent().localization_datasets[-1].pixelsize_nm,
                np.asarray(faces) - 1,
            )
        else:
            x = []
            y = []
            if create:
                self.calc_backup()
            for data in self.parent().localization_datasets:
                x.append(np.max(data.locs.x_pos_pixels))
                y.append(np.max(data.locs.y_pos_pixels))
                x.append(np.min(data.locs.x_pos_pixels))
                y.append(np.min(data.locs.y_pos_pixels))
            tmp = x
            x = y
            y = tmp
            coords = []
            slider_1 = self.value()[0] / self.Range
            slider_2 = self.value()[1] / self.Range
            if self.type == "x":
                x.append(self.x_backup)
                for j in [np.max(y), np.min(y)]:
                    for k in [slider_1 * np.max(x), slider_2 * np.max(x)]:
                        coords.append([j, k])
            elif self.type == "y":
                y.append(self.y_backup)
                for j in [slider_1 * np.max(y), slider_2 * np.max(y)]:
                    for k in [np.max(x), np.min(x)]:
                        coords.append([j, k])
            faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
            return (
                np.reshape(np.asarray(coords)+self.parent().localization_datasets[-1].offset_pixels[::-1], (4, 2))
                * self.parent().localization_datasets[-1].pixelsize_nm,
                np.asarray(faces) - 1,
            )
