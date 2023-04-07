import napari
import numpy as np
from PyQt5.QtCore import Qt
from superqt import QDoubleRangeSlider

from ..DataToLayerInterface import DataToLayerInterface


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

    def mouseReleaseEvent(self, event):
        self.backup = self.value()
        if self.type == "x":  # Napari coordinate system is weird
            self.parent().update_render_range(self.type, (self.Range-self.getRange()[1], self.Range-self.getRange()[0]))
        else:
            self.parent().update_render_range(self.type, self.getRange())
        self.remove_visual_feedback()

    def virtual_feedback(self):
        if not self.reset_in_progress:
            if self.created_feedback_layer:
                if self.value()[0] != self.backup[0]:
                    self.update_visual_feedback()
                elif self.value()[1] != self.backup[1]:
                    self.update_visual_feedback()
                else:
                    pass
            else:
                self.created_feedback_layer = True
                self.create_visual_feedback(0)

    def create_visual_feedback(self, slider):
        v = napari.current_viewer()
        coords, faces = self.get_coords_faces()
        self.range = v.add_surface(
            (coords, faces), opacity=0, shading="smooth", name="render-range",
            colormap=self.parent().active_render_range_box_color
        )

    def update_visual_feedback(self):
        v = napari.current_viewer()
        coords, faces = self.get_coords_faces()
        # print(coords)
        v.layers["render-range"].data = (coords, faces)
        v.layers["render-range"].opacity = self.parent().render_range_box_opacity
        self.range.coords = coords

    def remove_visual_feedback(self):
        v = napari.current_viewer()
        try:
            v.layers.remove(self.range)
        except:
            print("something went wrong while removing feedback layer")
        self.created_feedback_layer = False
        self.parent().dataset_itf.hard_refresh(update_data_range=True)

    def get_coords_faces(self):
        if self.parent().zdim:
            rrx_full = self.parent().data_to_layer_itf.render_range_x
            rrx = rrx_full.copy()
            rrx[0] += (rrx_full[1] - rrx_full[0]) * self.parent().render_range_slider_x_percent[0] / 100
            rrx[1] -= (rrx_full[1] - rrx_full[0]) * (1 - self.parent().render_range_slider_x_percent[1] / 100)
            rry_full = self.parent().data_to_layer_itf.render_range_y
            rry = rry_full.copy()
            rry[0] += (rry_full[1] - rry_full[0]) * self.parent().render_range_slider_y_percent[0] / 100
            rry[1] -= (rry_full[1] - rry_full[0]) * (1 - self.parent().render_range_slider_y_percent[1] / 100)
            rrz_full = self.parent().data_to_layer_itf.render_range_z
            rrz = rrz_full.copy()
            rrz[0] += (rrz_full[1] - rrz_full[0]) * self.parent().render_range_slider_z_percent[0] / 100
            rrz[1] -= (rrz_full[1] - rrz_full[0]) * (1 - self.parent().render_range_slider_z_percent[1] / 100)
            coords = []
            slider_1 = self.value()[0] / self.Range
            slider_2 = self.value()[1] / self.Range
            if self.type == "x":
                for i in [rrz[1], rrz[0]]:
                    for j in [rry[1], rry[0]]:
                        for k in [(1-slider_1) * rrx_full[1], (1-slider_2) * rrx_full[1]]:
                            coords.append([i, k, j])
            elif self.type == "y":
                for i in [rrz[1], rrz[0]]:
                    for j in [slider_1 * rry_full[1], slider_2 * rry_full[1]]:
                        for k in [rrx[1], rrx[0]]:
                            coords.append([i, k, j])
            else:
                for i in [slider_1 * rrz_full[1], slider_2 * rrz_full[1]]:
                    for j in [rry[1], rry[0]]:
                        for k in [rrx[1], rrx[0]]:
                            coords.append([i, k, j])
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
                np.reshape(np.asarray(coords), (8, 3)),
                np.asarray(faces) - 1,
            )
        else:
            rrx_full = self.parent().data_to_layer_itf.render_range_y
            rrx = rrx_full.copy()
            rrx[0] += (rrx_full[1] - rrx_full[0]) * self.parent().render_range_slider_x_percent[0] / 100
            rrx[1] -= (rrx_full[1] - rrx_full[0]) * (1 - self.parent().render_range_slider_x_percent[1] / 100)
            rry_full = self.parent().data_to_layer_itf.render_range_x
            rry = rry_full.copy()
            rry[0] += (rry_full[1] - rry_full[0]) * self.parent().render_range_slider_y_percent[0] / 100
            rry[1] -= (rry_full[1] - rry_full[0]) * (1 - self.parent().render_range_slider_y_percent[1] / 100)
            coords = []
            slider_1 = self.value()[0] / self.Range
            slider_2 = self.value()[1] / self.Range
            if self.type == "x":
                for j in [rry[1], rry[0]]:
                    for k in [(1-slider_1) * rrx_full[1], (1-slider_2) * rrx_full[1]]:
                        coords.append([k, j])
            elif self.type == "y":
                for j in [slider_1 * rry_full[1], slider_2 * rry_full[1]]:
                    for k in [rrx[1], rrx[0]]:
                        coords.append([k, j])
            faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
            return (
                np.reshape(np.asarray(coords), (4, 2)),
                np.asarray(faces) - 1,
            )
