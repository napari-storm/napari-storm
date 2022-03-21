from PyQt5.QtWidgets import QSlider
from .CustomErrors import *
from PyQt5.QtCore import Qt


class GridPlaneSlider(QSlider):
    def __init__(self, data_to_layer_interface, type_of_slider, parent=None, init_range=(0, 100), init_value=50):
        super().__init__()
        self._parent = parent
        self.setRange(init_range[0], init_range[1])
        self.setOrientation(Qt.Horizontal)
        self.setSingleStep(1)
        self.init_value = int(init_value)
        self.setValue(self.init_value)
        self.mouse_is_pressed = False
        self.itf = data_to_layer_interface
        self.type_of_slider = type_of_slider

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def mousePressEvent(self, event):
        if event.button() == 4:
            self.setValue(self.init_value)
            if self.type_of_slider == 'z_pos':
                self.itf.update_grid_plane(z_pos=self.value())
            elif self.type_of_slider == 'line_thickness':
                self.itf.update_grid_plane(line_thickness=self.value())
            elif self.type_of_slider == 'opacity':
                self.itf.update_grid_plane(opacity=self.value())
        self.mouse_is_pressed = True

    def mouseReleaseEvent(self, event):
        self.mouse_is_pressed = False

    def mouseMoveEvent(self, event):
        if self.mouse_is_pressed:
            delta_x = event.x() / 2.5 - self.value()
            self.setValue(int(self.value() + delta_x))
            if self.type_of_slider == 'z_pos':
                self.itf.update_grid_plane(z_pos=self.value())
            elif self.type_of_slider == 'line_thickness':
                self.itf.update_grid_plane(line_thickness=self.value())
            elif self.type_of_slider == 'opacity':
                self.itf.update_grid_plane(opacity=self.value())



