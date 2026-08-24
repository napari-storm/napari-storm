from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSlider

from ..CustomErrors import ParentError


class GridPlaneSlider(QSlider):
    def __init__(
        self,
        data_to_layer_interface,
        type_of_slider,
        parent=None,
        init_range=(0, 100),
        init_value=50,
    ):
        super().__init__()
        self._parent = parent
        self.setRange(init_range[0], init_range[1])
        self.setOrientation(Qt.Horizontal)
        self.setSingleStep(1)
        self.setPageStep(1)
        self.init_value = int(init_value)
        self.setValue(self.init_value)
        self.itf = data_to_layer_interface
        self.type_of_slider = type_of_slider
        self.valueChanged.connect(self._apply_value)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError("Cannot change parent of existing Widget")

    def _apply_value(self, value=None):
        value = self.value() if value is None else int(value)
        if self.type_of_slider == "z_pos":
            self.itf.update_grid_plane(z_pos=value)
        elif self.type_of_slider == "line_thickness":
            self.itf.update_grid_plane(line_thickness=value)
        elif self.type_of_slider == "opacity":
            self.itf.update_grid_plane(opacity=value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            changed = self.value() != self.init_value
            self.setValue(self.init_value)
            if not changed:
                self._apply_value()
            event.accept()
            return
        # Let QSlider track the handle.  The previous event.x()/2.5 mapping
        # assumed a 250 px-wide widget and jumped as soon as a drag began.
        super().mousePressEvent(event)
