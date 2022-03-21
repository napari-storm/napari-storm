from PyQt5.QtGui import (
    QPixmap,
    QIcon,
    QColor,
)

from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QComboBox,
    QWidget,
)

from PyQt5.QtWidgets import (
    QGridLayout,
    QSlider,
    QCheckBox,
)

from PyQt5.QtCore import Qt
from .CustomErrors import *


class ChannelControls(QWidget):
    """A QT widget that is created for every channel,
    which provides the visual controls"""

    def __init__(
            self,
            parent,
            name,
            channel_index,
            localization_datasets=None,
            data_to_layer_itf=None,
            z_color_encoding_mode=None,
            render_gaussian_mode=None,
    ):
        from .RangeSlider import RangeSlider

        super().__init__()
        # Attributes
        self._parent = parent
        self._name = name
        self._channel_index = channel_index

        self.localization_datasets = localization_datasets
        if not self.localization_datasets:
            self.localization_datasets=parent.localization_datasets

        self.data_to_layer_itf = data_to_layer_itf
        if not self.data_to_layer_itf:
            self.data_to_layer_itf = parent.data_to_layer_itf

        self._z_color_encoding_mode = z_color_encoding_mode
        if not self._z_color_encoding_mode:
            self._z_color_encoding_mode = parent.z_color_encoding_mode

        self._render_gaussian_mode = render_gaussian_mode
        if not self._render_gaussian_mode:
            self._render_gaussian_mode = parent.render_gaussian_mode

        self.show_channel_state = True
        self.colormap_range_low = 0.0
        self.colormap_range_high = 0.0
        self.opacity_slider_setting = 0.0
        self.colormap_index = 0

        self.Label = QLabel()
        self.Label.setText(name)

        self.Bshow_channel = QCheckBox()
        self.Bshow_channel.setChecked(self.show_channel_state)
        self.Bshow_channel.stateChanged.connect(self.show_channel)

        self.Breset = QPushButton()
        self.Breset.setText('Reset')
        self.Breset.clicked.connect(self.reset)

        self.Slider_colormap_range = RangeSlider(parent=parent)
        self.Slider_colormap_range.setRange(0, 100)
        self.Slider_colormap_range.setValue((10, 90))
        self.Slider_colormap_range.valueChanged.connect(self.adjust_colormap_range)

        self.Slider_opacity = QSlider(Qt.Horizontal)
        self.Slider_opacity.setRange(0, 100)
        self.Slider_opacity.setValue(100)
        self.Slider_opacity.hide()
        self.Slider_opacity.valueChanged.connect(self.adjust_z_color_encoding_opacity)

        self.Colormap_selector = QComboBox()
        items = []
        icons = []
        for cmap in self.data_to_layer_itf.colormap:
            items.append(cmap.name)
            pixmap = QPixmap(20, 20)
            color = QColor(
                int(cmap.colors[1][0] * 255),
                int(cmap.colors[1][1] * 255),
                int(cmap.colors[1][2] * 255),
                int(cmap.colors[1][3] * 255),
            )
            pixmap.fill(color)
            icons.append(QIcon(pixmap))
        self.Colormap_selector.addItems(items)
        for i in range(len(items)):
            self.Colormap_selector.setItemIcon(i, icons[i])
        self.Colormap_selector.setCurrentText(items[channel_index])
        self.Colormap_selector.currentIndexChanged.connect(self.change_color_map)

        self.layout = QGridLayout()
        self.layout.addWidget(self.Label, 0, 0)
        self.layout.addWidget(self.Breset, 0, 1)
        self.layout.addWidget(self.Bshow_channel, 0, 2)
        self.layout.addWidget(self.Colormap_selector, 1, 0, 1, 3)
        self.layout.addWidget(self.Slider_colormap_range, 2, 0, 1, 3)
        self.layout.addWidget(self.Slider_opacity, 3, 0, 1, 3)
        self.layout.setColumnStretch(0, 3)
        self.setLayout(self.layout)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    @property
    def z_color_encoding_mode(self):
        return self.parent.z_color_encoding_mode

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        self.parent.z_color_encoding_mode = value

    @property
    def render_gaussian_mode(self):
        return self.parent.render_gaussian_mode

    @render_gaussian_mode.setter
    def render_gaussian_mode(self, value):
        self.parent.render_gaussian_mode = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise StaticAttributeError('Name of Channel should not be changed!')

    @property
    def channel_index(self):
        return self._channel_index

    @channel_index.setter
    def channel_index(self, value):
        raise StaticAttributeError('Channel index should not be changed!')

    def adjust_colormap_range(self):

        tmp_range = self.Slider_colormap_range.getRange()

        self.colormap_range_low = tmp_range[0]
        self.colormap_range_high = tmp_range[1]

        tmp_dset_obj = self.localization_datasets[self.channel_index]
        tmp_layer = tmp_dset_obj.napari_layer_ref

        tmp_layer.contrast_limits = tmp_range

    def adjust_z_color_encoding_opacity(self):

        tmp_opacity = self.Slider_opacity.value()

        self.opacity_slider_setting = tmp_opacity

        tmp_dset_obj = self.localization_datasets[self.channel_index]
        tmp_layer = tmp_dset_obj.napari_layer_ref

        tmp_layer.opacity = tmp_opacity / 100.0

    def reset(self):
        self.Slider_colormap_range.setValue((10.0, 90.0))
        self.Slider_opacity.setValue(100.0)

    def change_color_map(self):
        tmp_cmap_index = self.Colormap_selector.currentIndex()
        self.colormap_index = tmp_cmap_index

        tmp_dset_obj = self.localization_datasets[self.channel_index]
        tmp_layer = tmp_dset_obj.napari_layer_ref

        if self.z_color_encoding_mode:

            assert self.render_gaussian_mode == 0
            tmp_layer.colormap = 'hsv'

        else:
            tmp_layer.colormap = self.data_to_layer_itf.colormap[tmp_cmap_index]

    def show_channel(self):

        tmp_dset_obj = self.localization_datasets[self.channel_index]
        tmp_layer = tmp_dset_obj.napari_layer_ref

        if self.show_channel_state:
            # Hide the channel

            self.show_channel_state = False
            self.Bshow_channel.setChecked(False)

            tmp_layer.opacity = 0.0

            self.Slider_colormap_range.hide()
            self.Colormap_selector.hide()
            self.Slider_opacity.hide()

        else:
            # Show the channel

            self.show_channel_state = True
            self.Bshow_channel.setChecked(True)

            if self.z_color_encoding_mode:

                self.Slider_opacity.setValue(self.opacity_slider_setting)

                self.Slider_colormap_range.hide()
                self.Colormap_selector.hide()

                self.Slider_opacity.show()

            else:

                tmp_layer.opacity = 1.0

                self.Slider_colormap_range.show()
                self.Colormap_selector.show()
                self.Slider_opacity.hide()
