from napari_plugin_engine import napari_hook_implementation

from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QComboBox,
    QListWidget,
    QApplication
)

from PyQt5.QtWidgets import (
    QGridLayout,
    QLineEdit,
    QCheckBox,
    QTabWidget,
    QFormLayout,
    QSlider
)

from PyQt5 import QtCore

from PyQt5.QtCore import Qt
from .Exp_Controls import *
from .DataToLayerInterface import DataToLayerInterface
from .FileToLocalizationDataInterface import FileToLocalizationDataInterface
from .ChannelControls import ChannelControls
from .CustomErrors import *
from .GridPlaneSlider import *


class napari_storm(QWidget):
    """The Heart of this code: A Dock Widget, but also
    an object where everthing runs together"""

    def __init__(self, napari_viewer):

        from .RangeSlider2 import RangeSlider2

        super().__init__()
        # Interfaces
        self._data_to_layer_itf = DataToLayerInterface(parent=self, viewer=napari_viewer)
        self._file_to_data_itf = FileToLocalizationDataInterface(parent=self)

        # Attributes
        self.testing_mode_enabled = False

        self.localization_datasets = []
        self.n_datasets = 0
        self.viewer = napari_viewer

        self._render_gaussian_mode = 0
        self.gaussian_render_modes = ['Fixed-size gaussian',
                                      'Variable-size gaussian']

        self._z_color_encoding_mode = False

        self._grid_plane_enabled = False
        self.grid_plane_line_distance_um = 1

        self.render_fixed_gauss_sigma_xy_nm = 20 / 2.354
        self.render_fixed_gauss_sigma_z_nm = 20 / 2.354
        self.render_var_gauss_PSF_sigma_xy_nm = 300 / 2.354
        self.render_var_gauss_PSF_sigma_z_nm = 700 / 2.354

        self.render_var_gauss_sigma_min_xy_nm = 10 / 2.354
        self.render_var_gauss_sigma_min_z_nm = 10 / 2.354

        self.render_range_slider_x_percent = np.arange(2) * 100
        self.render_range_slider_y_percent = np.arange(2) * 100
        self.render_range_slider_z_percent = np.arange(2) * 100

        self.pixelsize_nm = []
        self._zdim = None
        self.channel = []

        # GUI
        self.setAcceptDrops(True)
        self.tabs = QTabWidget()
        self.data_control_tab = QWidget()
        self.infos_tab = QWidget()
        self.decorator_tab = QWidget()

        self.tabs.addTab(self.data_control_tab, 'Data Controls')

        self.data_controls_tab_layout = QGridLayout()

        # Set up the GUI
        self.Bopen = QPushButton()
        self.Bopen.clicked.connect(self.open_localization_data_file_and_get_dataset)
        self.Bopen.setText('Import File Dialog')
        self.data_controls_tab_layout.addWidget(self.Bopen, 0, 0, 1, 2)

        self.Bmerge_with_additional_file = QPushButton()
        self.Bmerge_with_additional_file.setText('Merge with additional file')
        self.data_controls_tab_layout.addWidget(self.Bmerge_with_additional_file, 0, 2, 1, 2)
        self.Bmerge_with_additional_file.clicked.connect(lambda:
                                                         self.open_localization_data_file_and_get_dataset(merge=True))

        self.Lresetview = QLabel()
        self.Lresetview.setText('Reset view:')
        self.data_controls_tab_layout.addWidget(self.Lresetview, 2, 0)

        self.Baxis_xy = QPushButton()
        self.Baxis_xy.setText('XY')
        self.Baxis_xy.clicked.connect(lambda: self.change_camera(type='XY'))
        self.Baxis_xy.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xy, 2, 1)

        self.Baxis_yz = QPushButton()
        self.Baxis_yz.setText('YZ')
        self.Baxis_yz.clicked.connect(lambda: self.change_camera(type='YZ'))
        self.Baxis_yz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_yz, 2, 2)

        self.Baxis_xz = QPushButton()
        self.Baxis_xz.setText('XZ')
        self.Baxis_xz.clicked.connect(lambda: self.change_camera(type='XZ'))
        self.Baxis_xz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xz, 2, 3)

        self.Lrenderoptions = QLabel()
        self.Lrenderoptions.setText('Rendering options:')
        self.data_controls_tab_layout.addWidget(self.Lrenderoptions, 3, 0)

        self.Brenderoptions = QComboBox()
        self.Brenderoptions.addItems(self.gaussian_render_modes)
        self.Brenderoptions.currentIndexChanged.connect(self._render_options_changed)
        self.data_controls_tab_layout.addWidget(self.Brenderoptions, 3, 1, 1, 3)

        self.Lsigma_xy = QLabel()
        self.Lsigma_xy.setText('FWHM in XY [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy, 4, 0)

        self.Lsigma_z = QLabel()
        self.Lsigma_z.setText('FWHM in Z [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_z, 5, 0)

        self.Lsigma_xy_min = QLabel()
        self.Lsigma_xy_min.setText('Min. FWHM in XY [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy_min, 6, 0)

        self.Lsigma_z_min = QLabel()
        self.Lsigma_z_min.setText('Min. FWHM in Z [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_z_min, 7, 0)

        self.Esigma_xy = QLineEdit()
        self.Esigma_xy.setText(str(self.render_fixed_gauss_sigma_xy_nm * 2.354))
        self.Esigma_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_xy, 4, 1, 1, 3)
        self.typing_timer_sigma = QtCore.QTimer()
        self.typing_timer_sigma.setSingleShot(True)
        self.typing_timer_sigma.timeout.connect(self.update_sigma)

        self.Esigma_z = QLineEdit()
        self.Esigma_z.setText(str(self.render_fixed_gauss_sigma_xy_nm * 2.354))
        self.Esigma_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_z, 5, 1, 1, 3)

        self.Esigma_min_xy = QLineEdit()
        self.Esigma_min_xy.setText(str(self.render_var_gauss_sigma_min_xy_nm * 2.354))
        self.Esigma_min_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_xy, 6, 1, 1, 3)

        self.Esigma_min_z = QLineEdit()
        self.Esigma_min_z.setText(str(self.render_var_gauss_sigma_min_z_nm * 2.354))
        self.Esigma_min_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_z, 7, 1, 1, 3)

        self.Lrangex = QLabel()
        self.Lrangex.setText('X-range')
        self.data_controls_tab_layout.addWidget(self.Lrangex, 8, 0)

        self.Lrangey = QLabel()
        self.Lrangey.setText('Y-range')
        self.data_controls_tab_layout.addWidget(self.Lrangey, 9, 0)

        self.Lrangez = QLabel()
        self.Lrangez.setText('Z-range')
        self.data_controls_tab_layout.addWidget(self.Lrangez, 10, 0)

        self.Srender_rangex = RangeSlider2(parent=self, type='x')
        self.data_controls_tab_layout.addWidget(self.Srender_rangex, 8, 1, 1, 3)

        self.Srender_rangey = RangeSlider2(parent=self, type='y')
        self.data_controls_tab_layout.addWidget(self.Srender_rangey, 9, 1, 1, 3)

        self.Srender_rangez = RangeSlider2(parent=self, type='z')
        self.data_controls_tab_layout.addWidget(self.Srender_rangez, 10, 1, 1, 3)

        self.Breset_render_range = QPushButton()
        self.Breset_render_range.setText('Reset Render Range')
        self.Breset_render_range.clicked.connect(self.reset_render_range)
        self.data_controls_tab_layout.addWidget(self.Breset_render_range, 11, 0, 1, 4)

        self.Cscalebar = QCheckBox()
        self.Cscalebar.stateChanged.connect(self.scalebar_state_changed)
        self.Cscalebar.setText("Scalebar")
        self.data_controls_tab_layout.addWidget(self.Cscalebar, 12, 0, 1, 1)

        self.Bz_color_coding = QCheckBox()
        self.Bz_color_coding.setText('Activate Rainbow colorcoding in Z')
        self.Bz_color_coding.stateChanged.connect(self.colorcoding)
        self.data_controls_tab_layout.addWidget(self.Bz_color_coding, 12, 2, 1, 2)

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText('Size of Scalebar [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lscalebarsize, 13, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setText('500')
        self.Esbsize.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sbscale)
        )
        self.data_controls_tab_layout.addWidget(self.Esbsize, 13, 1, 1, 1)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self.data_to_layer_itf.scalebar)

        self.Bstarttestmode = QPushButton()
        self.Bstarttestmode.setText("Start Test Mode")
        self.Bstarttestmode.clicked.connect(self.start_test_mode)
        self.data_controls_tab_layout.addWidget(self.Bstarttestmode, 0, 2, 1, 1)

        # visual_controls
        self.channel_controls_widget_layout = QFormLayout()
        self.channel_controls_placeholder = QWidget()
        self.data_controls_tab_layout.addWidget(self.channel_controls_placeholder, 17, 0, 1, 4)
        self.channel_controls_placeholder.setLayout(self.channel_controls_widget_layout)

        # infos tab
        self.infos_tab_layout = QGridLayout()
        self.Lnumberoflocs = TestListView(parent=self)
        self.Lnumberoflocs.addItem(
            'STATISTICS \nWaiting for Data... \nImport or drag file here'
        )

        self.Lnumberoflocs.itemDoubleClicked.connect(self.Lnumberoflocs.remove_dataset)
        self.infos_tab_layout.addWidget(self.Lnumberoflocs, 0, 0)

        # Decorators tab
        self.decorator_tab_layout = QFormLayout()
        #self.decorator_tab_layout = QGridLayout()

        self.Cgrid_plane = QCheckBox()
        #self.Cgrid_plane.setText("Grid plane activated?")
        self.Cgrid_plane.stateChanged.connect(self.grid_plane)
        #self.decorator_tab_layout.addWidget(self.Cgrid_plane, 0, 0)
        self.decorator_tab_layout.addRow("Grid plane activated?", self.Cgrid_plane)

        """self.Lgrid_line_distance = QLabel()
        self.Lgrid_line_distance.setText("Grid line distance [µm]:")
        self.decorator_tab_layout.addWidget(self.Lgrid_line_distance, 1, 0)"""

        self.Egrid_line_distance = QLineEdit()
        self.Egrid_line_distance.setText(str(self.grid_plane_line_distance_um))
        self.Egrid_line_distance.textChanged.connect(lambda: self._start_typing_timer(self.typing_timer_grid))
        #self.decorator_tab_layout.addWidget(self.Egrid_line_distance, 1, 1)
        self.decorator_tab_layout.addRow("Grid line distance [µm]:", self.Egrid_line_distance)

        self.typing_timer_grid = QtCore.QTimer()
        self.typing_timer_grid.setSingleShot(True)
        self.typing_timer_grid.timeout.connect(self.update_grid_plane_line_distance)

        """self.Lgrid_line_thickness = QLabel()
        self.Lgrid_line_thickness.setText("Grid line thickness:")
        self.decorator_tab_layout.addWidget(self.Lgrid_line_thickness, 2, 0)"""

        self.Sgrid_line_thickness = GridPlaneSlider(parent=self, data_to_layer_interface=self.data_to_layer_itf,
                                                    type_of_slider='line_thickness', init_range=(1, 100),
                                                    init_value=50)
        #self.decorator_tab_layout.addWidget(self.Sgrid_line_thickness, 2, 1)
        self.decorator_tab_layout.addRow("Grid line thickness:", self.Sgrid_line_thickness)

        """self.Lgrid_z_pos = QLabel()
        self.Lgrid_z_pos.setText("Z Pos:")
        self.decorator_tab_layout.addWidget(self.Lgrid_z_pos, 3, 0)"""

        self.Sgrid_z_pos = GridPlaneSlider(parent=self, data_to_layer_interface=self.data_to_layer_itf,
                                           type_of_slider='z_pos', init_range=(0, 100),
                                           init_value=50)
        #self.decorator_tab_layout.addWidget(self.Sgrid_z_pos, 3, 1)
        self.decorator_tab_layout.addRow("Z Pos:", self.Sgrid_z_pos)

        self.decorator_tab.setLayout(self.decorator_tab_layout)
        self.layout = QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.data_controls_tab_layout.setColumnStretch(0, 4)
        self.data_control_tab.setLayout(self.data_controls_tab_layout)
        self.infos_tab.setLayout(self.infos_tab_layout)

        # Init Function Calls
        custom_keys_and_scalebar(self)
        self.hide_non_available_widgets()
        self.hide_testing_mode()

    @property
    def grid_plane_enabled(self):
        return self._grid_plane_enabled

    @grid_plane_enabled.setter
    def grid_plane_enabled(self, value):
        if value == 1 or value == 0:
            self._grid_plane_enabled = value
            self.data_to_layer_itf.create_remove_grid_plane_state(value)
        else:
            raise ValueError(f"Grid Plane can only be enabled or disabled not {value} of type {type(value)}")

    @property
    def data_to_layer_itf(self):
        return self._data_to_layer_itf

    @data_to_layer_itf.setter
    def data_to_layer_itf(self, value):
        raise StaticAttributeError('the dataset to layer interface should never be changed')

    @property
    def file_to_data_itf(self):
        return self._file_to_itf

    @file_to_data_itf.setter
    def file_to_data_itf(self, value):
        raise StaticAttributeError('the file to dataset interface should never be changed')

    @property
    def render_gaussian_mode(self):
        return self._render_gaussian_mode

    @render_gaussian_mode.setter
    def render_gaussian_mode(self, value):
        if value == 0 or value == 1:
            self._render_gaussian_mode = value
        else:
            raise ValueError(f"Wrong value for gaussian render mode, either 0 or 1 not {value} of type {type(value)}")

    @property
    def z_color_encoding_mode(self):
        return self._z_color_encoding_mode

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        if value == 0 or value == 1:
            self._z_color_encoding_mode = value
        else:
            raise ValueError(f"Wrong value for Z colorcoding, either 0 or 1 not {value} of type {type(value)}")

    @property
    def zdim(self):
        return self._zdim

    @zdim.setter
    def zdim(self, bool):
        """3D or 2D Mode"""
        if not type(bool) == type(True):
            raise ValueError(f"Zdim present can either be true or false not {bool} of type {type(bool)}")
        if (self.zdim == True and bool == False) or (self.zdim == False and bool == True):
            raise DimensionError('Error while merging, combination of 2D and 3D datasets not possible')
        else:
            self._zdim = bool
        self.adjust_available_options_to_data_dimension()

    def update_grid_plane_line_distance(self):
        value_str = self.Egrid_line_distance.text()
        value = float(value_str)
        if value == 0:
            raise ValueError('Line Distance must be > 0')
        self.grid_plane_line_distance_um = value
        self.data_to_layer_itf.update_grid_plane(line_distance_nm=1000*value)

    def grid_plane(self):
        if self.Cgrid_plane.isChecked():
            self.Egrid_line_distance.show()
            self.Sgrid_line_thickness.show()
            self.Sgrid_z_pos.show()
            self.grid_plane_enabled = 1

        else:
            self.Egrid_line_distance.hide()
            self.Sgrid_line_thickness.hide()
            self.Sgrid_z_pos.hide()
            self.grid_plane_enabled = 0

    def hide_testing_mode(self):
        if not self.testing_mode_enabled:
            self.Bstarttestmode.hide()

    def start_test_mode(self):
        self.Bstarttestmode.hide()
        from .Test_Mode import TestModeWindow
        window = TestModeWindow(parent=self)
        window.show()
        window.exec_()

    def scalebar_state_changed(self):
        if self.Cscalebar.isChecked():
            self.Lscalebarsize.show()
            self.Esbsize.show()
        else:
            self.Lscalebarsize.hide()
            self.Esbsize.hide()
        self.data_to_layer_itf.scalebar()

    def clear_datasets(self):
        """Erase the current dataset and reset the viewer"""
        v = self.viewer
        for i in range(self.n_datasets):
            v.layers.remove(self.localization_datasets[i].name)
        self.n_datasets = 0
        self.localization_datasets = []
        if not len(self.channel) == 0:  # Remove Channel of older files
            for i in range(len(self.channel)):
                self.channel[i].hide()
            self.channel = []
        self.Cscalebar.setCheckState(False)
        self.reset_render_range(full_reset=True)
        self.Lnumberoflocs.clear()
        self.Bz_color_coding.setCheckState(False)
        self._zdim = None

    def reset_render_range(self, full_reset=False):
        v = napari.current_viewer()
        self.Srender_rangex.reset()
        self.Srender_rangey.reset()
        self.Srender_rangez.reset()
        if self.Cgrid_plane.isChecked():
            self.Cgrid_plane.setState(False)
        if not full_reset:
            self.data_to_layer_itf.update_layers(self)
            self.move_camera_center_to_render_range_center()

    def update_render_range(self, slider_type, values):
        if slider_type == 'x':
            self.render_range_slider_x_percent = values
        elif slider_type == 'y':
            self.render_range_slider_y_percent = values
        else:
            self.render_range_slider_z_percent = values
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(line_distance_nm=self.grid_plane_line_distance_um*1000)
        self.move_camera_center_to_render_range_center()

    def move_camera_center_to_render_range_center(self):
        if not (self.data_to_layer_itf.render_range_x[1] == -np.inf
                or self.data_to_layer_itf.render_range_y[1] == -np.inf
                or self.data_to_layer_itf.render_range_z[1] == -np.inf):
            tmp_x_center_nm = self.data_to_layer_itf.render_range_x[1] * (
                    self.render_range_slider_x_percent[0] / 100 + 0.5 * self.render_range_slider_x_percent[1] / 100
                    - 0.5 * self.render_range_slider_x_percent[0] / 100)
            tmp_y_center_nm = self.data_to_layer_itf.render_range_y[1] * (
                    self.render_range_slider_y_percent[0] / 100 + 0.5 * self.render_range_slider_y_percent[1] / 100
                    - 0.5 * self.render_range_slider_y_percent[0] / 100)
            tmp_z_center_nm = self.data_to_layer_itf.render_range_z[1] * (
                    self.render_range_slider_z_percent[0] / 100 + 0.5 * self.render_range_slider_z_percent[1] / 100
                    - 0.5 * self.render_range_slider_z_percent[0] / 100)
            self.data_to_layer_itf.camera[1] = (tmp_z_center_nm, tmp_y_center_nm, tmp_x_center_nm)
            self.viewer.camera.center = (tmp_z_center_nm, tmp_y_center_nm, tmp_x_center_nm)

    #### D and D
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            u = event.mimeData().urls()
            file = u[0].toString()[8:]
            self.open_localization_data_file_and_get_dataset(file_path=file)
        else:
            event.ignore()
        #####

    def hide_non_available_widgets(self):
        """Hide controls which are better untouched atm"""
        self.Srender_rangex.hide()
        self.Srender_rangey.hide()
        self.Lrangex.hide()
        self.Lrangey.hide()
        self.Cscalebar.hide()
        self.Brenderoptions.hide()
        self.Lrenderoptions.hide()
        self.Lsigma_xy.hide()
        self.Esigma_xy.hide()
        self.Lsigma_z.hide()
        self.Esigma_z.hide()
        self.Lsigma_xy_min.hide()
        self.Lsigma_z_min.hide()
        self.Esigma_min_xy.hide()
        self.Esigma_min_z.hide()
        self.Bz_color_coding.hide()
        self.Lscalebarsize.hide()
        self.Esbsize.hide()
        self.Bmerge_with_additional_file.hide()
        self.Srender_rangez.hide()
        self.Lrangez.hide()
        self.Lresetview.hide()
        self.Baxis_xy.hide()
        self.Baxis_yz.hide()
        self.Baxis_xz.hide()
        self.Breset_render_range.hide()
        """self.Lgrid_line_thickness.hide()
        self.Lgrid_line_distance.hide()
        self.Lgrid_z_pos.hide()"""
        self.Egrid_line_distance.hide()
        self.Sgrid_line_thickness.hide()
        self.Sgrid_z_pos.hide()
        self.Cgrid_plane.hide()

    def show_avaiable_widgets(self):
        """Show the Controls usable atm"""
        self.Srender_rangex.show()
        self.Srender_rangey.show()
        self.Lrangex.show()
        self.Lrangey.show()
        self.Cscalebar.show()
        self.Brenderoptions.show()
        self.Lrenderoptions.show()
        self.Lsigma_xy.show()
        self.Esigma_xy.show()
        self.Lsigma_z.show()
        self.Esigma_z.show()
        self.Bmerge_with_additional_file.show()
        self.Breset_render_range.show()
        self.Cgrid_plane.show()
        self.tabs.addTab(self.infos_tab, 'File Infos')
        self.tabs.addTab(self.decorator_tab, 'Decorators')

    def adjust_available_options_to_data_dimension(self):
        if self.zdim:
            self.Lrangez.show()
            self.Srender_rangez.show()
            self.Baxis_xy.show()
            self.Baxis_xz.show()
            self.Baxis_yz.show()
            self.Lresetview.show()
            if self.z_color_encoding_mode == 0:
                self.Bz_color_coding.show()
            self.Brenderoptions.show()
            self.Lrenderoptions.show()
            self.Esigma_z.show()
            self.Lsigma_z.show()
            self.viewer.dims.ndisplay = 3
        else:
            self.Lrangez.hide()
            self.Srender_rangez.hide()
            self.Esigma_z.hide()
            self.Lsigma_z.hide()
            self.viewer.dims.ndisplay = 3

    def add_channel(self, name='Channel'):
        """Adds a Channel in the visual controls"""
        self.channel.append(
            ChannelControls(parent=self, name=name, channel_index=len(self.channel))
        )
        self.channel_controls_widget_layout.addRow(self.channel[-1])

    def colorcoding(self):
        """Check if Colorcoding is choosen"""
        if self.Bz_color_coding.isChecked():
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.hide()
                self.channel[i].Slider_opacity.show()
                self.channel[i].Slider_colormap_range.hide()
                self.channel[i].reset()
                self.channel[i].Label.setText('Contrast ' + self.channel[i].name)
                self.z_color_encoding_mode = True
        else:
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.show()
                self.channel[i].Slider_opacity.hide()
                self.channel[i].Slider_colormap_range.show()
                self.channel[i].Label.setText('Opacity ' + self.channel[i].name)
                self.channel[i].reset()
                self.z_color_encoding_mode = False
        self.data_to_layer_itf.update_layers(self)

    def _render_options_changed(self):
        if self.Brenderoptions.currentText() == self.gaussian_render_modes[1]:
            self.render_gaussian_mode = 1
            self.Lsigma_xy.setText('PSF FWHM in XY [nm]')
            self.Lsigma_z.setText('PSF FWHM in Z [nm]')
            self.Esigma_xy.setText('300')
            self.Esigma_z.setText('700')
            self.Bz_color_coding.hide()
            self.Bz_color_coding.setCheckState(False)
            self.Esigma_min_xy.show()
            if self.zdim:
                self.Esigma_min_z.show()
                self.Lsigma_z_min.show()
            self.Lsigma_xy_min.show()

        else:
            self.render_gaussian_mode = 0
            self.Lsigma_z.setText('FWHM in Z [nm]')
            self.Lsigma_xy.setText('FWHM in XY [nm]')
            self.Esigma_xy.setText('20')
            self.Esigma_z.setText('20')
            self.Bz_color_coding.show()
            self.Esigma_min_xy.hide()
            self.Esigma_min_z.hide()
            self.Lsigma_xy_min.hide()
            self.Lsigma_z_min.hide()
        self.data_to_layer_itf.update_layers(self)

    def _start_typing_timer(self, timer):
        timer.start(500)

    def change_camera(self, type='XY'):
        v = napari.current_viewer()
        values = {}
        if type == 'XY':
            v.camera.angles = (0, 0, -90)
        elif type == 'XZ':
            v.camera.angles = (180, 180, -180)
        else:
            v.camera.angles = (0, 90, 0)
        v.camera.center = self.data_to_layer_itf.camera[1]
        v.camera.update(values)

    def update_sigma(self):
        if self.render_gaussian_mode == 0:
            # fixed gaussian
            self.render_fixed_gauss_sigma_xy_nm = float(self.Esigma_xy.text()) / 2.354
            self.render_fixed_gauss_sigma_z_nm = float(self.Esigma_z.text()) / 2.354
        else:
            self.render_var_gauss_PSF_sigma_xy_nm = float(self.Esigma_xy.text()) / 2.354
            self.render_var_gauss_PSF_sigma_z_nm = float(self.Esigma_z.text()) / 2.354
            self.render_var_gauss_sigma_min_xy_nm = float(self.Esigma_min_xy.text()) / 2.354
            self.render_var_gauss_sigma_min_z_nm = float(self.Esigma_min_z.text()) / 2.354
        self.data_to_layer_itf.update_layers()

    def open_localization_data_file_and_get_dataset(self, merge=False, file_path=None):
        self.show_avaiable_widgets()
        if not merge:
            self.clear_datasets()
            self.data_to_layer_itf.reset_render_range_and_offset()

        datasets = self._file_to_data_itf.open_localization_data_file_and_get_dataset(file_path=file_path)
        if datasets[-1].zdim_present:
            self.zdim = True
        else:
            self.zdim = False
        for i in range(len(datasets)):
            self.localization_datasets.append(datasets[i])
            self.n_datasets += 1
            self.create_layer(self.localization_datasets[-1], idx=i, merge=merge)

    def get_dataset_from_test_mode(self, datasets):
        self.show_avaiable_widgets()
        self.clear_datasets()
        if datasets[-1].zdim_present:
            self.zdim = True
        else:
            self.zdim = False
        for i in range(len(datasets)):
            self.localization_datasets.append(datasets[i])
            self.n_datasets += 1
            self.create_layer(self.localization_datasets[-1], idx=i)

    def create_layer(self, dataset, idx=-1, merge=False):
        self.adjust_available_options_to_data_dimension()
        self.Lnumberoflocs.show_infos(filename=dataset.name, idx=idx)
        self.localization_datasets[idx].update_locs()
        self.data_to_layer_itf.create_new_layer(dataset, layer_name=dataset.name, idx=idx, merge=merge)
        self.add_channel(name=dataset.name)
        self.channel[-1].change_color_map()
        self.channel[-1].adjust_colormap_range()
        if dataset.sigma_present:
            self.Brenderoptions.setCurrentIndex(1)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return napari_storm


class TestListView(QListWidget):
    """Custom ListView Widget -> The Log, allows, d&d and displays infos on the files"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def show_infos(self, filename, idx):
        """Print Infos about files in Log"""
        if self.parent.localization_datasets[idx].zdim_present:
            self.addItem(
                "Statistics\n"
                + f"File: {filename}\n"
                + f"Number of locs: {len(self.parent.localization_datasets[idx].locs.x_pos_pixels)}\n"
                  f"Imagewidth: {np.round((max(self.parent.localization_datasets[idx].locs.x_pos_pixels) - min(self.parent.localization_datasets[idx].locs.x_pos_pixels)) * self.parent.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n"
                + f"Imageheigth: {np.round((max(self.parent.localization_datasets[idx].locs.y_pos_pixels) - min(self.parent.localization_datasets[idx].locs.y_pos_pixels)) * self.parent.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n"
                + f"Imagedepth: {np.round((max(self.parent.localization_datasets[idx].locs.z_pos_pixels) - min(self.parent.localization_datasets[idx].locs.z_pos_pixels)) * self.parent.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n"
                + f"Intensity per localization\nmean: {np.round(np.mean(self.parent.localization_datasets[idx].locs.photon_count), 3)}\nmax: "
                + f"{np.round(max(self.parent.localization_datasets[idx].locs.photon_count), 3)}\nmin:"
                + f" {np.round(min(self.parent.localization_datasets[idx].locs.photon_count), 3)}\n"
            )
        else:
            self.addItem(
                "Statistics\n"
                + f"File: {filename}\n"
                + f"Number of locs: {len(self.parent.localization_datasets[idx].locs.x_pos_pixels)}\n"
                  f"Imagewidth: {np.round((max(self.parent.localization_datasets[idx].locs.x_pos_pixels) - min(self.parent.localization_datasets[idx].locs.x_pos_pixels)) * self.parent.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n"
                + f"Imageheigth: {np.round((max(self.parent.localization_datasets[idx].locs.y_pos_pixels) - min(self.parent.localization_datasets[idx].locs.y_pos_pixels)) * self.parent.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n"
                + f"Intensity per localization\nmean: {np.round(np.mean(self.parent.localization_datasets[idx].locs.photon_count), 3)}\nmax: "
                + f"{np.round(max(self.parent.localization_datasets[idx].locs.photon_count), 3)}\nmin:"
                + f" {np.round(min(self.parent.localization_datasets[idx].locs.photon_count), 3)}\n"
            )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            u = event.mimeData().urls()
            file = u[0].toString()[8:]
            self.parent.file_to_data_itf.open_localization_data_file_and_get_dataset(file_path=file)
        else:
            event.ignore()

    def remove_dataset(self, item):
        print("Dataset removal not implemented yet...", item)
