import os.path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import QTabWidget, QWidget, QGridLayout, QPushButton, QLabel, QComboBox, QLineEdit, QCheckBox, \
    QFormLayout, QSlider, QListWidget
from qtpy import QtCore

from napari_storm.CustomErrors import ParentError
from napari_storm.pyqt.GridPlaneSlider import GridPlaneSlider
from napari_storm.pyqt.PyQTvisuals import QHSeperationLine
from napari_storm.pyqt.RenderRangeSlider import RangeSlider2
from napari_storm.ns_constants import standard_colors
from .Test_Mode import TestModeWindow
from .DataFilter import DataFilterWindow
from .DataAdjustment import DataAdjustmentWindow
from .pyqt.detachable_tab import DetachableTabWidget
from .file_and_data_recognition import *


class NapariStormGUI(QWidget):
    def __init__(self):
        super().__init__()

        # GUI
        self.setAcceptDrops(True)

        self.tabs = DetachableTabWidget()
        # Tabs
        self.data_control_tab = QWidget()
        self.infos_tab = QWidget()
        self.decorator_tab = QWidget()
        self.datafilter_tab = DataFilterWindow(parent=self)
        self.data_adjustment_tab = DataAdjustmentWindow(parent=self)

        self.test_mode_tab = TestModeWindow(parent=self)

        self.tabs.addTab(self.data_control_tab, 'Data Controls')
        self.data_controls_tab_layout = QGridLayout()

        # self.tabs.tabBarClicked.connect(self.handle_tab_bar_clicked)

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

        self.Bimport_by_file_recognition = QPushButton()
        self.Bimport_by_file_recognition.setText("File Recognition Import")
        self.data_controls_tab_layout.addWidget(self.Bimport_by_file_recognition, 2, 0, 1, 2)
        self.Bimport_by_file_recognition.clicked.connect(
            lambda:self.open_localization_data_file_and_get_dataset(merge=False, file_recognition=True))

        self.Bcustom_import = QPushButton()
        self.Bcustom_import.setText("Custom Import")
        self.data_controls_tab_layout.addWidget(self.Bcustom_import, 2, 2, 1, 2)
        self.Bcustom_import.clicked.connect(
            lambda: self.open_localization_data_file_and_get_dataset(merge=False, custom_import=True))

        self.Lresetview = QLabel()
        self.Lresetview.setText('Reset view:')
        self.data_controls_tab_layout.addWidget(self.Lresetview, 3, 0)

        self.Baxis_xy = QPushButton()
        self.Baxis_xy.setText('XY')
        self.Baxis_xy.clicked.connect(lambda: self.change_camera(set_view_to='XY'))
        self.Baxis_xy.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xy, 3, 1)

        self.Baxis_yz = QPushButton()
        self.Baxis_yz.setText('YZ')
        self.Baxis_yz.clicked.connect(lambda: self.change_camera(set_view_to='YZ'))
        self.Baxis_yz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_yz, 3, 2)

        self.Baxis_xz = QPushButton()
        self.Baxis_xz.setText('XZ')
        self.Baxis_xz.clicked.connect(lambda: self.change_camera(set_view_to='XZ'))
        self.Baxis_xz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xz, 3, 3)

        self.Lrenderoptions = QLabel()
        self.Lrenderoptions.setText('Rendering options:')
        self.data_controls_tab_layout.addWidget(self.Lrenderoptions, 4, 0)

        self.Brenderoptions = QComboBox()
        self.Brenderoptions.addItems(self.gaussian_render_modes)
        self.Brenderoptions.currentIndexChanged.connect(self._render_options_changed)
        self.data_controls_tab_layout.addWidget(self.Brenderoptions, 4, 1, 1, 3)

        self.Lsigma_xy = QLabel()
        self.Lsigma_xy.setText('FWHM in XY [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy, 5, 0)

        self.Lsigma_z = QLabel()
        self.Lsigma_z.setText('FWHM in Z [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_z, 6, 0)

        self.Lsigma_xy_min = QLabel()
        self.Lsigma_xy_min.setText('Min. FWHM in XY [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy_min, 7, 0)

        self.Lsigma_z_min = QLabel()
        self.Lsigma_z_min.setText('Min. FWHM in Z [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lsigma_z_min, 8, 0)

        self.Esigma_xy = QLineEdit()
        self.Esigma_xy.setText(str(self.render_fixed_gauss_sigma_xy_nm * 2.354))
        self.Esigma_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_xy, 5, 1, 1, 3)
        self.typing_timer_sigma = QtCore.QTimer()
        self.typing_timer_sigma.setSingleShot(True)
        self.typing_timer_sigma.timeout.connect(self.update_sigma)

        self.Esigma_z = QLineEdit()
        self.Esigma_z.setText(str(self.render_fixed_gauss_sigma_xy_nm * 2.354))
        self.Esigma_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_z, 6, 1, 1, 3)

        self.Esigma_min_xy = QLineEdit()
        self.Esigma_min_xy.setText(str(self.render_var_gauss_sigma_min_xy_nm * 2.354))
        self.Esigma_min_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_xy, 7, 1, 1, 3)

        self.Esigma_min_z = QLineEdit()
        self.Esigma_min_z.setText(str(self.render_var_gauss_sigma_min_z_nm * 2.354))
        self.Esigma_min_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_z, 8, 1, 1, 3)

        self.HL1 = QHSeperationLine()
        self.data_controls_tab_layout.addWidget(self.HL1, 9, 0, 1, 4)

        self.Lrangex = QLabel()
        self.Lrangex.setText('X-range')
        self.data_controls_tab_layout.addWidget(self.Lrangex, 10, 0)

        self.Lrangey = QLabel()
        self.Lrangey.setText('Y-range')
        self.data_controls_tab_layout.addWidget(self.Lrangey, 11, 0)

        self.Lrangez = QLabel()
        self.Lrangez.setText('Z-range')
        self.data_controls_tab_layout.addWidget(self.Lrangez, 12, 0)

        self.Srender_rangex = RangeSlider2(parent=self, type='x')
        self.data_controls_tab_layout.addWidget(self.Srender_rangex, 10, 1, 1, 3)

        self.Srender_rangey = RangeSlider2(parent=self, type='y')
        self.data_controls_tab_layout.addWidget(self.Srender_rangey, 11, 1, 1, 3)

        self.Srender_rangez = RangeSlider2(parent=self, type='z')
        self.data_controls_tab_layout.addWidget(self.Srender_rangez, 12, 1, 1, 3)

        self.Breset_render_range = QPushButton()
        self.Breset_render_range.setText('Reset Render Range')
        self.Breset_render_range.clicked.connect(self.reset_render_range)
        self.data_controls_tab_layout.addWidget(self.Breset_render_range, 13, 0, 1, 4)

        self.HL2 = QHSeperationLine()
        self.data_controls_tab_layout.addWidget(self.HL2, 14, 0, 1, 4)

        self.Cscalebar = QCheckBox()
        self.Cscalebar.stateChanged.connect(self.scalebar_state_changed)
        self.Cscalebar.setText("Scalebar")
        self.data_controls_tab_layout.addWidget(self.Cscalebar, 15, 0, 1, 1)

        self.Bz_color_coding = QCheckBox()
        self.Bz_color_coding.setText('Activate Rainbow colorcoding in Z')
        self.Bz_color_coding.stateChanged.connect(self.colorcoding)
        self.data_controls_tab_layout.addWidget(self.Bz_color_coding, 15, 2, 1, 2)

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText('Size of Scalebar [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lscalebarsize, 16, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setText('500')
        self.Esbsize.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sbscale)
        )
        self.data_controls_tab_layout.addWidget(self.Esbsize, 16, 1, 1, 1)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self.data_to_layer_itf.scalebar)

        # visual_controls
        self.channel_controls_widget_layout = QFormLayout()
        self.channel_controls_placeholder = QWidget()
        self.data_controls_tab_layout.addWidget(self.channel_controls_placeholder, 19, 0, 1, 4)
        self.channel_controls_placeholder.setLayout(self.channel_controls_widget_layout)

        self.Lcolor_encoding_scalebar = ZColorCodingScaleBarWidget()
        self.Lcolor_encoding_scalebar.hide()

        self.data_controls_tab_layout.addWidget(self.Lcolor_encoding_scalebar, 20, 0, 1, 4)



        # infos tab
        self.infos_tab_layout = QGridLayout()
        self.Lnumberoflocs = TestListView(self.localization_datasets, parent=self)
        self.Lnumberoflocs.addItem(
            'STATISTICS \nWaiting for Data... \nImport or drag file here'
        )

        self.Lnumberoflocs.itemDoubleClicked.connect(self.Lnumberoflocs.remove_dataset)
        self.infos_tab_layout.addWidget(self.Lnumberoflocs, 0, 0)

        # Decorators tab
        self.decorator_tab_layout = QFormLayout()

        self.Lgrid_plane = QLabel()
        self.Lgrid_plane.setText("Grid Plane")
        self.Lgrid_plane.setFont(QFont('Arial', 10))
        self.decorator_tab_layout.addRow(self.Lgrid_plane)

        self.Cgrid_plane = QCheckBox()
        self.Cgrid_plane.stateChanged.connect(self.grid_plane)
        self.decorator_tab_layout.addRow("Grid plane activated?", self.Cgrid_plane)

        self.Egrid_line_distance = QLineEdit()
        self.Egrid_line_distance.setText(str(self.grid_plane_line_distance_um))
        self.Egrid_line_distance.textChanged.connect(lambda: self._start_typing_timer(self.typing_timer_grid))
        self.decorator_tab_layout.addRow("Grid line distance [µm]:", self.Egrid_line_distance)

        self.typing_timer_grid = QtCore.QTimer()
        self.typing_timer_grid.setSingleShot(True)
        self.typing_timer_grid.timeout.connect(self.update_grid_plane_line_distance)

        self.Sgrid_line_thickness = GridPlaneSlider(parent=self, data_to_layer_interface=self.data_to_layer_itf,
                                                    type_of_slider='line_thickness', init_range=(1, 100),
                                                    init_value=50)
        self.decorator_tab_layout.addRow("Grid line thickness:", self.Sgrid_line_thickness)

        self.Sgrid_z_pos = GridPlaneSlider(parent=self, data_to_layer_interface=self.data_to_layer_itf,
                                           type_of_slider='z_pos', init_range=(0, 100),
                                           init_value=50)
        self.decorator_tab_layout.addRow("Z Pos:", self.Sgrid_z_pos)

        self.Bgrid_plane_color = QComboBox()
        self.Bgrid_plane_color.addItems(standard_colors)
        self.Bgrid_plane_color.currentIndexChanged.connect(self.update_grid_plane_color)
        self.decorator_tab_layout.addRow("Grid line color:", self.Bgrid_plane_color)

        self.Sgrid_plane_opacity = GridPlaneSlider(parent=self, data_to_layer_interface=self.data_to_layer_itf,
                                                   type_of_slider='opacity', init_range=(0, 100),
                                                   init_value=100 * self.grid_plane_opacity)
        self.decorator_tab_layout.addRow("Grid plane opacity:", self.Sgrid_plane_opacity)

        self.HL3 = QHSeperationLine()
        self.decorator_tab_layout.addRow(self.HL3)

        self.Lrender_range_box = QLabel()
        self.Lrender_range_box.setText("Render Range Box")
        self.Lrender_range_box.setFont(QFont('Arial', 10))
        self.decorator_tab_layout.addRow(self.Lrender_range_box)

        self.Brender_range_box_color = QComboBox()
        self.Brender_range_box_color.addItems(standard_colors)
        self.Brender_range_box_color.currentIndexChanged.connect(self.update_render_range_box_color)
        self.decorator_tab_layout.addRow("Render Range Box color:", self.Brender_range_box_color)

        self.Srender_range_box_opacity = QSlider()
        self.Srender_range_box_opacity.setOrientation(Qt.Horizontal)
        self.Srender_range_box_opacity.setRange(0, 100)
        self.Srender_range_box_opacity.setSingleStep(1)
        self.Srender_range_box_opacity.setValue(int(self.render_range_box_opacity * 100))
        self.Srender_range_box_opacity.valueChanged.connect(self.update_render_range_box_opacity)
        self.decorator_tab_layout.addRow("Render Range Box opacity:", self.Srender_range_box_opacity)

        self.decorator_tab.setLayout(self.decorator_tab_layout)
        self.layout = QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.data_controls_tab_layout.setColumnStretch(0, 4)
        self.data_control_tab.setLayout(self.data_controls_tab_layout)
        self.infos_tab.setLayout(self.infos_tab_layout)

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

    """def handle_tab_bar_clicked(self, index):

        if self.tabs.tabText(index) == 'Data Filter':
            self.reset_render_range()"""

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
        self.Egrid_line_distance.hide()
        self.Sgrid_line_thickness.hide()
        self.Sgrid_z_pos.hide()
        self.Cgrid_plane.hide()
        self.Bgrid_plane_color.hide()
        self.HL1.hide()
        self.HL2.hide()

    def hide_testing_mode(self):
        if self.testing_mode_enabled:
            self.tabs.addTab(self.test_mode_tab, 'Test Mode')

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
        self.tabs.addTab(self.datafilter_tab, 'Data Filter')
        self.tabs.addTab(self.data_adjustment_tab, "Data adjustment")
        self.HL1.show()
        self.HL2.show()

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


class TestListView(QListWidget):
    """Custom ListView Widget -> The Log, allows, d&d and displays infos on the files"""

    def __init__(self, datasets, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))
        self.datasets = datasets

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def update_dataset_ref(self):
        self.datasets = self.parent.localization_datasets

    def show_infos(self, filename, idx):
        self.update_dataset_ref()
        """Print Infos about files in Log"""
        if self.datasets[idx].zdim_present:
            self.addItem(
                "Statistics\n"
                + f"File: {filename}\n"
                + f"Dataset-type: {self.datasets[idx].dataset_type}\n"
                + f"Number of locs: {len(self.datasets[idx].x_pos_nm)}\n"
                  f"Imagewidth: {np.round((max(self.datasets[idx].x_pos_nm) - min(self.datasets[idx].x_pos_nm))) / 1000}  µm\n"
                + f"Imageheigth: {np.round((max(self.datasets[idx].y_pos_nm) - min(self.datasets[idx].y_pos_nm))) / 1000}  µm\n"
                + f"Imagedepth: {np.round((max(self.datasets[idx].z_pos_nm) - min(self.datasets[idx].z_pos_nm))) / 1000}  µm\n"
            )
        else:
            self.addItem(
                "Statistics\n"
                + f"File: {filename}\n"
                + f"Dataset-type: {self.datasets[idx].dataset_type}\n"
                + f"Number of locs: {len(self.datasets[idx].x_pos_nm)}\n"
                  f"Imagewidth: {np.round((max(self.datasets[idx].x_pos_nm) - min(self.datasets[idx].x_pos_nm))) / 1000}  µm\n"
                + f"Imageheigth: {np.round((max(self.datasets[idx].y_pos_nm) - min(self.datasets[idx].y_pos_nm))) / 1000}  µm\n"

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


class ZColorCodingScaleBarWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.colorbar_set = False
        self.setFixedWidth(256)
        self.setFixedHeight(128)
        self.titel = QLabel("Z-color-encoding scalebar:")
        self.colorbar = QLabel("")

        self.label = QLabel("min                                                   max")

        self.layout = QFormLayout()
        self.layout.addRow(self.titel)
        self.layout.addRow(self.scalebar)
        self.layout.addRow(self.label)

        self.setLayout(self.layout)

    def set_pixmap(self, scalebar_pixmap):
        if not self.colorbar_set:
            self.colorbar.setPixmap(scalebar_pixmap.scaledToWidth(194))
            self.colorbar_set = True

