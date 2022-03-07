from napari_plugin_engine import napari_hook_implementation

from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QComboBox,
    QListWidget,
)

from PyQt5.QtWidgets import (
    QGridLayout,
    QLineEdit,
    QCheckBox,
    QTabWidget,
    QFormLayout,
)

from PyQt5 import QtCore

from PyQt5.QtCore import Qt
from .Exp_Controls import *
from .DataToLayerInterface import DataToLayerInterface
from .FileToLocalizationDataInterface import FileToLocalizationDataInterface
from .ChannelControls import ChannelControls
from .CustomErrors import *


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
        self.localization_datasets = []
        self.n_datasets = 0
        self.viewer = napari_viewer

        self._render_gaussian_mode = 0
        self.gaussian_render_modes = ['Fixed-size gaussian',
                                      'Variable-size gaussian']

        self._z_color_encoding_mode = False
        self.render_fixed_gauss_sigma_xy_nm = 20 / 2.354
        self.render_fixed_gauss_sigma_z_nm = 20 / 2.354
        self.render_var_gauss_PSF_sigma_xy_nm = 300 / 2.354
        self.render_var_gauss_PSF_sigma_z_nm = 700 / 2.354

        self.render_var_gauss_sigma_min_xy_nm = 10 / 2.354
        self.render_var_gauss_sigma_min_z_nm = 10 / 2.354

        self.render_range_x_percent = np.arange(2) * 100
        self.render_range_y_percent = np.arange(2) * 100
        self.render_range_z_percent = np.arange(2) * 100

        self.pixelsize_nm = []
        self._zdim = None
        self.channel = []

        # GUI
        self.setAcceptDrops(True)
        self.tabs = QTabWidget()
        self.data_control_tab = QWidget()
        self.infos_tab = QWidget()

        self.tabs.addTab(self.data_control_tab, 'Data Controls')
        self.tabs.addTab(self.infos_tab, 'File Infos')

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

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText('Size of Scalebar [nm]:')
        self.data_controls_tab_layout.addWidget(self.Lscalebarsize, 13, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setText('500')
        self.Esbsize.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sbscale)
        )
        self.data_controls_tab_layout.addWidget(self.Esbsize, 13, 1, 1, 3)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self.data_to_layer_itf.scalebar)

        self.Bz_color_coding = QCheckBox()
        self.Bz_color_coding.setText('Activate Rainbow colorcoding in Z')
        self.Bz_color_coding.stateChanged.connect(self.colorcoding)
        self.data_controls_tab_layout.addWidget(self.Bz_color_coding, 12, 2, 1, 2)

        # visual_control_tab
        self.channel_controls_widget_layout = QFormLayout()
        self.channel_controls_placeholder = QWidget()
        self.data_controls_tab_layout.addWidget(self.channel_controls_placeholder, 17, 0, 1, 4)
        self.channel_controls_placeholder.setLayout(self.channel_controls_widget_layout)

        self.infos_tab_layout = QGridLayout()
        self.Lnumberoflocs = TestListView(parent=self)
        self.Lnumberoflocs.addItem(
            'STATISTICS \nWaiting for Data... \nImport or drag file here'
        )

        self.Lnumberoflocs.itemDoubleClicked.connect(self.Lnumberoflocs.remove_dataset)
        self.infos_tab_layout.addWidget(self.Lnumberoflocs, 0, 0)

        self.layout = QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.data_controls_tab_layout.setColumnStretch(0, 4)
        self.data_control_tab.setLayout(self.data_controls_tab_layout)
        self.infos_tab.setLayout(self.infos_tab_layout)

        # Init Function Calls
        custom_keys_and_scalebar(self)
        self.hide_non_available_widgets()

    @property
    def data_to_layer_itf(self):
        return self._data_to_layer_itf

    @data_to_layer_itf.setter
    def data_to_layer_itf(self,value):
        raise StaticAttributeError('the dataset to layer interface should never be changed')

    @property
    def file_to_data_itf(self):
        return self._file_to_itf

    @file_to_data_itf.setter
    def file_to_data_itf(self,value):
        raise StaticAttributeError('the file to dataset interface should never be changed')

    @property
    def render_gaussian_mode(self):
        return self._render_gaussian_mode

    @render_gaussian_mode.setter
    def render_gaussian_mode(self, value):
        if value == 0 or value == 1:
            self._render_gaussian_mode = value
        else:
            raise ValueError('Wrong value for gaussian render mode')

    @property
    def z_color_encoding_mode(self):
        return self._z_color_encoding_mode

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        if value == 0 or value == 1:
            self._z_color_encoding_mode = value
        else:
            raise ValueError('Wrong value for Z Colorcoding')

    @property
    def zdim(self):
        return self._zdim

    @zdim.setter
    def zdim(self, bool):
        """3D or 2D Mode"""
        if not type(bool) == type(True):
            raise ValueError('Zdim present can either be true or false')
        if (self.zdim == True and bool == False) or (self.zdim == False and bool == True):
            raise DimensionError('Error while merging, combination of 2D and 3D datasets not possible')
        else:
            self._zdim = bool
        self.adjust_available_options_to_data_dimension()

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
        self._zdim=None

    def reset_render_range(self, full_reset=False):
        v = napari.current_viewer()
        self.Srender_rangex.reset()
        self.Srender_rangey.reset()
        self.Srender_rangez.reset()
        if not full_reset:
            self.data_to_layer_itf.update_layers(self)

    def update_render_range(self, slider_type, values):
        if slider_type == 'x':
            self.render_range_x_percent = values
        elif slider_type == 'y':
            self.render_range_y_percent = values
        else:
            self.render_range_z_percent = values

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

    def adjust_available_options_to_data_dimension(self):
        if self.zdim:
            self.Lrangez.show()
            self.Srender_rangez.show()
            self.Baxis_xy.show()
            self.Baxis_xz.show()
            self.Baxis_yz.show()
            self.Lresetview.show()
            self.Bz_color_coding.show()
            self.Brenderoptions.show()
            self.Lrenderoptions.show()
            self.Esigma_z.show()
            self.Lsigma_z.show()
            self.viewer.dims.ndisplay = 3
        else:
            self.Lrenderoptions.hide()
            self.Brenderoptions.hide()
            self.Bz_color_coding.show()
            self.Lrangez.hide()
            self.Srender_rangez.hide()
            self.Esigma_z.hide()
            self.Lsigma_z.hide()
            self.viewer.dims.ndisplay = 2

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
            self.Esigma_min_z.show()
            self.Lsigma_xy_min.show()
            self.Lsigma_z_min.show()

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
            v.camera.angles = (0, 0, 90)
        elif type == 'XZ':
            v.camera.angles = (0, 0, 180)
        else:
            v.camera.angles = (-90, -90, -90)
        v.camera.center = self.localization_datasets[-1].camera_center[0]
        v.camera.zoom = self.localization_datasets[-1].camera_center[1]
        v.camera.update(values)

    def update_sigma(self):
        if self.render_gaussian_mode == 0:
            # fixed gaussian
            self.render_fixed_gauss_sigma_xy_nm = float(self.Esigma_xy.text())/2.354
            self.render_fixed_gauss_sigma_z_nm = float(self.Esigma_z.text())/2.354
        else:
            self.render_var_gauss_PSF_sigma_xy_nm = float(self.Esigma_xy.text())/2.354
            self.render_var_gauss_PSF_sigma_z_nm = float(self.Esigma_z.text())/2.354
            self.render_var_gauss_sigma_min_xy_nm = float(self.Esigma_min_xy.text()) / 2.354
            self.render_var_gauss_sigma_min_z_nm = float(self.Esigma_min_z.text()) / 2.354
        self.data_to_layer_itf.update_layers()

    def open_localization_data_file_and_get_dataset(self, merge=False, file_path=None):
        self.show_avaiable_widgets()
        if merge == False:
            self.clear_datasets()

        if self.n_datasets != 0 and not merge:
            self.clear_dataset()
        datasets = self._file_to_data_itf.open_localization_data_file_and_get_dataset(file_path=file_path)
        if datasets[-1].zdim_present:
            self.zdim = True
        else:
            self.zdim = False
        for i in range(len(datasets)):
            self.localization_datasets.append(datasets[i])
            self.n_datasets += 1
            self.create_layer(self.localization_datasets[-1], idx=i)

    def create_layer(self, dataset, idx=-1):
        self.adjust_available_options_to_data_dimension()
        self.Lnumberoflocs.show_infos(filename=dataset.name, idx=idx)
        self.localization_datasets[idx].update_locs()
        self.data_to_layer_itf.create_new_layer(dataset, layer_name=dataset.name, idx=idx)
        self.add_channel(name=dataset.name)
        self.channel[-1].change_color_map()
        self.channel[-1].adjust_colormap_range()


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
