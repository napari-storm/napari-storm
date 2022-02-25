from PyQt5.QtGui import (
    QPaintEvent,
    QPainter,
    QPalette,
    QBrush,
    QMouseEvent,
    QPixmap,
    QIcon,
    QGradient,
    QLinearGradient,
    QColor,
)

from napari_plugin_engine import napari_hook_implementation

from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QListWidget,
    QWidget,
)

from PyQt5.QtWidgets import (
    QGridLayout,
    QStyleOptionSlider,
    QSlider,
    QSizePolicy,
    QStyle,
    QApplication,
    QLineEdit,
    QCheckBox,
    QTabWidget,
    QFormLayout,
)

from PyQt5 import QtCore, QtGui

from PyQt5.QtCore import Qt, QRect, QSize

import easygui

import numpy as np
import yaml as _yaml
import os.path as _ospath
import h5py

from tkinter import filedialog as fd
from tkinter import Tk
import matplotlib.pyplot as plt

import napari

from .ns_constants import *
from .particles import Particles, BillboardsFilter
from .utils import generate_billboards_2d
from .Exp_Controlls import *

class localization_data:
    """An Object which contains the localization data,
    as well as rendering parameters used by the napari
    particles layer"""

    def __init__(
        self,
        parent,
        locs,
        name=None,
        pixelsize_nm=None,
        offset_pixels=None,
        zdim_present=False,
        sigma_present=False,
        photon_count_present=False,
    ):

        assert isinstance(parent, napari_storm) == True

        assert isinstance(locs, np.recarray)
        assert locs.dtype == LOCS_DTYPE

        if name == None :
            name = 'Untitled'

        if pixelsize_nm == None :
            pixelsize_nm = 100.0

        if offset_pixels == None :
            offset_pixels = np.zeros(3)

        self.parent = parent
        self.napari_layer_ref = None

        self.name = name

        self.locs_active = locs
        self.locs_all = locs.copy()
        self.pixelsize_nm = pixelsize_nm
        self.offset_pixels = offset_pixels
        self.zdim_present = zdim_present
        self.sigma_present = sigma_present
        self.photon_count_present = photon_count_present
        self.uncertainty_defined = sigma_present or photon_count_present

        self.render_sigma = None
        self.render_size = None
        self.render_values = None
        self.render_colormap = None
        self.render_anti_alias = 0

        self.set_render_sigmas()
        self.set_render_values()


    def get_coords(self):
        # Returns the coordinates of the active localizations, with
        # the offset applied.

        COORDS_DTYPE = [('x_pos_pixels', 'f4'),
                        ('y_pos_pixels', 'f4'),
                        ('z_pos_pixels', 'f4')]

        tmp_x = self.locs_active.x_pos_pixels + self.offset_pixels[0]
        tmp_y = self.locs_active.y_pos_pixels + self.offset_pixels[1]
        tmp_z = self.locs_active.z_pos_pixels + self.offset_pixels[2]

        tmp_records = np.recarray((tmp_x.size,), dtype = COORDS_DTYPE)
        tmp_records.x_pos_pixels = tmp_x
        tmp_records.y_pos_pixels = tmp_y
        tmp_records.z_pos_pixels = tmp_z

        return tmp_records


    def set_render_values(self):
        """Update values, which are used to determine the rendered
        color and intensity of each localization"""

        if (self.parent.render_gaussian_mode == 0) :
            # Fixed gaussian mode

            tmp_values = np.ones((self.locs_active.size,))

        elif (self.parent.render_gaussian_mode == 1) :
            # Variable gaussian mode

            assert self.uncertainty_defined == True

            if self.zdim_present :
                # 3D data

                if self.sigma_present :
                    # Sigma values present

                    sigma_x_pixels = self.locs_active.sigma_x_pixels
                    sigma_y_pixels = self.locs_active.sigma_y_pixels
                    sigma_z_pixels = self.locs_active.sigma_z_pixels

                    tmp_product = sigma_x_pixels * sigma_y_pixels * sigma_z_pixels

                    tmp_values = 1.0 / tmp_product

                else :
                    #Calculate sigma according to photon count

                    psf_sigma_xy_nm = self.parent.render_var_gauss_PSF_sigma_xy_nm
                    psf_sigma_z_nm = self.parent.render_var_gauss_PSF_sigma_z_nm

                    psf_sigma_xy_pixels = psf_sigma_xy_nm / self.pixelsize_nm
                    psf_sigma_z_pixels = psf_sigma_z_nm / self.pixelsize_nm

                    sigma_xy_pixels = psf_sigma_xy_pixels / sqrt(self.locs_active.photons)
                    sigma_z_pixels = psf_sigma_z_pixels / sqrt(self.locs_active.photons)

                    tmp_product = sigma_xy_pixels**2 * sigma_z_pixels

                    tmp_values = 1.0 / tmp_product

            else:
                # 2D data

                if self.sigma_present :
                    # Sigma values present

                    sigma_x_pixels = self.locs_active.sigma_x_pixels
                    sigma_y_pixels = self.locs_active.sigma_y_pixels

                    tmp_product = sigma_x_pixels * sigma_y_pixels

                    tmp_values = 1.0 / tmp_product

                else :
                    #Calculate sigma according to photon count

                    psf_sigma_xy_pixels = psf_sigma_xy_nm / self.pixelsize_nm

                    sigma_xy_pixels = psf_sigma_xy_pixels / sqrt(self.locs_active.photons)

                    tmp_product = sigma_xy_pixels**2

                    tmp_values = 1.0 / tmp_product


        if self.parent.z_color_encoding == 1 :
            # Color the localizations according to their Z-coordinate
            #
            # In this case, it is not possible to render unit volume gaussians
            # by adjusting their "intensity" via the value parameter.  Rather,
            # the value parameter is used in conjunction with the color map to
            # assign a z-dependent color to each localization.

            assert self.zdim_present == True

            tmp_coords = self.get_coords()

            tmp_values = tmp_coords.z_pos_pixels

            tmp_values = tmp_values - np.min(tmp_values)


        # Normalize values to 1.0
        assert np.max(tmp_values) > 0
        tmp_values = tmp_values / np.max(tmp_values)

        # Store values
        self.render_values = tmp_values


    def set_render_sigmas(self):
        """Update rendered sigma values"""

        if (self.parent.render_gaussian_mode == 0) :
            # Fixed gaussian mode

            sigma_xy_nm = self.parent.render_fixed_gauss_sigma_xy_nm
            sigma_z_nm = self.parent.render_fixed_gauss_sigma_z_nm

            sigma_xy_pixels = sigma_xy_nm / self.pixelsize_nm
            sigma_z_pixels = sigma_z_nm / self.pixelsize_nm

            tmp_sigma_xy = sigma_xy_pixels * np.ones_like(self.locs_active.x_pos_pixels)
            tmp_sigma_z = sigma_z_pixels * np.ones_like(self.locs_active.x_pos_pixels)

            tmp_render_sigma_pixels = np.swapaxes([tmp_sigma_z, tmp_sigma_xy, tmp_sigma_xy], 0, 1)

        elif (self.parent.render_gaussian_mode == 1) :
            # Variable gaussian mode

            if self.sigma_present:
                # Sigma values present

                sigma_x_pixels = self.locs_active.sigma_x_pixels
                sigma_y_pixels = self.locs_active.sigma_y_pixels
                sigma_z_pixels = self.locs_active.sigma_z_pixels

                tmp_render_sigma_pixels = np.swapaxes([sigma_z_pixels, sigma_xy_pixels, sigma_xy_pixels], 0, 1)

            else :
                # Calculate sigma values based on photon counts

                psf_sigma_xy_nm = self.parent.render_var_gauss_PSF_sigma_xy_nm
                psf_sigma_z_nm = self.parent.render_var_gauss_PSF_sigma_z_nm

                psf_sigma_xy_pixels = psf_sigma_xy_nm / self.pixelsize_nm
                psf_sigma_z_pixels = psf_sigma_z_nm / self.pixelsize_nm

                sigma_xy_pixels = psf_sigma_xy_pixels / sqrt(self.locs_active.photons)
                sigma_z_pixels = psf_sigma_z_pixels / sqrt(self.locs_active.photons)

                tmp_render_sigma_pixels = np.swapaxes([sigma_z_pixels, sigma_xy_pixels, sigma_xy_pixels], 0, 1)


        tmp_render_sigma_norm = tmp_render_sigma_pixels / np.max(tmp_render_sigma_pixels)

        # Store sigma values and set render size

        self.render_sigma = tmp_render_sigma_norm
        self.render_size = 5 * np.max(tmp_render_sigma_pixels)


    def update_locs(self):

        self.locs_active = self.locs_all.copy()
        coords = self.get_coords()

        render_xrange = self.parent.render_range_x_pixels
        render_yrange = self.parent.render_range_y_pixels
        render_zrange = self.parent.render_range_z_pixels

        xmin = render_xrange[0]
        xmax = render_xrange[1]
        ymin = render_yrange[0]
        ymax = render_yrange[1]
        zmin = render_zrange[0]
        zmax = render_zrange[1]

        xcoords = coords.x_pos_pixels
        ycoords = coords.y_pos_pixels
        zcoords = coords.z_pos_pixels

        filtered_locs = self.locs_active[np.where(xcoords >= xmin and
                                                  xcoords <= xmax and
                                                  ycoords >= ymin and
                                                  ycoords <= ymax and
                                                  zcoords >= zmin and
                                                  zcoords <= zmax)]

        self.locs_active = filtered_locs

        self.set_render_sigmas()
        self.set_render_values()


class ChannelControls(QWidget):
    """A QT widget that is created for every channel,
    which provides the visual controls"""

    def __init__(
        self,
        parent,
        name,
        channel_index,
    ):
        from .RangeSlider import RangeSlider

        super().__init__()
        self.parent = parent
        self.name = name
        self.channel_index = channel_index

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
        self.Slider_colormap_range.setRange(0.0, 100.0)
        self.Slider_colormap_range.setValue((10.0, 90.0))
        self.Slider_colormap_range.valueChanged.connect(self.adjust_colormap_range)

        self.Slider_opacity = QSlider(Qt.Horizontal)
        self.Slider_opacity.setRange(0.0, 100.0)
        self.Slider_opacity.setValue(100.0)
        self.Slider_opacity.hide()
        self.Slider_opacity.valueChanged.connect(self.adjust_z_color_encoding_opacity)

        self.Colormap_selector = QComboBox()
        items = []
        icons = []
        for cmap in self.parent.colormaps:
            items.append(cmap.name)
            pixmap = QPixmap(20, 20)
            color = QColor(
                cmap.colors[1][0] * 255,
                cmap.colors[1][1] * 255,
                cmap.colors[1][2] * 255,
                cmap.colors[1][3] * 255,
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

    def adjust_colormap_range(self):

        tmp_range = self.Slider_colormap_range.getRange()

        self.colormap_range_low = tmp_range[0]
        self.colormap_range_high = tmp_range[1]

        tmp_dset_obj = self.parent.list_of_datasets[self.channel_index]
        tmp_layer = dset_obj.napari_layer_ref

        tmp_layer.contrast_limits = tmp_range


    def adjust_z_color_encoding_opacity(self):

        tmp_opacity = self.Slider_opacity.value()

        self.opacity_slider_setting = tmp_opacity

        tmp_dset_obj = self.parent.list_of_datasets[self.channel_index]
        tmp_layer = dset_obj.napari_layer_ref

        tmp_layer.opacity = tmp_opacity / 100.0


    def reset(self):

        self.Slider_colormap_range.setValue((10.0, 90.0))
        self.Slider_opacity.setValue(100.0)


    def change_color_map(self):
        # adjust colormap

        tmp_cmap_index = self.Colormap_selector.currentIndex()
        self.colormap_index = tmp_cmap_index

        tmp_dset_obj = self.parent.list_of_datasets[self.channel_index]
        tmp_layer = dset_obj.napari_layer_ref

        if self.parent.z_color_encoding_mode == True :

            assert self.parent.render_gaussian_mode == 0
            tmp_layer.render_colormap = 'hsv'

        else:

            tmp.layer.render_colormap =  self.parent.colormaps[tmp_cmap_index]


    def show_channel(self):

        tmp_dset_obj = self.parent.list_of_datasets[self.channel_index]
        tmp_layer = dset_obj.napari_layer_ref

        if self.show_channel_state == True :
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

            if self.parent.z_color_encoding_mode == True:

                self.Slider_opacity.setValue(self.opacity_slider_setting)

                self.Slider_colormap_range.hide()
                self.Colormap_selector.hide()

                self.Slider_opacity.show()

            else:

                tmp_layer.opacity = 1.0

                self.Slider_colormap_range.show()
                self.Colormap_selector.show()
                self.Slider_opacity.hide()



class napari_storm(QWidget):
    """The Heart of this code: A Dock Widget, but also
    an object where everthing runs together"""

    def __init__(self, napari_viewer):

        from .RangeSlider2 import RangeSlider2

        super().__init__()

        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.data_control_tab = QWidget()
        self.visual_control_tab = QWidget()

        # self.setStyleSheet('background-color: #414851')

        self.tabs.addTab(self.data_control_tab, 'Data Controls')
        self.tabs.addTab(self.visual_control_tab, 'Visual Controls')

        gaussian_render_modes = ['Fixed-size gaussian',
                                 'Variable-size gaussian']

        self.localization_datasets = []
        self.n_datasets = 0

        self.render_gaussian_mode = 0
        self.z_color_encoding_mode = False
        self.render_fixed_gauss_sigma_xy_nm = 0.0
        self.render_fixed_gauss_sigma_z_nm = 0.0
        self.render_var_gauss_PSF_sigma_xy_nm = 0.0
        self.render_var_gauss_PSF_sigma_z_nm = 0.0

        self.render_range_x_pixels = np.zeros(2)
        self.render_range_y_pixels = np.zeros(2)
        self.render_range_z_pixels = np.zeros(2)

        self.pixelsize = []
        self.layer = []
        self.layer_names = []
        self.scalebar_layer = []
        self.scalebar_exists = False
        self.zdim = False
        self.channel = []
        self.colormaps = colormaps()

        self.viewer = napari_viewer
        layout = QGridLayout()

        # Set up the GUI

        self.Limport = QLabel()
        self.Limport.setText('Import \nSTORM Data')
        layout.addWidget(self.Limport, 0, 0)

        self.Bopen = QPushButton()
        self.Bopen.clicked.connect(self.open_localization_data)
        self.Bopen.setText('Import File Dialog')
        layout.addWidget(self.Bopen, 0, 1, 1, 4)

        # self.Lnumberoflocs = TestListView(self, parent=self)
        # self.Lnumberoflocs.addItem(
        #     'STATISTICS \nWaiting for Data... \nImport or drag file here'
        # )
        # self.Lnumberoflocs.itemDoubleClicked.connect(self.Lnumberoflocs.remove_dataset)
        # layout.addWidget(self.Lnumberoflocs, 1, 0, 1, 4)

        self.Lresetview = QLabel()
        self.Lresetview.setText('Reset view:')
        layout.addWidget(self.Lresetview, 2, 0)

        self.Baxis_xy = QPushButton()
        self.Baxis_xy.setText('XY')
        self.Baxis_xy.clicked.connect(lambda: self.change_camera(type='XY'))
        layout.addWidget(self.Baxis_xy, 2, 1)

        self.Baxis_yz = QPushButton()
        self.Baxis_yz.setText('YZ')
        self.Baxis_yz.clicked.connect(lambda: self.change_camera(type='YZ'))
        layout.addWidget(self.Baxis_yz, 2, 2)

        self.Baxis_xz = QPushButton()
        self.Baxis_xz.setText('XZ')
        self.Baxis_xz.clicked.connect(lambda: self.change_camera(type='XZ'))
        layout.addWidget(self.Baxis_xz, 2, 3)

        self.Lrenderoptions = QLabel()
        self.Lrenderoptions.setText('Rendering options:')
        layout.addWidget(self.Lrenderoptions, 3, 0)

        self.Brenderoptions = QComboBox()
        self.Brenderoptions.addItems(gaussian_render_modes)
        self.Brenderoptions.currentIndexChanged.connect(self.render_options_changed)
        layout.addWidget(self.Brenderoptions, 3, 1, 1, 3)

        self.Lsigma = QLabel()
        self.Lsigma.setText('FWHM in XY [nm]:')
        layout.addWidget(self.Lsigma, 4, 0)

        self.Lsigma2 = QLabel()
        self.Lsigma2.setText('FWHM in Z [nm]:')
        layout.addWidget(self.Lsigma2, 5, 0)

        self.Esigma_xy = QLineEdit()
        self.Esigma_xy.setText('10')
        self.Esigma_xy.textChanged.connect(
            lambda: self.start_typing_timer(self.typing_timer_sigma)
        )
        layout.addWidget(self.Esigma_xy, 4, 1, 1, 3)
        self.typing_timer_sigma = QtCore.QTimer()
        self.typing_timer_sigma.setSingleShot(True)
        self.typing_timer_sigma.timeout.connect(lambda: update_layers(self))

        self.Esigma_z = QLineEdit()
        self.Esigma_z.setText('10')
        self.Esigma_z.textChanged.connect(
            lambda: self.start_typing_timer(self.typing_timer_sigma)
        )
        layout.addWidget(self.Esigma_z, 5, 1, 1, 3)

        self.Lrangex = QLabel()
        self.Lrangex.setText('X-range')
        layout.addWidget(self.Lrangex, 6, 0)

        self.Lrangey = QLabel()
        self.Lrangey.setText('Y-range')
        layout.addWidget(self.Lrangey, 7, 0)

        self.Lrangez = QLabel()
        self.Lrangez.setText('Z-range')
        layout.addWidget(self.Lrangez, 8, 0)

        self.Srender_rangex = RangeSlider2(parent=self, type='x')
        layout.addWidget(self.Srender_rangex, 6, 1, 1, 3)

        self.Srender_rangey = RangeSlider2(parent=self, type='y')
        layout.addWidget(self.Srender_rangey, 7, 1, 1, 3)

        self.Srender_rangez = RangeSlider2(parent=self, type='z')
        layout.addWidget(self.Srender_rangez, 8, 1, 1, 3)

        self.Lscalebar = QLabel()
        self.Lscalebar.setText('Scalebar?')
        layout.addWidget(self.Lscalebar, 9, 0)

        self.Cscalebar = QCheckBox()
        self.Cscalebar.stateChanged.connect(self.scalebar)
        layout.addWidget(self.Cscalebar, 9, 1, 1, 3)

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText('Size of Scalebar [nm]:')
        layout.addWidget(self.Lscalebarsize, 10, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setText('500')
        self.Esbsize.textChanged.connect(
            lambda: self.start_typing_timer(self.typing_timer_sbscale)
        )
        layout.addWidget(self.Esbsize, 10, 1, 1, 3)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self.scalebar)

        self.Bz_color_coding = QCheckBox()
        self.Bz_color_coding.setText('Activate Rainbow colorcoding in Z')
        self.Bz_color_coding.stateChanged.connect(self.colorcoding)
        layout.addWidget(self.Bz_color_coding, 13, 0, 1, 4)

        self.Bmerge_with_additional_file = QPushButton()
        self.Bmerge_with_additional_file.setText('Merge with additional file')
        layout.addWidget(self.Bmerge_with_additional_file, 14, 0, 1, 2)
        self.Bmerge_with_additional_file.clicked.connect(self.open_localization_data(merge=True))


        self.Breset_render_range = QPushButton()
        self.Breset_render_range.setText('Reset Render Range')
        self.Breset_render_range.clicked.connect(self.reset_render_range)
        layout.addWidget(self.Breset_render_range, 14, 2, 1, 2)

        ########################################## visual_control_tab
        self.layout2 = QFormLayout()

        ##############################################
        self.layout = QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.custom_controls = MouseControls()
        layout.setColumnStretch(0, 4)
        self.data_control_tab.setLayout(layout)
        self.visual_control_tab.setLayout(self.layout2)
        # self.right_click_pan()
        custom_keys_and_scalebar(self)
        self.hide_stuff()

    def open_localization_data(self, file_path=None, merge=False):
        """Determine which file type is being opened, and call the
        corresponding importer function"""

        self.update_widget_states()

        from .importer import (
            load_hdf5,
            load_csv,
            load_SMLM,
            load_h5,
            load_mfx_json,
            load_mfx_npy,
            start_testing,
        )

        if self.n_datasets != 0 and merge == False:
            self.clear_datasets()

        Tk().withdraw()

        if not file_path:
            file_path = fd.askopenfilename()

        filetype = file_path.split('.')[-1]
        filename = file_path.split('/')[-1]

        if filetype == 'hdf5':
            load_hdf5(self, file_path)
        elif filetype == 'yaml':
            file_path = file_path[: -(len(filetype))] + 'hdf5'
            load_hdf5(self, file_path)
        elif filetype == 'csv':
            load_csv(self, file_path)
        elif filetype == 'smlm':
            load_SMLM(self, file_path)
        elif filetype == 'h5':
            load_h5(self, file_path)
        elif filetype == 'json':
            load_mfx_json(self, file_path)
        elif filetype == 'npy':
            load_mfx_npy(self, file_path)
        elif filetype == 'test':
            start_testing(self)
        else:
            raise TypeError('Unknown SMLM data file extension')

    def load_h5(self, file_path):
        """Loads localisations from .h5 files"""
        filename = file_path.split('/')[-1]

        with h5py.File(file_path, "r") as locs_file:
            data = locs_file['molecule_set_data']['datatable'][...]
            pixelsize = locs_file['molecule_set_data']['xy_pixel_size_um'][
                            ...] * 1E3  # to µm to nm
        try:
            locs = np.rec.array((data['FRAME_NUMBER'], data['X_POS_PIXELS'],
                                 data['Y_POS_PIXELS'], data['Z_POS_PIXELS'],
                                 data['PHOTONS']), dtype=LOCS_DTYPE_3D)
            zdim = True
        except:
            locs = np.rec.array((data['FRAME_NUMBER'], data['X_POS_PIXELS'],
                                 data['Y_POS_PIXELS'],
                                 data['PHOTONS']), dtype=LOCS_DTYPE_2D)
            zdim = False
        num_channel = max(data['CHANNEL']) + 1
        offset = look_for_offset(locs, zdim)
        for i in range(num_channel):
            filename_pluschannel = check_namespace(self,
                                                   filename + f" Channel {i + 1}")
            locs_in_ch = locs[data['CHANNEL'] == i]
            self.localization_datasets.append(
                localization_data(locs=locs_in_ch, zdim_present=zdim,
                                  parent=self, pixelsize_nm=pixelsize,
                                  name=filename_pluschannel, offset=offset,
                                  channel_index=len(
                                      self.localization_datasets)))
            create_new_layer(self=self, aas=0.1,
                             layer_name=filename_pluschannel,
                             idx=len(self.localization_datasets) - 1)

    def clear_datasets(self):
        """Erase the current dataset and reset the viewer"""

        v = self.viewer

        for i in range(self.n_datasets):
            v.layers.remove(self.localization_datasets[i].name)

        self.localization_datasets = []

        # self.Lnumberoflocs.clear()

        if not len(self.channel) == 0:  # Remove Channel of older files
            for i in range(len(self.channel)):
                self.channel[i].hide()
            self.channel = []

        self.Cscalebar.setCheckState(False)

        self.reset_render_range(full_reset=True)

        self.Bz_color_coding.setCheckState(False)

    def reset_render_range(self, full_reset=False):
        v = napari.current_viewer()
        self.Srender_rangex.reset()
        self.Srender_rangey.reset()
        self.Srender_rangez.reset()
        if not full_reset:
            update_layers(self)

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
            self.open_localization_data(file_path=file)
        else:
            event.ignore()
        #####

    def hide_stuff(self):
        """Hide controls which are better untouched atm"""
        self.Srender_rangex.hide()
        self.Srender_rangey.hide()
        self.Lrangex.hide()
        self.Lrangey.hide()
        self.Cscalebar.hide()
        self.Lscalebar.hide()
        self.Brenderoptions.hide()
        self.Lrenderoptions.hide()
        self.Lsigma.hide()
        self.Esigma_xy.hide()
        self.Lsigma2.hide()
        self.Esigma_z.hide()
        self.Bz_color_coding.hide()
        self.Lscalebarsize.hide()
        self.Esbsize.hide()
        self.Bmerge_with_additional_file.hide()
        self.Srender_rangez.hide()
        self.Lrangez.hide()
        self.Lresetview.hide()
        # self.Baxis.hide()
        self.Baxis_xy.hide()
        self.Baxis_yz.hide()
        self.Baxis_xz.hide()

    def update_widget_states(self):
        """Show the Controls usable atm"""
        self.Srender_rangex.show()
        self.Srender_rangey.show()
        self.Lrangex.show()
        self.Lrangey.show()
        self.Cscalebar.show()
        self.Lscalebar.show()
        self.Brenderoptions.show()
        self.Lrenderoptions.show()
        self.Lsigma.show()
        self.Esigma_xy.show()
        self.Lsigma2.show()
        self.Esigma_z.show()
        self.Lscalebar.show()
        self.Lscalebarsize.show()
        self.Esbsize.show()
        self.Bmerge_with_additional_file.show()

    def add_channel(self, name='Channel'):
        """Adds a Channel in the visual controlls"""
        self.channel.append(
            ChannelControls(parent=self, name=name, channel_index=len(self.channel))
        )
        self.layout2.addRow(self.channel[-1])

    def colorcoding(self):
        """Check if Colorcoding is choosen"""
        print(self.Bz_color_coding.isChecked())
        if self.Bz_color_coding.isChecked():
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.hide()
                self.channel[i].Slider_opacity.show()
                self.channel[i].Slider_colormap_range.hide()
                self.channel[i].Label.setText('Contrast ' + self.channel[i].name)
        else:
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.show()
                self.channel[i].Slider_opacity.hide()
                self.channel[i].Slider_colormap_range.show()
                self.channel[i].Label.setText('Opacity ' + self.channel[i].name)
                self.channel[i].clear_datasets()
        update_layers(self)

    def render_options_changed(self):
        if self.Brenderoptions.currentText() == 'variable gaussian':
            self.Lsigma.setText('PSF FWHM in XY [nm]')
            self.Lsigma2.setText('PSF FWHM in Z [nm]')
            self.Esigma_xy.setText('300')
            self.Esigma_z.setText('700')
            self.Bz_color_coding.hide()
            self.Bz_color_coding.setCheckState(False)

        else:
            self.Lsigma2.setText('FWHM in Z [nm]')
            self.Lsigma.setText('FWHM in XY [nm]')
            self.Esigma_xy.setText('10')
            self.Esigma_z.setText('10')
            self.Bz_color_coding.show()
        update_layers(self)

    def scalebar(self):
        """Creating/Removing/Updating the custom Scalebar in 2 and 3D"""
        v = napari.current_viewer()
        cpos = v.camera.center
        l = int(self.Esbsize.text())
        if self.Cscalebar.isChecked() and not not all(self.localization_datasets[-1].locs_active):
            if self.localization_datasets[-1].zdim_present:
                list = [l, 0.125 * l / 2, 0.125 * l / 2]
                faces = np.asarray(
                    [
                        [0, 1, 2],
                        [1, 2, 3],
                        [4, 5, 6],
                        [5, 6, 7],
                        [0, 2, 4],
                        [4, 6, 2],
                        [1, 3, 7],
                        [1, 5, 7],
                        [2, 3, 6],
                        [3, 6, 7],
                        [4, 5, 0],
                        [0, 1, 5],
                    ]
                )
                vertices = []
                """
                for c in range(3):
                    i=list[c%3]
                    j=list[(c+1)%3]
                    k=list[(c+2)%3]
                    m=0
                    verts = [[-i,-k,-j],[-i,k,-j],[-i,-k,j],[-i,k,j],[i,-k,-j],[i,k,-j],[i,-k,j],[i,k,j]]
                    vertices.append(np.asarray(verts + cpos*np.ones_like(verts)))"""
                vertices = np.asarray(
                    [
                        [0, list[1], list[2]],
                        [0, -list[1], list[2]],
                        [0, list[1], -list[2]],
                        [0, -list[1], -list[2]],
                        [l, list[1], list[2]],
                        [l, -list[1], list[2]],
                        [l, list[1], -list[2]],
                        [l, -list[1], -list[2]],
                        [list[1], 0, list[2]],
                        [-list[1], 0, list[2]],
                        [list[1], 0, -list[2]],
                        [-list[1], 0, -list[2]],
                        [list[1], l, list[2]],
                        [-list[1], l, list[2]],
                        [list[1], l, -list[2]],
                        [-list[1], l, -list[2]],
                        [list[1], list[2], 0],
                        [-list[1], list[2], 0],
                        [list[1], -list[2], 0],
                        [-list[1], -list[2], 0],
                        [list[1], list[2], l],
                        [-list[1], list[2], l],
                        [list[1], -list[2], l],
                        [-list[1], -list[2], l],
                    ]
                )
                for i in range(len(vertices)):
                    vertices[i] = vertices[i] + cpos - (l / 2, 0, 0)

                faces = np.asarray(np.vstack((faces, faces + 8, faces + 16)))
                # vertices=np.reshape(np.asarray(vertices),(24,3))
            else:
                list = [l, 0.05 * l]

                faces = np.asarray([[0, 1, 3], [1, 2, 3], [4, 5, 7], [5, 6, 7]])
                verts = [
                    [cpos[1], cpos[2] - list[1]],
                    [cpos[1] + list[0], cpos[2] - list[1]],
                    [cpos[1] + list[0], cpos[2] + list[1]],
                    [cpos[1], cpos[2] + list[1]],
                    [cpos[1] - list[1], cpos[2]],
                    [cpos[1] + list[1], cpos[2]],
                    [cpos[1] + list[1], cpos[2] + list[0]],
                    [cpos[1] - list[1], cpos[2] + list[0]],
                ]
                vertices = np.reshape(np.asarray(verts), (8, 2))
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_layer = v.add_surface(
                    (vertices, faces), name='scalebar', shading='none'
                )
            else:
                self.scalebar_layer = v.add_surface(
                    (vertices, faces), name='scalebar', shading='none'
                )
                self.scalebar_exists = True
        else:
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_exists = False

    def threed(self):
        """3D or 2D Mode"""
        v = napari.current_viewer()
        # print(v.camera.view_direction)
        if self.localization_datasets[-1].zdim_present:
            self.Lrangez.show()
            self.Srender_rangez.show()
            # self.Baxis.show()
            self.Baxis_xy.show()
            self.Baxis_xz.show()
            self.Baxis_yz.show()
            self.Lresetview.show()
            self.Bz_color_coding.show()
            v.dims.ndisplay = 3
        else:
            self.Bz_color_coding.show()
            self.Lrangez.hide()
            self.Srender_rangez.hide()
            v.dims.ndisplay = 2

    def start_typing_timer(self, timer):
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


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return napari_storm





##### Highest Order Function
def colormaps():
    """Creating the Custom Colormaps"""
    cmaps = []
    names = ['red', 'green', 'blue']
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
    names = ['yellow', 'cyan', 'pink']
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i + 1) % 3] = 1
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
    names = ['orange', 'mint', 'purple']
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i + 1) % 3] = 0.5
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
    colors = np.ones((2, 4))
    cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name='gray'))
    return cmaps








def create_new_layer(self, aas=0, layer_name='SMLM Data', idx=-1):
    """Creating a Particle Layer"""
    if not self.localization_datasets[idx].zdim_present:  # Hide 3D Options if 2D Dataset
        self.Srender_rangez.hide()
        self.Lrangez.hide()
        # self.Baxis.hide()
        self.Baxis_xy.hide()
        self.Baxis_yz.hide()
        self.Baxis_xz.hide()
        self.Lresetview.hide()
        self.Bz_color_coding.hide()
    else:
        self.Srender_rangez.show()
        self.Lrangez.show()
        # self.Baxis.show()
        self.Baxis_xy.show()
        self.Baxis_yz.show()
        self.Baxis_xz.show()
        self.Lresetview.show()
        self.Bz_color_coding.show()
    self.localization_datasets[idx].update_locs()
    coords = get_coords_from_locs(self, self.localization_datasets[idx].pixelsize_nm, idx)
    v = napari.current_viewer()  # Just to get the sigmas
    """print(f'Create:\n #################\n'
          f'size = {self.list_of_datasets[idx].size}\n values = {self.list_of_datasets[idx].values}\n '
          f'sigmas {self.list_of_datasets[idx].sigma}\n Pixelsize {self.list_of_datasets[idx].pixelsize}'
          f'\n coords ={coords}\n##############')"""
    self.localization_datasets[idx].napari_layer_ref = Particles(
        coords,
        size=self.localization_datasets[idx].render_size,
        values=self.localization_datasets[idx].render_values,
        antialias=self.localization_datasets[idx].render_anti_alias,
        colormap=self.colormaps[idx],
        sigmas=self.localization_datasets[idx].render_sigma,
        filter=None,
        name=layer_name,
    )
    self.localization_datasets[idx].name = layer_name
    self.localization_datasets[idx].napari_layer_ref.add_to_viewer(v)
    # v.window.qt_viewer.layer_to_visual[self.layer[-1]].node.canvas.measure_fps()
    if self.localization_datasets[idx].zdim_present:
        v.dims.ndisplay = 3
    else:
        v.dims.ndisplay = 2
    v.camera.perspective = 50
    self.localization_datasets[idx].napari_layer_ref.shading = 'gaussian'
    show_infos(self, layer_name, idx)
    self.localization_datasets[idx].camera_center = [
        v.camera.center,
        v.camera.zoom,
        v.camera.angles,
    ]
    self.add_channel(name=layer_name)
    self.channel[-1].change_color_map()
    self.channel[-1].adjust_colormap_range()
    # print(len(self.list_of_datasets[-1].index),'idx,locs',len(self.list_of_datasets[-1].locs.x))


def update_layers(self, aas=0, layer_name='SMLM Data'):
    """Updating a Particle Layer"""
    v = napari.current_viewer()
    self.localization_datasets[-1].camera = [v.camera.zoom, v.camera.center, v.camera.angles]
    for i in range(len(self.localization_datasets)):
        self.localization_datasets[i].update_locs()
        v.layers.remove(self.localization_datasets[i].name)
        coords = get_coords_from_locs(self, self.localization_datasets[i].pixelsize_nm, i)
        """print(f'Update:\n #################\n'
          f'size = {self.list_of_datasets[i].size}\n values = {self.list_of_datasets[i].values}\n '
          f'sigmas ={self.list_of_datasets[i].sigma}\n Pixelsize ={self.list_of_datasets[i].pixelsize}'
          f'\n coords ={coords}\n##############')"""
        self.localization_datasets[i].napari_layer_ref = Particles(
            coords,
            size=self.localization_datasets[i].render_size,
            values=self.localization_datasets[i].render_values,
            antialias=aas,
            sigmas=self.localization_datasets[i].render_sigma,
            filter=None,
            name=self.localization_datasets[i].name,
            opacity=self.channel[i].Slider_opacity.value(),
        )
        self.localization_datasets[i].napari_layer_ref.add_to_viewer(v)
        self.channel[i].adjust_colormap_range()
        self.channel[i].adjust_z_color_encoding_opacity()
        self.channel[i].change_color_map()
        # if np.min(self.list_of_datasets[i].values) != np.max(self.list_of_datasets[i].values):
        #    self.list_of_datasets[i].layer.contrast_limits = (np.min(self.list_of_datasets[i].values),
        #                                                        np.max(self.list_of_datasets[i].values))
        self.localization_datasets[i].napari_layer_ref.shading = 'gaussian'
    v.camera.angles = self.localization_datasets[-1].camera[2]
    v.camera.zoom = self.localization_datasets[-1].camera[0]
    v.camera.center = self.localization_datasets[-1].camera[1]
    v.camera.update({})


def update_layers2(self):
    """Still doesn't work"""
    v = napari.current_viewer()
    for i in range(len(self.localization_datasets)):
        self.localization_datasets[i].update_locs()
        coords = get_coords_from_locs(self, self.localization_datasets[i].pixelsize_nm, i)
        values = self.localization_datasets[i].render_values
        size = self.localization_datasets[i].render_size

        coords = np.asarray(coords)
        if np.isscalar(values):
            values = values * np.ones(len(coords))
        values = np.broadcast_to(values, len(coords))
        size = np.broadcast_to(size, len(coords))
        if coords.shape[1] == 2:
            coords = np.concatenate([np.zeros((len(coords), 1)), coords], axis=-1)

        vertices, faces, texcoords = generate_billboards_2d(coords, size=size)

        values = np.repeat(values, 4, axis=0)
        vertices_old, faces_old, values_old = v.layers[
            self.localization_datasets[i].name
        ].data
        print(f'before: {len(vertices_old),len(faces_old),len(values_old)}')
        print(f'after: {len(vertices), len(faces), len(values)}')
        v.layers[self.localization_datasets[i].name].data = (vertices, faces, values)


def get_coords_from_locs(self, pixelsize, idx):
    """Calculating Particle Coordinates from Locs"""
    if self.localization_datasets[idx].zdim_present:
        num_of_locs = len(self.localization_datasets[idx].locs_active.x)
        coords = np.zeros([num_of_locs, 3])
        coords[:, 0] = self.localization_datasets[idx].locs_active.z * pixelsize
        coords[:, 1] = self.localization_datasets[idx].locs_active.x * pixelsize
        coords[:, 2] = self.localization_datasets[idx].locs_active.y * pixelsize
    else:
        num_of_locs = len(self.localization_datasets[idx].locs_active.x)
        coords = np.zeros([num_of_locs, 2])
        coords[:, 0] = self.localization_datasets[idx].locs_active.x * pixelsize
        coords[:, 1] = self.localization_datasets[idx].locs_active.y * pixelsize
    return coords


##### Semi Order Functions
def show_infos(self, filename, idx):
    """Print Infos about files in Log"""
    if self.localization_datasets[idx].zdim_present:
        self.Lnumberoflocs.addItem(
            'Statistics\n'
            + f'File: {filename}\n'
            + f'Number of locs: {len(self.localization_datasets[idx].locs_active.x)}\n'
            f'Imagewidth: {np.round((max(self.localization_datasets[idx].locs_active.x) - min(self.localization_datasets[idx].locs_active.x)) * self.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n'
            + f'Imageheigth: {np.round((max(self.localization_datasets[idx].locs_active.y) - min(self.localization_datasets[idx].locs_active.y)) * self.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n'
            + f'Imagedepth: {np.round((max(self.localization_datasets[idx].locs_active.z) - min(self.localization_datasets[idx].locs_active.z)) * self.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n'
            + f'Intensity per localisation\nmean: {np.round(np.mean(self.localization_datasets[idx].locs_active.photons), 3)}\nmax: '
            + f'{np.round(max(self.localization_datasets[idx].locs_active.photons), 3)}\nmin:'
            + f' {np.round(min(self.localization_datasets[idx].locs_active.photons), 3)}\n'
        )
    else:
        self.Lnumberoflocs.addItem(
            'Statistics\n'
            + f'File: {filename}\n'
            + f'Number of locs: {len(self.localization_datasets[idx].locs_active.x)}\n'
            f'Imagewidth: {np.round((max(self.localization_datasets[idx].locs_active.x) - min(self.localization_datasets[idx].locs_active.x)) * self.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n'
            + f'Imageheigth: {np.round((max(self.localization_datasets[idx].locs_active.y) - min(self.localization_datasets[idx].locs_active.y)) * self.localization_datasets[idx].pixelsize_nm / 1000, 3)} µm\n'
            + f'Intensity per localisation\nmean: {np.round(np.mean(self.localization_datasets[idx].locs_active.photons), 3)}\nmax: '
            + f'{np.round(max(self.localization_datasets[idx].locs_active.photons), 3)}\nmin:'
            + f' {np.round(min(self.localization_datasets[idx].locs_active.photons), 3)}\n'
        )


class TestListView(QListWidget):
    """Custom ListView Widget -> The Log, allows, d&d and displays infos on the files"""

    def __init__(self, type, parent=None):
        super(TestListView, self).__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

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
            parent.open_localization_data(file_path=file)
        else:
            event.ignore()

    def remove_dataset(self, item):
        print('Dataset removal not implemented yet...', item)
