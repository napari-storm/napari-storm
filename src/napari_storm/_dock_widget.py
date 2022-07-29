from napari_plugin_engine import napari_hook_implementation

from .Exp_Controls import *
from .DataToLayerInterface import DataToLayerInterface
from .FileToLocalizationDataInterface import FileToLocalizationDataInterface
from .ChannelControls import ChannelControls
from .pyqt.CustomSliders import *
from .ns_constants import *
from .GUI import *


class napari_storm(NapariStormGUI):
    """The Heart of this code: A Dock Widget, but also
    an object where everthing runs together"""

    def __init__(self, napari_viewer):

        napari_viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        napari_viewer.window.qt_viewer.dockLayerList.setVisible(False)

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
        self.grid_plane_standard_color_index = 0
        self.possible_grid_plane_orientations = ["XY", "YZ", "XZ"]
        self.active_grid_plane_orientation = self.possible_grid_plane_orientations[0]
        self.grid_plane_opacity = 0.7

        self.active_render_range_box_color = standard_colormaps[0]
        self.render_range_box_opacity = 0.25

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
        self.zoom = 0

        # GUI
        super().__init__()

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
            if self._z_color_encoding_mode == 1 and value == 1:
                self.z_color_encoding_mode = 0
                self.Bz_color_coding.hide()
            else:
                self.Bz_color_coding.show()
            self._render_gaussian_mode = value
        else:
            raise ValueError(f"Wrong value for gaussian render mode, either 0 or 1 not {value} of type {type(value)}")

    @property
    def z_color_encoding_mode(self):
        return self._z_color_encoding_mode

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        if value == 0 or value == 1:
            if self._render_gaussian_mode == 0:
                self._z_color_encoding_mode = value
            else:
                self._z_color_encoding_mode = 0
                raise ValueError(f"Color Encoding only works for fixed gaussian mode")
        else:
            raise ValueError(f"Wrong value for Z colorcoding, either 0 or 1 not {value} of type {type(value)}")

    @property
    def zdim(self):
        return self._zdim

    @zdim.setter
    def zdim(self, boolean):
        """3D or 2D Mode"""
        if not type(boolean) == type(True):
            raise ValueError(f"Zdim present can either be true or false not {boolean} of type {type(boolean)}")
        if (self.zdim == True and not boolean) or (self.zdim == False and boolean):
            raise DimensionError('Error while merging, combination of 2D and 3D datasets not possible')
        else:
            self._zdim = boolean
        self.adjust_available_options_to_data_dimension()

    def update_render_range_box_opacity(self):
        self.render_range_box_opacity = self.Srender_range_box_opacity.value() / 100

    def update_render_range_box_color(self):
        idx = self.Brender_range_box_color.currentIndex()
        self.active_render_range_box_color = standard_colormaps[idx]

    def update_grid_plane_color(self):
        idx = self.Bgrid_plane_color.currentIndex()
        self.grid_plane_standard_color_index = idx
        self.data_to_layer_itf.update_grid_plane(color=standard_colors[idx])

    def update_grid_plane_line_distance(self):
        value_str = self.Egrid_line_distance.text()
        value = float(value_str)
        if value == 0:
            raise ValueError('Line Distance must be > 0')
        self.grid_plane_line_distance_um = value
        self.data_to_layer_itf.update_grid_plane(line_distance_nm=1000 * value)

    def grid_plane(self):
        if self.Cgrid_plane.isChecked():
            self.Egrid_line_distance.show()
            self.Sgrid_line_thickness.show()
            self.Sgrid_z_pos.show()
            self.Bgrid_plane_color.show()
            self.grid_plane_enabled = 1

        else:
            self.Egrid_line_distance.hide()
            self.Sgrid_line_thickness.hide()
            self.Sgrid_z_pos.hide()
            self.grid_plane_enabled = 0
            self.grid_plane_standard_color_index = 0
            self.Bgrid_plane_color.setCurrentIndex(0)
            self.Bgrid_plane_color.hide()

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
            self.data_to_layer_itf.update_grid_plane(line_distance_nm=self.grid_plane_line_distance_um * 1000)
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
            self.data_to_layer_itf.update_grid_plane(line_distance_nm=self.grid_plane_line_distance_um * 1000)
        self.move_camera_center_to_render_range_center()

    def move_camera_center_to_render_range_center(self):
        if not (self.data_to_layer_itf.render_range_x[1] == -np.inf
                or self.data_to_layer_itf.render_range_y[1] == -np.inf
                or self.data_to_layer_itf.render_range_z[1] == -np.inf):
            tmp_x_center_nm = self.data_to_layer_itf.render_range_y[1] * (
                    self.render_range_slider_x_percent[0] / 100 + 0.5 * self.render_range_slider_x_percent[1] / 100
                    - 0.5 * self.render_range_slider_x_percent[0] / 100)
            tmp_y_center_nm = self.data_to_layer_itf.render_range_x[1] * (
                    self.render_range_slider_y_percent[0] / 100 + 0.5 * self.render_range_slider_y_percent[1] / 100
                    - 0.5 * self.render_range_slider_y_percent[0] / 100)
            """tmp_z_center_nm = self.data_to_layer_itf.render_range_z[1] * (
                    self.render_range_slider_z_percent[0] / 100 + 0.5 * self.render_range_slider_z_percent[1] / 100
                    - 0.5 * self.render_range_slider_z_percent[0] / 100)"""
            tmp_z_center_nm = self.viewer.camera.center[0]
            self.data_to_layer_itf.camera[1] = (tmp_z_center_nm, tmp_x_center_nm, tmp_y_center_nm)
            self.viewer.camera.center = (tmp_z_center_nm, tmp_x_center_nm, tmp_y_center_nm)

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

    def change_camera(self, set_view_to='XY'):
        v = napari.current_viewer()
        values = {}
        if set_view_to == 'XY':
            v.camera.angles = (90, 0, -90)
        elif set_view_to == 'XZ':
            v.camera.angles = (-180, 90, 180)
        else:
            v.camera.angles = (-180, 0, -180)
        v.camera.center = self.data_to_layer_itf.camera[1]
        v.camera.zoom = self.zoom
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
            self.Cgrid_plane.setCheckState(False)
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
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(line_distance_nm=self.grid_plane_line_distance_um * 1000)

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
        self.data_to_layer_itf.create_new_layer(dataset, layer_name=dataset.name, idx=idx, merge=merge)
        self.zoom = self.viewer.camera.zoom
        self.add_channel(name=dataset.name)
        self.channel[-1].change_color_map()
        self.channel[-1].adjust_colormap_range()
        if dataset.sigma_present:
            self.render_gaussian_mode = 1
            self.Brenderoptions.setCurrentIndex(1)
            self.Bz_color_coding.hide()


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return napari_storm
