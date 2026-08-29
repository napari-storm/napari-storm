import logging
import weakref

import napari
import numpy as np
from qtpy.QtCore import Qt

from .background_loading import load_in_background
from .ChannelControls import ChannelControls
from .core import DatasetStore, LayerAppearance, Scene, WorldTransform
from .CustomErrors import DimensionError, StaticAttributeError
from .DataAdjustment import DataAdjustmentInterface
from .DataFilter import DataFilterInterface
from .DataToLayerInterface import DataToLayerInterface
from .Exp_Controls import custom_keys_and_scalebar
from .FileToLocalizationDataInterface import FileToLocalizationDataInterface
from .GUI import NapariStormGUI
from .localization_dataset_types import LocalizationDataBaseClass, StormDataClass
from .napari_particles._napari_compat import guard_camera_drag_state
from .ns_constants import (FWHM_TO_SIGMA, MAX_FWHM_NM, MIN_FWHM_NM,
                           standard_colors, standard_colormaps)
from .render_config import RenderConfig


def _non_negative_number(text):
    """Parse *text* as a float of zero or more, or return None.

    Same contract as :func:`_positive_number`, for the controls where zero is
    a real setting rather than a rejected one.
    """
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def _positive_number(text):
    """Parse *text* as a positive float, or return None.

    Returns None rather than raising because these are live `textChanged`
    handlers: half-typed input is normal, and an exception from a keystroke
    would reach the Qt event loop rather than the user.
    """
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0:
        return None
    return value


class napari_storm(NapariStormGUI):
    """The Heart of this code: A Dock Widget, but also
    an object where everything runs together"""

    # Reader-hook registry keyed per viewer.  Weak values avoid keeping closed
    # dock widgets alive, and unlike the former gc.get_objects() heap scan this
    # is deterministic and supports more than one napari viewer.
    _instances = weakref.WeakValueDictionary()

    @classmethod
    def get_instance(cls, viewer=None):
        """Return the live dock registered for *viewer*, if one exists."""
        if viewer is not None:
            return cls._instances.get(id(viewer))
        instances = list(cls._instances.values())
        return instances[0] if len(instances) == 1 else None

    def __init__(self, napari_viewer, renderer=None):

        # napari's own layer list and layer controls are deliberately left
        # alone.  Opening this dock used to hide them both, which §7.4 rules
        # out: they are the host's UI, they work on our layers, and a user who
        # wants them back had no way to ask.  The plugin adds a panel; it does
        # not take the viewer over.

        # Render configuration — single source of truth for render settings
        self.render_config = RenderConfig()

        # Interfaces
        self._data_to_layer_itf = DataToLayerInterface(
            parent=self,
            viewer=napari_viewer,
            render_config=self.render_config,
            renderer=renderer,
        )
        self._data_to_layer_itf.on_grid_line_distance_clamped = (
            self._on_grid_line_distance_clamped
        )
        self._data_to_layer_itf.on_layer_updated = self._on_layer_updated
        self._data_to_layer_itf.on_resource_limit_applied = self._warn_user
        self._file_to_data_itf = FileToLocalizationDataInterface(parent=self)

        # Attributes
        self.testing_mode_enabled = False

        # Datasets are owned by the store, which assigns each a stable id and
        # announces openings and closings.  The renderer subscribes rather than
        # being told, so unloading no longer means remembering every dependant.
        self._dataset_store = DatasetStore()
        # Tracked so close_session can let go of them again; a subscription to a
        # bound method keeps this widget alive for as long as the store lives.
        self._store_listeners = [
            self._dataset_store.subscribe(self._data_to_layer_itf.on_store_event)
        ]
        self._closed = False
        self.n_datasets = 0
        self.viewer = napari_viewer

        self.gaussian_render_modes = ["Fixed-size gaussian", "Variable-size gaussian"]

        self._grid_plane_enabled = False
        self.grid_plane_standard_color_index = 0
        self.possible_grid_plane_orientations = ["XY", "YZ", "XZ"]
        self.active_grid_plane_orientation = self.possible_grid_plane_orientations[0]
        self.grid_plane_opacity = 0.7

        self.active_render_range_box_color = standard_colormaps[0]
        self.render_range_box_opacity = 0.25
        self._render_range_preview = None  # single shared Surface layer for range-box feedback

        self.pixelsize_nm = []
        self.channel = []
        self.image_layer_controls = []
        self.zoom = 0

        # GUI
        super().__init__()
        self._data_filter_itf = DataFilterInterface(
            parent=self, data_filter_window=self.datafilter_tab
        )
        # Like the renderer, the filter interface owns state keyed by dataset id
        # and releases it on the store's event rather than on being asked.
        self._store_listeners.append(
            self._dataset_store.subscribe(self._data_filter_itf.on_store_event)
        )
        self._data_adjustment_itf = DataAdjustmentInterface(
            parent=self, data_adjustment_window=self.data_adjustment_tab
        )

        # VisPy leaves one drag's camera state where the next drag reads it, so
        # releasing Shift mid-drag crashes the 3-D camera.  Re-applied when
        # napari swaps cameras on an ndisplay change.
        guard_camera_drag_state(napari_viewer)
        napari_viewer.dims.events.ndisplay.connect(
            lambda event=None: guard_camera_drag_state(napari_viewer)
        )

        # Init Function Calls
        custom_keys_and_scalebar(self)
        self.hide_non_available_widgets()
        self.hide_testing_mode()
        self._instances[id(napari_viewer)] = self

    @property
    def dataset_store(self):
        return self._dataset_store

    def close_session(self):
        """Release everything this dock owns.  Safe to call more than once.

        Without it, closing the dock left the renderer holding layers, the
        store holding subscriptions to methods of a dead widget, and the
        viewer holding callbacks into both.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for listener in list(self._store_listeners):
            self._dataset_store.unsubscribe(listener)
        self._store_listeners.clear()
        self._data_to_layer_itf.close()
        self._dataset_store.clear()

    def closeEvent(self, event):
        """Qt is tearing the dock down; let go of the viewer first."""
        self.close_session()
        super().closeEvent(event)

    @property
    def localization_datasets(self):
        """The live, ordered dataset list.  Add and remove via the store."""
        return self._dataset_store.datasets

    @property
    def grid_plane_enabled(self):
        return self._grid_plane_enabled

    @grid_plane_enabled.setter
    def grid_plane_enabled(self, value):
        if value == 1 or value == 0:
            self._grid_plane_enabled = value
            self.data_to_layer_itf.create_remove_grid_plane_state(value)
        else:
            raise ValueError(
                f"Grid Plane can only be enabled or disabled not {value} of type {type(value)}"
            )

    @property
    def data_to_layer_itf(self):
        return self._data_to_layer_itf

    @data_to_layer_itf.setter
    def data_to_layer_itf(self, value):
        raise StaticAttributeError(
            "the dataset to layer interface should never be changed"
        )

    @property
    def data_filter_itf(self):
        return self._data_filter_itf

    @data_filter_itf.setter
    def data_filter_itf(self, value):
        raise StaticAttributeError("the data filter interface should never be changed")

    @property
    def data_adjustment_itf(self):
        return self._data_adjustment_itf

    @data_adjustment_itf.setter
    def data_adjustment_itf(self, value):
        raise StaticAttributeError(
            "the data adjustment interface should never be changed"
        )

    @property
    def file_to_data_itf(self):
        return self._file_to_data_itf

    @file_to_data_itf.setter
    def file_to_data_itf(self, value):
        raise StaticAttributeError(
            "the file to dataset interface should never be changed"
        )

    @property
    def render_gaussian_mode(self):
        return self.render_config.gaussian_mode

    @render_gaussian_mode.setter
    def render_gaussian_mode(self, value):
        if value == 0 or value == 1:
            if self.render_config.z_color_encoding == 1 and value == 1:
                self.z_color_encoding_mode = 0
                self.Bz_color_coding.hide()
            else:
                self.Bz_color_coding.show()
            self.render_config.gaussian_mode = value
        else:
            raise ValueError(
                f"Wrong value for gaussian render mode, either 0 or 1 not {value} of type {type(value)}"
            )

    @property
    def z_color_encoding_mode(self):
        return self.render_config.z_color_encoding

    @z_color_encoding_mode.setter
    def z_color_encoding_mode(self, value):
        if value == 0 or value == 1:
            if self.render_config.gaussian_mode == 0:
                self.render_config.z_color_encoding = value
            else:
                self.render_config.z_color_encoding = 0
                raise ValueError("Color Encoding only works for fixed gaussian mode")
        else:
            raise ValueError(
                f"Wrong value for Z colorcoding, either 0 or 1 not {value} of type {type(value)}"
            )

    @property
    def zdim(self):
        return self.render_config.zdim

    @zdim.setter
    def zdim(self, boolean):
        """3D or 2D Mode"""
        if not isinstance(boolean, bool):
            raise ValueError(
                f"Zdim present can either be true or false not {boolean} of type {type(boolean)}"
            )
        if self.zdim is not None and self.zdim != boolean:
            raise DimensionError(
                "Error while merging, combination of 2D and 3D datasets not possible"
            )
        else:
            self.render_config.zdim = boolean
        self.adjust_available_options_to_data_dimension()

    # --- Delegating properties: render settings backed by render_config ---

    @property
    def render_fixed_gauss_sigma_xy_nm(self):
        return self.render_config.fixed_sigma_xy_nm

    @render_fixed_gauss_sigma_xy_nm.setter
    def render_fixed_gauss_sigma_xy_nm(self, value):
        self.render_config.fixed_sigma_xy_nm = value

    @property
    def render_fixed_gauss_sigma_z_nm(self):
        return self.render_config.fixed_sigma_z_nm

    @render_fixed_gauss_sigma_z_nm.setter
    def render_fixed_gauss_sigma_z_nm(self, value):
        self.render_config.fixed_sigma_z_nm = value

    @property
    def render_var_gauss_PSF_sigma_xy_nm(self):
        return self.render_config.var_psf_sigma_xy_nm

    @render_var_gauss_PSF_sigma_xy_nm.setter
    def render_var_gauss_PSF_sigma_xy_nm(self, value):
        self.render_config.var_psf_sigma_xy_nm = value

    @property
    def render_var_gauss_PSF_sigma_z_nm(self):
        return self.render_config.var_psf_sigma_z_nm

    @render_var_gauss_PSF_sigma_z_nm.setter
    def render_var_gauss_PSF_sigma_z_nm(self, value):
        self.render_config.var_psf_sigma_z_nm = value

    @property
    def render_var_gauss_sigma_min_xy_nm(self):
        return self.render_config.var_sigma_min_xy_nm

    @render_var_gauss_sigma_min_xy_nm.setter
    def render_var_gauss_sigma_min_xy_nm(self, value):
        self.render_config.var_sigma_min_xy_nm = value

    @property
    def render_var_gauss_sigma_min_z_nm(self):
        return self.render_config.var_sigma_min_z_nm

    @render_var_gauss_sigma_min_z_nm.setter
    def render_var_gauss_sigma_min_z_nm(self, value):
        self.render_config.var_sigma_min_z_nm = value

    @property
    def render_range_slider_x_percent(self):
        return self.render_config.range_x_percent

    @render_range_slider_x_percent.setter
    def render_range_slider_x_percent(self, value):
        self.render_config.range_x_percent = value

    @property
    def render_range_slider_y_percent(self):
        return self.render_config.range_y_percent

    @render_range_slider_y_percent.setter
    def render_range_slider_y_percent(self, value):
        self.render_config.range_y_percent = value

    @property
    def render_range_slider_z_percent(self):
        return self.render_config.range_z_percent

    @render_range_slider_z_percent.setter
    def render_range_slider_z_percent(self, value):
        self.render_config.range_z_percent = value

    @property
    def grid_plane_line_distance_um(self):
        return self.render_config.grid_plane_line_distance_um

    @grid_plane_line_distance_um.setter
    def grid_plane_line_distance_um(self, value):
        self.render_config.grid_plane_line_distance_um = value

    @property
    def grid_plane_margin_percent(self):
        return self.render_config.grid_plane_margin_percent

    @grid_plane_margin_percent.setter
    def grid_plane_margin_percent(self, value):
        self.render_config.grid_plane_margin_percent = value

    def update_render_range_box_opacity(self):
        self.render_range_box_opacity = self.Srender_range_box_opacity.value() / 100

    def update_render_range_box_color(self):
        idx = self.Brender_range_box_color.currentIndex()
        self.active_render_range_box_color = standard_colormaps[idx]

    def update_grid_plane_color(self):
        idx = self.Bgrid_plane_color.currentIndex()
        self.grid_plane_standard_color_index = idx
        self.data_to_layer_itf.current_grid_plane_color = standard_colors[idx]
        self.data_to_layer_itf.update_grid_plane(color=standard_colors[idx])

    def update_grid_plane_line_distance(self):
        value = _positive_number(self.Egrid_line_distance.text())
        if value is None:
            # Mid-edit, or not a number at all.  Leaving the last good value in
            # place is what the user expects; raising would surface a traceback
            # from the Qt event loop on a keystroke.
            return
        self.grid_plane_line_distance_um = value
        self.data_to_layer_itf.update_grid_plane(line_distance_nm=1000 * value)

    def update_grid_plane_margin(self):
        """Issue #38: how far past the data the grid is allowed to run.

        Rebuilt through the line-distance path because that is the one that
        remakes the vectors, and the geometry is what a margin changes.
        """
        value = _non_negative_number(self.Egrid_margin.text())
        if value is None:
            return
        self.grid_plane_margin_percent = value
        self.data_to_layer_itf.update_grid_plane(
            line_distance_nm=self.grid_plane_line_distance_um * 1000
        )

    def grid_plane(self):
        if self.Cgrid_plane.isChecked():
            self.Egrid_line_distance.show()
            self.Egrid_margin.show()
            self.Sgrid_line_thickness.show()
            self.Sgrid_z_pos.show()
            self.Bgrid_plane_color.show()
            self.grid_plane_enabled = 1

        else:
            self.Egrid_line_distance.hide()
            self.Egrid_margin.hide()
            self.Sgrid_line_thickness.hide()
            self.Sgrid_z_pos.hide()
            self.grid_plane_enabled = 0
            self.Bgrid_plane_color.hide()

    def _sync_scalebar_config(self):
        """Sync scalebar GUI state to render_config, then update the layer."""
        self.render_config.scalebar_enabled = self.Cscalebar.isChecked()
        # A validator keeps most nonsense out of the field, but it still admits
        # an empty string and a lone "-" while typing, and `int()` on either
        # raises into the Qt event loop (§7.2: no numeric control may accept
        # input that cannot be parsed).
        size_nm = _positive_number(self.Esbsize.text())
        if size_nm is None:
            return
        self.render_config.scalebar_size_nm = int(size_nm)
        self.data_to_layer_itf.scalebar()

    def _on_grid_line_distance_clamped(self, max_distance_um):
        """Callback: DTL detected grid line distance is too large, update GUI."""
        self.Egrid_line_distance.setText(str(max_distance_um))

    def _warn_user(self, message):
        """Surface something the user needs to know about their data.

        Routed through napari's notification system rather than ``print()``:
        the user is looking at the viewer, and quietly drawing a subsample of
        their data -- or quietly failing to open their file -- would be worse
        than not drawing it at all.
        """
        try:
            from napari.utils.notifications import show_warning

            show_warning(f"napari-storm: {message}")
        except Exception:
            # A missing notification manager (headless tests, an embedded
            # viewer) must not turn a resource limit into a failed render.
            logging.getLogger(__name__).warning("napari-storm: %s", message)

    def _on_layer_updated(self, channel_index):
        """Callback: DTL finished updating a layer, refresh channel controls."""
        self.channel[channel_index].adjust_colormap_range()
        self.channel[channel_index].adjust_z_color_encoding_opacity()
        self.channel[channel_index].change_color_map()

    def scalebar_state_changed(self):
        if self.Cscalebar.isChecked():
            self.Lscalebarsize.show()
            self.Esbsize.show()
        else:
            self.Lscalebarsize.hide()
            self.Esbsize.hide()
        self._sync_scalebar_config()

    def clear_datasets(self):
        """Unload all localization datasets and reset their dependent state."""
        self.Cgrid_plane.setCheckState(Qt.CheckState.Unchecked)
        self.n_datasets = 0
        # Emits DatasetClosed per dataset and then StoreCleared, which is what
        # releases the renderer's per-dataset state.
        self._dataset_store.clear()
        for controls in self.channel:
            self.channel_controls_widget_layout.removeWidget(controls)
            controls.setParent(None)
            controls.deleteLater()
        self.channel.clear()
        self.data_filter_itf.clear_entries()
        self.data_adjustment_itf.clear_entries()
        self.file_to_data_itf.clear_entries()
        self.Cscalebar.setCheckState(Qt.CheckState.Unchecked)
        if (
            self._render_range_preview is not None
            and self._render_range_preview in self.viewer.layers
        ):
            self.viewer.layers.remove(self._render_range_preview.name)
        self._render_range_preview = None
        self.reset_render_range(full_reset=True)
        self.Lnumberoflocs.clear()
        self.Bz_color_coding.setCheckState(Qt.CheckState.Unchecked)
        self.render_config.zdim = None
        self.hide_non_available_widgets()

    def unload_dataset(self, dataset_or_index):
        """Unload one localization dataset and all of its associated UI state."""
        if isinstance(dataset_or_index, int):
            dataset_index = dataset_or_index
        else:
            try:
                dataset_index = next(
                    index
                    for index, dataset in enumerate(self.localization_datasets)
                    if dataset is dataset_or_index
                )
            except StopIteration:
                return False

        if not 0 <= dataset_index < len(self.localization_datasets):
            return False
        if len(self.localization_datasets) == 1:
            self.clear_datasets()
            return True

        dataset = self.localization_datasets[dataset_index]
        # Removing it from the store closes its layer and releases every piece
        # of per-dataset state subscribed to that event.
        self._dataset_store.remove(dataset)
        self.n_datasets = len(self.localization_datasets)

        controls = self.channel.pop(dataset_index)
        self.channel_controls_widget_layout.removeWidget(controls)
        controls.setParent(None)
        controls.deleteLater()

        self.data_filter_itf.remove_dataset_entry(dataset_index)
        self.data_adjustment_itf.remove_dataset_entry(dataset_index)
        self.Lnumberoflocs.remove_dataset(dataset_index)
        self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)

        # Recompute only the aggregate extents.  The surviving layers and
        # controls keep their identities, avoiding a costly all-layer rebuild.
        self.data_to_layer_itf.set_render_range_and_offset()
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(
                line_distance_nm=self.grid_plane_line_distance_um * 1000
            )
        return True

    def reset_render_range(self, full_reset=False):
        self.Srender_rangex.reset()
        self.Srender_rangey.reset()
        self.Srender_rangez.reset()
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(
                line_distance_nm=self.grid_plane_line_distance_um * 1000
            )
        if not full_reset:
            self.data_to_layer_itf.update_layers(self)
            self.move_camera_center_to_render_range_center()

    def update_render_range(self, slider_type, values):
        if slider_type == "x":
            self.render_range_slider_x_percent = values
        elif slider_type == "y":
            self.render_range_slider_y_percent = values
        else:
            self.render_range_slider_z_percent = values
            self._refresh_z_colorbar()
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(
                line_distance_nm=self.grid_plane_line_distance_um * 1000
            )
        self.move_camera_center_to_render_range_center()

    def move_camera_center_to_render_range_center(self):
        itf = self.data_to_layer_itf
        if not (
            itf.render_range_x[1] == -np.inf
            or itf.render_range_y[1] == -np.inf
        ):
            # Midpoint of the selected percentage window, mapped onto the true
            # nanometre extent of each axis.  This used to multiply the axis
            # maximum alone, which was only correct while the auto-offset forced
            # every axis to start at zero.
            def _window_centre(axis_range, percent_pair):
                low, high = itf.percent_to_absolute(axis_range, percent_pair)
                return 0.5 * (low + high)

            # render_range_x/y now hold the x/y extents in both 2-D and 3-D.
            # This used to read them swapped, which was right for 2-D only
            # because set_render_range stored them swapped there.
            tmp_x_center_nm = _window_centre(
                itf.render_range_x, self.render_range_slider_x_percent
            )
            tmp_y_center_nm = _window_centre(
                itf.render_range_y, self.render_range_slider_y_percent
            )
            if self.zdim and itf.render_range_z[1] != -np.inf:
                tmp_z_center_nm = _window_centre(
                    itf.render_range_z, self.render_range_slider_z_percent
                )
            else:
                # 2-D localization layers live in the z=1 plane.
                tmp_z_center_nm = 1.0
            self.data_to_layer_itf.camera[1] = (
                tmp_z_center_nm,
                tmp_x_center_nm,
                tmp_y_center_nm,
            )
            self.viewer.camera.center = (
                tmp_z_center_nm,
                tmp_x_center_nm,
                tmp_y_center_nm,
            )

    def add_channel(self, name="Channel"):
        """Adds a Channel in the visual controls"""
        self.channel.append(
            ChannelControls(parent=self, name=name, channel_index=len(self.channel))
        )
        self.channel_controls_widget_layout.addRow(self.channel[-1])
        # A second dataset changes both the z extent and whether one pair of
        # numbers can speak for the scene at all.
        self._refresh_z_colorbar()

    def _refresh_z_colorbar(self):
        """Label the z colour bar with the interval it currently encodes.

        The planner normalizes each dataset against its own z extent, so the
        numbers only describe the whole scene while one dataset is loaded;
        past that the bar says so rather than picking a channel to speak for.
        """
        itf = getattr(self, "data_to_layer_itf", None)
        if itf is None:
            return
        low, high = itf.percent_to_absolute(
            itf.render_range_z, self.render_config.range_z_percent
        )
        self.Lcolor_encoding_bar.set_range(
            low, high, shared=len(self.localization_datasets) <= 1
        )

    def colorcoding(self):
        """Check if Colorcoding is choosen"""
        enabled = self.Bz_color_coding.isChecked()
        self.z_color_encoding_mode = enabled
        if enabled:
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.hide()
                self.channel[i].Slider_opacity.show()
                self.channel[i].Slider_colormap_range.hide()
                self.channel[i].reset()
                self.channel[i].Label.setText("Contrast " + self.channel[i].name)
                self.Lcolor_encoding_bar.set_pixmap(
                    self.data_to_layer_itf.colormap_icons[-1]
                )
                self.Lcolor_encoding_bar.show()
            self._refresh_z_colorbar()
        else:
            for i in range(len(self.channel)):
                self.channel[i].Colormap_selector.show()
                self.channel[i].Slider_opacity.hide()
                self.channel[i].Slider_colormap_range.show()
                self.channel[i].Label.setText("Opacity " + self.channel[i].name)
                self.channel[i].reset()
                self.Lcolor_encoding_bar.hide()
        self.data_to_layer_itf.update_layer_appearance()

    def _render_options_changed(self):
        only_storm_datasets = True
        zdim_present = True
        for dataset in self.localization_datasets:
            if not isinstance(dataset, StormDataClass):
                only_storm_datasets = False
            if not dataset.zdim_present:
                zdim_present = False
        if only_storm_datasets is False:
            self.Brenderoptions.setCurrentIndex(0)
        if self.Brenderoptions.currentText() == self.gaussian_render_modes[1]:
            self.render_gaussian_mode = 1
            self.Lsigma_xy.setText("PSF FWHM in XY [nm]")
            self.Lsigma_z.setText("PSF FWHM in Z [nm]")
            self.Esigma_xy.setText("300")
            self.Esigma_z.setText("700")
            self.Bz_color_coding.hide()
            self.Bz_color_coding.setCheckState(Qt.CheckState.Unchecked)
            self.Lcolor_encoding_bar.hide()
            self.Esigma_min_xy.show()
            if self.zdim:
                self.Esigma_min_z.show()
                self.Lsigma_z_min.show()
            self.Lsigma_xy_min.show()

        else:
            self.render_gaussian_mode = 0
            self.Lsigma_z.setText("FWHM in Z [nm]")
            self.Lsigma_xy.setText("FWHM in XY [nm]")
            self.Esigma_xy.setText("20")
            self.Esigma_z.setText("20")
            self.Bz_color_coding.show()
            self.Esigma_min_xy.hide()
            self.Esigma_min_z.hide()
            self.Lsigma_xy_min.hide()
            self.Lsigma_z_min.hide()
        if not zdim_present:
            self.Bz_color_coding.hide()
        self.data_to_layer_itf.update_layer_appearance()

    def _start_typing_timer(self, timer):
        timer.start(500)

    def change_camera(self, set_view_to="XY"):
        v = napari.current_viewer()
        values = {}
        if set_view_to == "XY":
            v.camera.angles = (90, 0, -90)
        elif set_view_to == "XZ":
            v.camera.angles = (-180, 90, 180)
        else:
            v.camera.angles = (-180, 0, -180)
        v.camera.center = self.data_to_layer_itf.camera[1]
        v.camera.zoom = self.zoom
        v.camera.update(values)

    @staticmethod
    def _read_fwhm(field, current_sigma_nm):
        """Read a FWHM entry field and return the corresponding sigma in nm.

        The field carries a QDoubleValidator, but a validator still permits
        intermediate states -- an empty field, a lone "-", a partially typed
        number -- and this runs from a debounce timer rather than on commit.
        Anything unusable leaves the current value in place instead of raising
        out of a Qt slot.
        """
        try:
            fwhm_nm = float(field.text())
        except (TypeError, ValueError):
            return current_sigma_nm
        if not np.isfinite(fwhm_nm):
            return current_sigma_nm
        fwhm_nm = min(max(fwhm_nm, MIN_FWHM_NM), MAX_FWHM_NM)
        return fwhm_nm / FWHM_TO_SIGMA

    def update_sigma(self):
        if self.render_gaussian_mode == 0:
            # fixed gaussian
            self.render_fixed_gauss_sigma_xy_nm = self._read_fwhm(
                self.Esigma_xy, self.render_fixed_gauss_sigma_xy_nm
            )
            self.render_fixed_gauss_sigma_z_nm = self._read_fwhm(
                self.Esigma_z, self.render_fixed_gauss_sigma_z_nm
            )
        else:
            self.render_var_gauss_PSF_sigma_xy_nm = self._read_fwhm(
                self.Esigma_xy, self.render_var_gauss_PSF_sigma_xy_nm
            )
            self.render_var_gauss_PSF_sigma_z_nm = self._read_fwhm(
                self.Esigma_z, self.render_var_gauss_PSF_sigma_z_nm
            )
            self.render_var_gauss_sigma_min_xy_nm = self._read_fwhm(
                self.Esigma_min_xy, self.render_var_gauss_sigma_min_xy_nm
            )
            self.render_var_gauss_sigma_min_z_nm = self._read_fwhm(
                self.Esigma_min_z, self.render_var_gauss_sigma_min_z_nm
            )
        self.data_to_layer_itf.update_layer_appearance()

    def allow_variable_gaussian_mode_for_storm_datasets(self):
        only_storm_datasets = True
        for dataset in self.localization_datasets:
            if not isinstance(dataset, LocalizationDataBaseClass):
                only_storm_datasets = False

        if not only_storm_datasets:
            self.Brenderoptions.removeItem(1)

    def open_localization_data_file_and_get_dataset(
        self,
        merge=False,
        file_path=None,
        file_recognition=False,
        custom_import=False,
        background=False,
    ):
        """Open a localization file and add it to the session.

        With ``background=False`` this reads the file inline and returns True or
        False, which is what the reader hook and the tests rely on.  With
        ``background=True`` -- how the import buttons call it -- the read runs on
        a worker behind a cancellable progress dialog and the method returns
        immediately; the datasets are applied from the worker's callback.  In
        that mode the return value is the worker (whose ``load_handle`` can
        cancel it), or ``True`` if no worker was available and it ran inline, or
        ``False`` if the user closed the file picker.
        """
        if background:
            return self._open_in_background(
                merge=merge,
                file_path=file_path,
                file_recognition=file_recognition,
                custom_import=custom_import,
            )
        # File selection and import happen before touching the current session.
        # Closing a dialog is therefore a no-op instead of an accidental clear.
        datasets = self._file_to_data_itf.open_localization_data_file_and_get_dataset(
            file_path=file_path,
            file_type_recognizer=file_recognition,
            custom_import=custom_import,
        )
        if not datasets:
            return False
        return self._apply_loaded_datasets(datasets, merge=merge)

    def _open_in_background(
        self, merge, file_path, file_recognition, custom_import
    ):
        """Read a localization file on a worker, then apply it on the GUI thread.

        Picking the file stays here: a modal file dialog belongs to the gesture
        that opened it, and starting a worker only to immediately block it on a
        dialog would defeat the point.
        """
        if file_path is None:
            file_path = self._file_to_data_itf.choose_file_path()
            if not file_path:
                return False

        def _read(handle):
            return self._file_to_data_itf.open_localization_data_file_and_get_dataset(
                file_path=file_path,
                file_type_recognizer=file_recognition,
                custom_import=custom_import,
                handle=handle,
            )

        def _apply(datasets):
            if datasets:
                self._apply_loaded_datasets(datasets, merge=merge)

        def _failed(error):
            self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)
            self._warn_user(f"could not open {file_path}: {error}")

        def _cancelled():
            # The importers roll their own bookkeeping back on failure; this
            # re-syncs it against what is actually loaded either way.
            self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)

        worker = load_in_background(
            _read,
            on_result=_apply,
            on_error=_failed,
            on_cancelled=_cancelled,
            description=f"Reading {file_path}…",
            parent=self,
        )
        return worker if worker is not None else True

    def _apply_loaded_datasets(self, datasets, merge=False):
        """Add already-read datasets to the session.  GUI thread only."""
        incoming_dimensions = {bool(dataset.zdim_present) for dataset in datasets}
        if len(incoming_dimensions) != 1:
            self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)
            raise DimensionError("A single import cannot mix 2D and 3D datasets")
        incoming_zdim = incoming_dimensions.pop()
        if merge and self.zdim is not None and self.zdim != incoming_zdim:
            self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)
            raise DimensionError(
                "Merging 2D and 3D localization datasets is not supported"
            )

        if not merge:
            self.clear_datasets()
            self.Cgrid_plane.setCheckState(Qt.CheckState.Unchecked)
        self.show_avaiable_widgets()
        self.zdim = incoming_zdim
        for dataset in datasets:
            self._dataset_store.add(dataset)
            self.add_dataset_entries_for_all_itfs(dataset=dataset)
            self.n_datasets += 1
            self.create_layer(
                self.localization_datasets[-1],
                idx=len(self.localization_datasets) - 1,
                merge=merge,
            )
        self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)
        if self.Cgrid_plane.isChecked():
            self.data_to_layer_itf.update_grid_plane(
                line_distance_nm=self.grid_plane_line_distance_um * 1000
            )
        return True

    def add_dataset_entries_for_all_itfs(self, dataset):
        self.data_filter_itf.add_dataset_entry(dataset.name)
        self.data_adjustment_itf.add_dataset_entry(dataset.name)

    def get_dataset_from_test_mode(self, datasets):
        self.clear_datasets()
        self.show_avaiable_widgets()
        if datasets[-1].zdim_present:
            self.zdim = True
        else:
            self.zdim = False
        for i in range(len(datasets)):
            self._dataset_store.add(datasets[i])
            self.add_dataset_entries_for_all_itfs(dataset=datasets[i])
            self.n_datasets += 1
            self.create_layer(self.localization_datasets[-1], idx=i)
        self.file_to_data_itf.sync_dataset_entries(self.localization_datasets)

    # ------------------------------------------------------------------
    # Scene persistence
    # ------------------------------------------------------------------

    def save_scene_dialog(self, *_):
        """Ask where, then save."""
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            self, "Save scene", "session.napari-storm.json", "Scene (*.json)"
        )
        if not path:
            return None
        try:
            return self.save_scene_to(path)
        except OSError as error:
            self._warn_user(f"could not save scene: {error}")
            return None

    def load_scene_dialog(self, *_):
        """Ask which, then apply."""
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            self, "Load scene", "", "Scene (*.json)"
        )
        if not path:
            return None
        return self.load_scene_from(path)

    def current_scene(self):
        """This session's decisions, as a value.

        Decisions only -- no localizations and no pixels. Reloading re-reads the
        files and re-applies these, so a scene cannot go stale against the data
        it describes.
        """
        from .core import CameraState, DatasetEntry, ReferenceImageEntry, Scene

        store = self.dataset_store
        datasets = []
        for dataset in self.localization_datasets:
            state = store.state(getattr(dataset, "dataset_id", None))
            datasets.append(
                DatasetEntry(
                    name=getattr(dataset, "name", "dataset"),
                    source_path=getattr(dataset, "file_path", None),
                    transform=state.transform if state else None or WorldTransform(),
                    appearance=state.appearance if state else LayerAppearance(),
                )
            )

        images = []
        for controls in self.image_layer_controls:
            layer = getattr(controls, "_layer", None)
            if layer is None:
                continue
            translate = tuple(float(v) for v in layer.translate)
            scale = tuple(float(v) for v in layer.scale)
            images.append(
                ReferenceImageEntry(
                    name=layer.name,
                    source_path=getattr(controls, "_file_path", None),
                    orientation=getattr(controls, "_orientation", "XY"),
                    pixel_size_xy_nm=scale[1],
                    pixel_size_z_nm=scale[0],
                    offset_nm=translate,
                    appearance=LayerAppearance(
                        opacity=float(layer.opacity), visible=bool(layer.visible)
                    ),
                )
            )

        camera = self.viewer.camera
        return Scene(
            datasets=tuple(datasets),
            reference_images=tuple(images),
            gaussian=self.data_to_layer_itf.gaussian_settings(),
            camera=CameraState(
                center_nm=tuple(float(v) for v in camera.center),
                zoom=float(camera.zoom),
                angles=tuple(float(v) for v in camera.angles),
                ndisplay=int(self.viewer.dims.ndisplay),
            ),
        )

    def save_scene_to(self, path):
        """Write the current scene to *path*."""
        from .core import save_scene

        return save_scene(path, self.current_scene())

    def apply_scene(self, scene):
        """Re-apply a scene's decisions to whatever is loaded now.

        Deliberately does *not* open files: which datasets are loaded is the
        user's business, and a scene that silently loaded four gigabytes on
        open would be a surprise rather than a convenience. Entries are matched
        to loaded datasets by name; anything unmatched is reported, not guessed
        at.
        """
        from .core import WorldTransform as _WorldTransform

        by_name = {
            getattr(dataset, "name", None): dataset
            for dataset in self.localization_datasets
        }
        unmatched = []
        for entry in scene.datasets:
            dataset = by_name.get(entry.name)
            if dataset is None:
                unmatched.append(entry.name)
                continue
            dataset_id = getattr(dataset, "dataset_id", None)
            if dataset_id is None:
                continue
            self.dataset_store.set_transform(
                dataset_id, entry.transform or _WorldTransform()
            )
            appearance = entry.appearance
            if appearance is not None:
                self.data_to_layer_itf.set_appearance(
                    dataset,
                    colormap=appearance.colormap,
                    opacity=appearance.opacity,
                    contrast_limits=appearance.contrast_limits,
                    visible=appearance.visible,
                )

        camera = scene.camera
        self.viewer.dims.ndisplay = camera.ndisplay
        self.viewer.camera.center = camera.center_nm
        self.viewer.camera.zoom = camera.zoom
        self.viewer.camera.angles = camera.angles

        if unmatched:
            self._warn_user(
                "the scene refers to datasets that are not loaded: "
                + ", ".join(unmatched)
                + ". Open them and load the scene again to place them."
            )
        return unmatched

    def load_scene_from(self, path):
        """Read a scene from *path* and apply it."""
        from .core import SceneFormatError, load_scene

        try:
            scene = load_scene(path)
        except (OSError, SceneFormatError) as error:
            self._warn_user(f"could not load scene: {error}")
            return None
        self.apply_scene(scene)
        return scene

    def export_image(
        self, _checked=False, options=None, path=None, force_synchronous=False
    ):
        """Rasterize the reconstruction and write it as a calibrated OME-TIFF.

        Pass *options* and *path* to skip the dialog, which is what tests and
        scripted exports do; the interactive path leaves both as None.

        *force_synchronous* writes on the calling thread and returns only once
        the file is complete. The interactive path never uses it -- a large
        export would freeze the window, which is the thing this is built to
        avoid -- but a caller that needs the file to exist when it returns has
        no other way to say so.
        """
        from .image_export import plan_from_widget, run_export

        if not self.localization_datasets:
            self._warn_user("there is nothing loaded to export")
            return None

        if options is None or path is None:
            from .pyqt.export_dialog import ExportImageDialog

            dialog = ExportImageDialog(
                lambda chosen: plan_from_widget(self, chosen), parent=self
            )
            if dialog.exec_() != dialog.Accepted:
                return None
            options, path = dialog.options(), dialog.output_path()

        try:
            plan = plan_from_widget(self, options)
        except Exception as error:  # noqa: BLE001 - reported, not raised at the user
            self._warn_user(f"cannot export: {error}")
            return None

        return run_export(
            path,
            plan,
            parent=self,
            force_synchronous=force_synchronous,
            on_finished=lambda written: self._report_export_finished(written, plan),
            on_error=lambda error: self._warn_user(f"export failed: {error}"),
            on_cancelled=lambda: self._warn_user("export cancelled"),
        )

    def _report_export_finished(self, path, plan):
        try:
            from napari.utils.notifications import show_info

            show_info(f"napari-storm: wrote {path} ({plan.n_localizations:,} localizations)")
        except Exception:
            logging.getLogger(__name__).info("napari-storm: wrote %s", path)

    def open_image_file(self, file_path=None):
        """Open the image-import dialog, then load the image as a napari layer.

        If *file_path* is given the dialog is pre-loaded with that file so the
        user only needs to adjust parameters and click Import.
        """
        from .pyqt.image_import_dialog import ImageImportDialog

        dlg = ImageImportDialog(
            file_path=file_path,
            parent=self,
            default_z_off_nm=self.data_to_layer_itf.reference_plane_z_nm(),
        )
        if not dlg.exec_():
            return
        result = dlg.get_result()
        if result is None:
            return
        self._add_image_layer_from_result(result)

    def _add_image_layer_from_result(self, result):
        """Create napari Image layer + ImageLayerControls from an ImageImportResult."""
        from .pyqt.image_layer_controls import (
            ImageLayerControls,
            _expand_image,
            _is_rgb_reference,
            _reference_image_rendering_options,
        )

        try:
            data, scale, translate = _expand_image(result)
        except Exception as exc:
            self._warn_user(f"image preparation failed: {exc}")
            return

        try:
            rgb = _is_rgb_reference(result)
            rendering_options = _reference_image_rendering_options(
                data, result.orientation, rgb=rgb
            )
            layer = self.viewer.add_image(
                data,
                name=result.layer_name,
                scale=scale,
                rgb=rgb,
                **rendering_options,
            )
            layer.translate = translate
        except Exception as exc:
            self._warn_user(f"could not add image layer: {exc}")
            return

        # Place beneath localisation layers
        try:
            self.viewer.layers.move(len(self.viewer.layers) - 1, 0)
        except Exception:
            pass

        # Create live controls and insert at top of channel-controls section
        controls = ImageLayerControls(
            layer=layer, result=result, dock_widget=self
        )
        self.image_layer_controls.append(controls)
        self.channel_controls_widget_layout.insertRow(0, controls)

    def create_layer(self, dataset, idx=-1, merge=False):
        self.adjust_available_options_to_data_dimension()
        self.Lnumberoflocs.show_infos(filename=dataset.name, idx=idx)
        self.data_to_layer_itf.create_new_layer(
            dataset, layer_name=dataset.name, idx=idx, merge=merge
        )
        self.zoom = self.viewer.camera.zoom
        self.add_channel(name=dataset.name)
        self.channel[-1].change_color_map()
        self.channel[-1].adjust_colormap_range()
