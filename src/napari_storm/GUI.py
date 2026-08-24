import logging
import numpy as np
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator, QFont
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout,
                            QLabel, QLineEdit, QListWidget, QPushButton,
                            QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
                            QWidget)

from napari_storm.CustomErrors import ParentError
from napari_storm.ns_constants import (FWHM_TO_SIGMA, MAX_FWHM_NM,
                                       MIN_FWHM_NM, standard_colors)
from napari_storm.pyqt.dataset_info_widget import DatasetInfoPanel
from napari_storm.pyqt.GridPlaneSlider import GridPlaneSlider
from napari_storm.pyqt.PyQTvisuals import QHSeperationLine
from napari_storm.pyqt.RenderRangeSlider import RangeSlider2

from .DataAdjustment import DataAdjustmentWindow
from .DataFilter import DataFilterWindow
from .pyqt.detachable_tab import DetachableTabWidget
from .Test_Mode import TestModeWindow

_LOG = logging.getLogger(__name__)


class NapariStormGUI(QWidget):
    #: Width the controls actually need.  The widest rows are the two-button
    #: pairs and the render-range sliders; below this Qt clips their labels and
    #: the dock has to be dragged wider before anything can be read.  It is a
    #: minimum rather than a fixed width, so the dock stays resizable.
    PREFERRED_WIDTH_PX = 420

    def __init__(self):
        super().__init__()

        # GUI
        self.setAcceptDrops(True)
        self.setMinimumWidth(self.PREFERRED_WIDTH_PX)

        self.tabs = DetachableTabWidget()
        # Tabs
        self.data_control_tab = QWidget()
        self.infos_tab = QWidget()
        self.decorator_tab = QWidget()
        self.datafilter_tab = DataFilterWindow(parent=self)
        self.data_adjustment_tab = DataAdjustmentWindow(parent=self)

        self.test_mode_tab = TestModeWindow(parent=self)

        self.tabs.addTab(self.data_control_tab, "Data Controls")
        # Keep the controls usable as dataset cards are appended.  The content
        # widget grows to its size hint while the tab itself remains bounded by
        # the napari dock, so Qt supplies a vertical scrollbar when needed.
        self.data_controls_content = QWidget()
        self.data_controls_content.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.data_controls_tab_layout = QGridLayout(self.data_controls_content)

        # self.tabs.tabBarClicked.connect(self.handle_tab_bar_clicked)

        # ── Import section ────────────────────────────────────────────────
        # Row 0: primary import button (full width)
        self.Bopen = QPushButton("Import Localization File…")
        self.Bopen.setToolTip(
            "Open a localization file (HDF5/YAML, CSV, SMLM, NPY, JSON, …)"
        )
        self.Bopen.clicked.connect(
            lambda: self.open_localization_data_file_and_get_dataset(background=True)
        )
        self.data_controls_tab_layout.addWidget(self.Bopen, 0, 0, 1, 4)

        # Row 1: merge button — hidden until first dataset is loaded
        self.Bmerge_with_additional_file = QPushButton("⊕  Merge with Additional File")
        self.Bmerge_with_additional_file.setToolTip(
            "Add another localization file to the current dataset (same FOV)"
        )
        self.data_controls_tab_layout.addWidget(
            self.Bmerge_with_additional_file, 1, 0, 1, 4
        )
        self.Bmerge_with_additional_file.clicked.connect(
            lambda: self.open_localization_data_file_and_get_dataset(
                merge=True, background=True
            )
        )

        # Row 2: advanced import options (auto-detect format, custom importer)
        _adv_label = QLabel("Advanced:")
        _adv_label.setStyleSheet("color: rgb(130, 142, 154); font-size: 10px;")
        self.data_controls_tab_layout.addWidget(_adv_label, 2, 0)

        self.Bimport_by_file_recognition = QPushButton("Auto-detect Format")
        self.Bimport_by_file_recognition.setToolTip(
            "Attempt automatic format recognition for unsupported file types"
        )
        self.data_controls_tab_layout.addWidget(
            self.Bimport_by_file_recognition, 2, 1, 1, 2
        )
        self.Bimport_by_file_recognition.clicked.connect(
            lambda: self.open_localization_data_file_and_get_dataset(
                merge=False, file_recognition=True, background=True
            )
        )

        self.Bcustom_import = QPushButton("Custom")
        self.Bcustom_import.setToolTip("Run the user-defined custom import function")
        self.data_controls_tab_layout.addWidget(self.Bcustom_import, 2, 3)
        self.Bcustom_import.clicked.connect(
            lambda: self.open_localization_data_file_and_get_dataset(
                merge=False, custom_import=True, background=True
            )
        )

        # Row 3-ish: image layer import
        self.Bimport_image = QPushButton("Import Reference Image…")
        self.Bimport_image.setToolTip(
            "Load a TIFF / PNG / JPEG as a napari Image layer beneath the "
            "localisation data (useful for widefield overlays)"
        )
        self.Bimport_image.clicked.connect(self.open_image_file)
        self.data_controls_tab_layout.addWidget(self.Bimport_image, 3, 0, 1, 4)

        self.Lresetview = QLabel()
        self.Lresetview.setText("Reset view:")
        self.data_controls_tab_layout.addWidget(self.Lresetview, 4, 0)

        self.Baxis_xy = QPushButton()
        self.Baxis_xy.setText("XY")
        self.Baxis_xy.clicked.connect(lambda: self.change_camera(set_view_to="XY"))
        self.Baxis_xy.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xy, 4, 1)

        self.Baxis_yz = QPushButton()
        self.Baxis_yz.setText("YZ")
        self.Baxis_yz.clicked.connect(lambda: self.change_camera(set_view_to="YZ"))
        self.Baxis_yz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_yz, 4, 2)

        self.Baxis_xz = QPushButton()
        self.Baxis_xz.setText("XZ")
        self.Baxis_xz.clicked.connect(lambda: self.change_camera(set_view_to="XZ"))
        self.Baxis_xz.setFixedSize(75, 20)
        self.data_controls_tab_layout.addWidget(self.Baxis_xz, 4, 3)

        self.Lrenderoptions = QLabel()
        self.Lrenderoptions.setText("Rendering options:")
        self.data_controls_tab_layout.addWidget(self.Lrenderoptions, 5, 0)

        self.Brenderoptions = QComboBox()
        self.Brenderoptions.addItems(self.gaussian_render_modes)
        self.Brenderoptions.currentIndexChanged.connect(self._render_options_changed)
        self.data_controls_tab_layout.addWidget(self.Brenderoptions, 5, 1, 1, 3)

        self.Lsigma_xy = QLabel()
        self.Lsigma_xy.setText("FWHM in XY [nm]:")
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy, 6, 0)

        self.Lsigma_z = QLabel()
        self.Lsigma_z.setText("FWHM in Z [nm]:")
        self.data_controls_tab_layout.addWidget(self.Lsigma_z, 7, 0)

        self.Lsigma_xy_min = QLabel()
        self.Lsigma_xy_min.setText("Min. FWHM in XY [nm]:")
        self.data_controls_tab_layout.addWidget(self.Lsigma_xy_min, 8, 0)

        self.Lsigma_z_min = QLabel()
        self.Lsigma_z_min.setText("Min. FWHM in Z [nm]:")
        self.data_controls_tab_layout.addWidget(self.Lsigma_z_min, 9, 0)

        self.Esigma_xy = QLineEdit()
        self.Esigma_xy.setValidator(self._make_fwhm_validator())
        self.Esigma_xy.setText(str(self.render_fixed_gauss_sigma_xy_nm * FWHM_TO_SIGMA))
        self.Esigma_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_xy, 6, 1, 1, 3)
        self.typing_timer_sigma = QtCore.QTimer()
        self.typing_timer_sigma.setSingleShot(True)
        self.typing_timer_sigma.timeout.connect(self.update_sigma)

        self.Esigma_z = QLineEdit()
        self.Esigma_z.setValidator(self._make_fwhm_validator())
        self.Esigma_z.setText(str(self.render_fixed_gauss_sigma_z_nm * FWHM_TO_SIGMA))
        self.Esigma_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_z, 7, 1, 1, 3)

        self.Esigma_min_xy = QLineEdit()
        self.Esigma_min_xy.setValidator(self._make_fwhm_validator())
        self.Esigma_min_xy.setText(str(self.render_var_gauss_sigma_min_xy_nm * FWHM_TO_SIGMA))
        self.Esigma_min_xy.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_xy, 8, 1, 1, 3)

        self.Esigma_min_z = QLineEdit()
        self.Esigma_min_z.setValidator(self._make_fwhm_validator())
        self.Esigma_min_z.setText(str(self.render_var_gauss_sigma_min_z_nm * FWHM_TO_SIGMA))
        self.Esigma_min_z.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sigma)
        )
        self.data_controls_tab_layout.addWidget(self.Esigma_min_z, 9, 1, 1, 3)

        self.HL1 = QHSeperationLine()
        self.data_controls_tab_layout.addWidget(self.HL1, 10, 0, 1, 4)

        self.Lrangex = QLabel()
        self.Lrangex.setText("X-range")
        self.data_controls_tab_layout.addWidget(self.Lrangex, 11, 0)

        self.Lrangey = QLabel()
        self.Lrangey.setText("Y-range")
        self.data_controls_tab_layout.addWidget(self.Lrangey, 12, 0)

        self.Lrangez = QLabel()
        self.Lrangez.setText("Z-range")
        self.data_controls_tab_layout.addWidget(self.Lrangez, 13, 0)

        self.Srender_rangex = RangeSlider2(parent=self, type="x")
        self.data_controls_tab_layout.addWidget(self.Srender_rangex, 11, 1, 1, 3)

        self.Srender_rangey = RangeSlider2(parent=self, type="y")
        self.data_controls_tab_layout.addWidget(self.Srender_rangey, 12, 1, 1, 3)

        self.Srender_rangez = RangeSlider2(parent=self, type="z")
        self.data_controls_tab_layout.addWidget(self.Srender_rangez, 13, 1, 1, 3)

        self.Breset_render_range = QPushButton()
        self.Breset_render_range.setText("Reset Render Range")
        self.Breset_render_range.clicked.connect(self.reset_render_range)
        self.data_controls_tab_layout.addWidget(self.Breset_render_range, 14, 0, 1, 2)

        self.Bsave_scene = QPushButton("Save Scene…")
        self.Bsave_scene.setToolTip(
            "Save this session's decisions -- alignment, colours, render "
            "settings, camera -- as a small JSON file. Localizations are not "
            "copied into it; the scene points at the files it came from."
        )
        self.Bsave_scene.clicked.connect(self.save_scene_dialog)
        self.data_controls_tab_layout.addWidget(self.Bsave_scene, 22, 0, 1, 2)

        self.Bload_scene = QPushButton("Load Scene…")
        self.Bload_scene.setToolTip(
            "Re-apply a saved scene to the datasets loaded now. It does not "
            "open files: which data is loaded stays your choice."
        )
        self.Bload_scene.clicked.connect(self.load_scene_dialog)
        self.data_controls_tab_layout.addWidget(self.Bload_scene, 22, 2, 1, 2)

        self.Bexport_image = QPushButton("Export OME-TIFF…")
        self.Bexport_image.setToolTip(
            "Rasterise the reconstruction at a pixel size you choose and save "
            "it with that calibration written into the file. The image is "
            "never downsampled to fit, and it contains every localisation your "
            "filters left active."
        )
        self.Bexport_image.clicked.connect(self.export_image)
        self.data_controls_tab_layout.addWidget(self.Bexport_image, 14, 2, 1, 2)

        self.HL2 = QHSeperationLine()
        self.data_controls_tab_layout.addWidget(self.HL2, 15, 0, 1, 4)

        self.Cscalebar = QCheckBox()
        self.Cscalebar.stateChanged.connect(self.scalebar_state_changed)
        self.Cscalebar.setText("Scalebar")
        self.data_controls_tab_layout.addWidget(self.Cscalebar, 16, 0, 1, 1)

        self.Bz_color_coding = QCheckBox()
        self.Bz_color_coding.setText("Activate Rainbow colorcoding in Z")
        self.Bz_color_coding.stateChanged.connect(self.colorcoding)
        self.data_controls_tab_layout.addWidget(self.Bz_color_coding, 16, 2, 1, 2)

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText("Size of Scalebar [nm]:")
        self.data_controls_tab_layout.addWidget(self.Lscalebarsize, 17, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setValidator(QDoubleValidator(0.001, 1e9, 3, self))
        self.Esbsize.setText("500")
        self.Esbsize.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_sbscale)
        )
        self.data_controls_tab_layout.addWidget(self.Esbsize, 17, 1, 1, 1)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self._sync_scalebar_config)

        # visual_controls
        self.channel_controls_widget_layout = QFormLayout()
        self.channel_controls_placeholder = QWidget()
        self.data_controls_tab_layout.addWidget(
            self.channel_controls_placeholder, 20, 0, 1, 4
        )
        self.channel_controls_placeholder.setLayout(self.channel_controls_widget_layout)

        self.Lcolor_encoding_bar = ZColorCodingColorBarWidget()
        self.Lcolor_encoding_bar.hide()

        self.data_controls_tab_layout.addWidget(self.Lcolor_encoding_bar, 21, 0, 1, 4)

        # infos tab
        self.infos_tab_layout = QGridLayout()
        self.infos_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.Lnumberoflocs = DatasetInfoPanel(self.localization_datasets, parent=self)
        self.infos_tab_layout.addWidget(self.Lnumberoflocs, 0, 0)

        # Decorators tab
        self.decorator_tab_layout = QFormLayout()

        self.Lgrid_plane = QLabel()
        self.Lgrid_plane.setText("Grid Plane")
        self.Lgrid_plane.setFont(QFont("Arial", 10))
        self.decorator_tab_layout.addRow(self.Lgrid_plane)

        self.Cgrid_plane = QCheckBox()
        self.Cgrid_plane.stateChanged.connect(self.grid_plane)
        self.decorator_tab_layout.addRow("Grid plane activated?", self.Cgrid_plane)

        self.Egrid_line_distance = QLineEdit()
        self.Egrid_line_distance.setValidator(QDoubleValidator(0.001, 1e6, 3, self))
        self.Egrid_line_distance.setText(str(self.grid_plane_line_distance_um))
        self.Egrid_line_distance.textChanged.connect(
            lambda: self._start_typing_timer(self.typing_timer_grid)
        )
        self.decorator_tab_layout.addRow(
            "Grid line distance [µm]:", self.Egrid_line_distance
        )

        self.typing_timer_grid = QtCore.QTimer()
        self.typing_timer_grid.setSingleShot(True)
        self.typing_timer_grid.timeout.connect(self.update_grid_plane_line_distance)

        self.Sgrid_line_thickness = GridPlaneSlider(
            parent=self,
            data_to_layer_interface=self.data_to_layer_itf,
            type_of_slider="line_thickness",
            init_range=(1, 100),
            init_value=50,
        )
        self.decorator_tab_layout.addRow(
            "Grid line thickness:", self.Sgrid_line_thickness
        )

        self.Sgrid_z_pos = GridPlaneSlider(
            parent=self,
            data_to_layer_interface=self.data_to_layer_itf,
            type_of_slider="z_pos",
            init_range=(0, 100),
            init_value=50,
        )
        self.decorator_tab_layout.addRow("Z Pos:", self.Sgrid_z_pos)

        self.Bgrid_plane_color = QComboBox()
        self.Bgrid_plane_color.addItems(standard_colors)
        self.Bgrid_plane_color.currentIndexChanged.connect(self.update_grid_plane_color)
        self.decorator_tab_layout.addRow("Grid line color:", self.Bgrid_plane_color)

        self.Sgrid_plane_opacity = GridPlaneSlider(
            parent=self,
            data_to_layer_interface=self.data_to_layer_itf,
            type_of_slider="opacity",
            init_range=(0, 100),
            init_value=100 * self.grid_plane_opacity,
        )
        self.decorator_tab_layout.addRow(
            "Grid plane opacity:", self.Sgrid_plane_opacity
        )

        self.HL3 = QHSeperationLine()
        self.decorator_tab_layout.addRow(self.HL3)

        self.Lrender_range_box = QLabel()
        self.Lrender_range_box.setText("Render Range Box")
        self.Lrender_range_box.setFont(QFont("Arial", 10))
        self.decorator_tab_layout.addRow(self.Lrender_range_box)

        self.Brender_range_box_color = QComboBox()
        self.Brender_range_box_color.addItems(standard_colors)
        self.Brender_range_box_color.currentIndexChanged.connect(
            self.update_render_range_box_color
        )
        self.decorator_tab_layout.addRow(
            "Render Range Box color:", self.Brender_range_box_color
        )

        self.Srender_range_box_opacity = QSlider()
        self.Srender_range_box_opacity.setOrientation(Qt.Horizontal)
        self.Srender_range_box_opacity.setRange(0, 100)
        self.Srender_range_box_opacity.setSingleStep(1)
        self.Srender_range_box_opacity.setValue(
            int(self.render_range_box_opacity * 100)
        )
        self.Srender_range_box_opacity.valueChanged.connect(
            self.update_render_range_box_opacity
        )
        self.decorator_tab_layout.addRow(
            "Render Range Box opacity:", self.Srender_range_box_opacity
        )

        self.decorator_tab.setLayout(self.decorator_tab_layout)
        self.layout = QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.data_controls_tab_layout.setColumnStretch(0, 4)

        self.data_controls_scroll = QScrollArea(self.data_control_tab)
        self.data_controls_scroll.setObjectName("dataControlsScrollArea")
        self.data_controls_scroll.setWidgetResizable(True)
        self.data_controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self.data_controls_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.data_controls_scroll.setWidget(self.data_controls_content)
        data_control_outer_layout = QVBoxLayout(self.data_control_tab)
        data_control_outer_layout.setContentsMargins(0, 0, 0, 0)
        data_control_outer_layout.addWidget(self.data_controls_scroll)
        self.infos_tab.setLayout(self.infos_tab_layout)

    #### D and D
    def _make_fwhm_validator(self):
        """Validator for the FWHM entry fields.

        Keeps non-numeric text and out-of-range magnitudes out of the render
        path.  An unbounded Gaussian footprint is a GPU-stall risk independent
        of dataset size, and the bare float() parse in update_sigma used to
        raise from inside a Qt slot.
        """
        validator = QDoubleValidator(MIN_FWHM_NM, MAX_FWHM_NM, 3)
        try:
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        except AttributeError:  # pragma: no cover - Qt5 unscoped enum
            validator.setNotation(QDoubleValidator.StandardNotation)
        return validator

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
            u = event.mimeData().urls()
            file = u[0].toString()[8:]
            self.open_localization_data_file_and_get_dataset(
                file_path=file, background=True
            )
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
        self.Bexport_image.hide()
        self.Bsave_scene.hide()
        self.Egrid_line_distance.hide()
        self.Sgrid_line_thickness.hide()
        self.Sgrid_z_pos.hide()
        self.Cgrid_plane.hide()
        self.Bgrid_plane_color.hide()
        self.HL1.hide()
        self.HL2.hide()

    def hide_testing_mode(self):
        if self.testing_mode_enabled:
            self.tabs.addTab(self.test_mode_tab, "Test Mode")

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
        self.Bexport_image.show()
        self.Bsave_scene.show()
        self.Cgrid_plane.show()
        self.tabs.addTab(self.infos_tab, "File Infos")
        self.tabs.addTab(self.decorator_tab, "Decorators")
        self.tabs.addTab(self.datafilter_tab, "Data Filter")
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
            # Flat data gets napari's 2-D canvas, so dragging pans instead of
            # rotating a scene with no depth.  This branch said 3 for years
            # because the line was copied from the 3-D one.
            #
            # It could not simply be changed: reference images were placed at
            # z = 0 while flat localizations sit at z = 1, and 2-D display
            # shows a single slice -- so the switch used to make the overlay
            # vanish.  Reference images are now imported onto the centre of the
            # localizations' depth range, which for flat data *is* their plane,
            # so both are on the slice napari shows.
            self.viewer.dims.ndisplay = 2


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
        raise ParentError("Cannot change parent of existing Widget")

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
            u = event.mimeData().urls()
            file = u[0].toString()[8:]
            self.parent.file_to_data_itf.open_localization_data_file_and_get_dataset(
                file_path=file
            )
        else:
            event.ignore()

    def remove_dataset(self, item):
        _LOG.info(
            "napari-storm: dataset removal from this list is not implemented (%s)",
            item,
        )


class ZColorCodingColorBarWidget(QWidget):
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
        self.layout.addRow(self.colorbar)
        self.layout.addRow(self.label)

        self.setLayout(self.layout)

    def set_pixmap(self, scalebar_pixmap):
        if not self.colorbar_set:
            self.colorbar.setPixmap(scalebar_pixmap.scaledToWidth(194))
            self.colorbar_set = True
