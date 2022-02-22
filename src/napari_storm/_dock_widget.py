from PyQt5.QtGui import QPaintEvent, QPainter, QPalette, QBrush, QMouseEvent, QPixmap, QIcon, QGradient, \
    QLinearGradient, QColor
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QComboBox, QListWidget, QWidget
from PyQt5.QtWidgets import QGridLayout, QStyleOptionSlider, QSlider, QSizePolicy, QStyle, QApplication, QLineEdit, \
    QCheckBox, QTabWidget, QFormLayout
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QRect, QSize
import easygui
from .Exp_Controlls import *
import napari
from .utils import generate_billboards_2d

class dataset():
    """An Object where the localisation data is stored, updated and things like Sigma get calculated"""
    def __init__(self, locs=None, zdim=False, parent=None, name=None, pixelsize=130,offset=None,index=0):
        self.zdim = zdim
        self.locs = locs
        self.locs_backup = locs  # Needed if dataset is cut with sliders and then you want the data back
        self.coords=None
        self.index= index
        self.layer = None
        self.name = name
        self.sigma = None
        self.size = None
        self.values = None
        self.pixelsize = pixelsize
        self.parent = parent
        self.camera_center = None
        self.colormap = None
        self.aas = 0
        self.offset = offset
        self.check_offset()
        self.calc_sigmas()
        self.calc_values()

    def check_offset(self):
        """First check which dataset needs the highest offset and then adjust every dataset"""
        if len(self.parent.list_of_datasets)!=0:
            for i in range(len(self.parent.list_of_datasets)):
                for j in range(len(self.parent.list_of_datasets[i].offset)):
                    if self.parent.list_of_datasets[i].offset[j]>self.offset[j]:
                        self.offset[j]=self.parent.list_of_datasets[i].offset[j]
            for i in range(len(self.parent.list_of_datasets)):
                for j in range(len(self.parent.list_of_datasets[i].offset)):
                    self.parent.list_of_datasets[i].locs.x -= self.offset[0]
                    self.parent.list_of_datasets[i].locs.y -= self.offset[1]
                    if self.zdim:
                        self.parent.list_of_datasets[i].locs.z -= self.offset[2]
        self.locs_backup.x -= self.offset[0]
        self.locs_backup.y -= self.offset[1]
        if self.zdim:
            self.locs_backup.z -= self.offset[2]

    def calc_values(self):
        """Update Values which are mapped to colormap"""
        if self.zdim and self.parent.Bspecial_colorcoding.isChecked():
            if self.parent.Brenderoptions.currentText()=="variable gaussian":
                self.values=self.locs.photons**(1/3)
                self.values/=np.max(self.values)
                self.parent.list_of_datasets[self.index].contrast_limits=(0,1)
            else:
                self.values=(self.locs.z-np.min(self.locs.z))/(np.max(self.locs.z)-np.min(self.locs.z))
                #self.values=1/(self.locs.z+1)
                #self.values /= np.max(self.values)
                self.parent.list_of_datasets[self.index].contrast_limits = (0, 1)
            if np.min(self.values)==np.max(self.values):
                self.values=1
            else:
                self.values=(self.values-np.min(self.values))/(max(self.values)-np.min(self.values))
        else:
            self.values=1

    def calc_sigmas(self):
        """Update sigmas"""
        if self.parent.Brenderoptions.currentText() == "variable gaussian":
            sigma = float(self.parent.Esigma.text()) / np.sqrt(self.locs.photons / 10) / 2.354
            sigmaz = float(self.parent.Esigma2.text()) / np.sqrt(self.locs.photons / 10) / 2.354
            self.size = 5 * max(sigma[:])
            self.sigma = np.swapaxes([sigmaz, sigma, sigma], 0, 1)
            self.sigma = self.sigma / np.max(self.sigma)

        else:
            self.size = 5 * float(self.parent.Esigma.text()) / 2.354
            self.sigma = 1

    def update_locs(self):
        LOCS_DTYPE_2D = [("frame", "f4"), ("x", "f4"), ("y", "f4"), ("photons", "f4")]
        LOCS_DTYPE_3D = [("frame", "f4"), ("x", "f4"), ("y", "f4"), ("z", "f4"), ("photons", "f4")]
        self.locs = self.locs_backup
        x0, x1 = self.parent.Srender_rangey.getRange()  # x and y are swapped in napari
        y0, y1 = self.parent.Srender_rangex.getRange()
        xscale = max(self.locs.x) - min(self.locs.x)
        yscale = max(self.locs.y) - min(self.locs.y)


        x0 = x0 * xscale / 100
        x1 = x1 * xscale / 100
        y0 = y0 * yscale / 100
        y1 = y1 * yscale / 100
        filterer = np.ones(self.locs.x.shape)
        if len(np.unique(self.locs.x))>2:
            filterer[self.locs.x < x0] = np.nan
            filterer[self.locs.x > x1] = np.nan
        if len(np.unique(self.locs.y))>2:
            filterer[self.locs.y < y0] = np.nan
            filterer[self.locs.y > y1] = np.nan
        #print(f"len filt {np.shape(filterer)} len idx {np.shape(self.index)} len locs x {np.shape(self.locs.x)}")
        if self.zdim:
            z0, z1 = self.parent.Srender_rangez.getRange()
            zscale = max(self.locs.z) - min(self.locs.z)
            z0 = z0 * zscale / 100
            z1 = z1 * zscale / 100
            if len(np.unique(self.locs.z))>2:
                filterer[self.locs.z < z0] = np.nan
                filterer[self.locs.z > z1] = np.nan
            self.locs = np.rec.array((self.locs.frame[~ np.isnan(filterer)], self.locs.x[~ np.isnan(filterer)],
                                      self.locs.y[~ np.isnan(filterer)], self.locs.z[~ np.isnan(filterer)],
                                      self.locs.photons[~ np.isnan(filterer)]), dtype=LOCS_DTYPE_3D)
        else:
            self.locs = np.rec.array((self.locs.frame[~ np.isnan(filterer)], self.locs.x[~ np.isnan(filterer)],
                                      self.locs.y[~ np.isnan(filterer)],
                                      self.locs.photons[~ np.isnan(filterer)]), dtype=LOCS_DTYPE_2D)
        #print("Filtering done",f"len filt {np.shape(filterer)} len idx {np.shape(self.index)} len locs x {np.shape(self.locs.x)}")
        self.calc_sigmas()
        self.calc_values()


class ChannelControls(QWidget):
    """A QT widget that is created for every channel, which provides the visual controls"""
    def __init__(self,parent,name,idx):
        super().__init__()
        self.parent=parent
        self.name=name
        self.idx=idx # Number of channel corresponds to number of dataset
        self.Label = QLabel()
        self.Label.setText(name)

        self.Bhide_channel=QCheckBox()
        self.Bhide_channel.setChecked(True)
        self.Bhide_channel.stateChanged.connect(self.hide_channel)

        self.Slider = QSlider(Qt.Horizontal) # Contrast
        self.Slider.setMinimum(50)
        self.Slider.setMaximum(100)
        self.Slider.setSingleStep(1)
        self.Slider.setValue(100)
        self.Slider.valueChanged.connect(self.adjust_contrast)

        self.Colormap = QComboBox()
        items=[]
        icons=[]
        for cmap in self.parent.colormaps:
            items.append(cmap.name)
            pixmap= QPixmap(20,20)
            color=QColor(cmap.colors[1][0]*255,cmap.colors[1][1]*255,cmap.colors[1][2]*255,cmap.colors[1][3]*255)
            pixmap.fill(color)
            icons.append(QIcon(pixmap))
        self.Colormap.addItems(items)
        for i in range(len(items)):
            self.Colormap.setItemIcon(i,icons[i])
        self.Colormap.setCurrentText(items[idx])
        self.Colormap.currentIndexChanged.connect(self.adjust_cmap)


        self.layout=QGridLayout()
        self.layout.addWidget(self.Label,0,0)
        self.layout.addWidget(self.Bhide_channel,0,1)
        self.layout.addWidget(self.Colormap,1,0,1,2)
        self.layout.addWidget(self.Slider,2,0,1,2)
        self.layout.setColumnStretch(0,2)
        self.setLayout(self.layout)

    def adjust_contrast(self):
        """...adjust contrast limits"""
        if self.parent.Bspecial_colorcoding.isChecked():
            self.parent.list_of_datasets[self.idx].layer.opacity=self.Slider.value()/50-1
        else:
            self.parent.list_of_datasets[self.idx].layer.contrast_limits=(0,149/np.exp(self.Slider.value()/100 * 5))

    def adjust_cmap(self):
        """...adjust colormap"""
        if self.parent.Bspecial_colorcoding.isChecked():
            if self.parent.Brenderoptions.currentText() == "fixed gaussian":
                self.parent.list_of_datasets[self.idx].layer.colormap = 'hsv'
            else:
                self.parent.list_of_datasets[self.idx].layer.colormap = self.parent.colormaps[-1]
        else:
            self.parent.list_of_datasets[self.idx].layer.colormap = self.parent.colormaps[self.Colormap.currentIndex()]

    def hide_channel(self):
        if not self.Bhide_channel.isChecked():
            self.parent.list_of_datasets[self.idx].layer.opacity=0
            self.parent.list_of_datasets[self.idx].layer.contrast_limits = (0, 1E5)
            self.Slider.hide()
            self.Colormap.hide()
        else:
            self.parent.list_of_datasets[self.idx].layer.opacity = 1
            self.parent.list_of_datasets[self.idx].layer.contrast_limits = (0, 1)
            self.Colormap.show()
            self.Slider.show()
            self.Slider.setValue(100)


class napari_storm(QWidget):
    """The Heart of this code: A Dock Widget, but also an object where everthing runs together"""
    def __init__(self, napari_viewer):
        super().__init__()
        ###### D and D
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.data_control_tab = QWidget()
        self.visual_control_tab = QWidget()
        #self.setStyleSheet("background-color: #414851")
        self.tabs.addTab(self.data_control_tab,"Data Controls")
        self.tabs.addTab(self.visual_control_tab,"Visual Controls")

        self.list_of_datasets = []
        self.pixelsize = []
        self.layer = []
        self.layer_names = []
        self.scalebar_layer = []
        self.scalebar_exists = False
        self.zdim = False
        self.channel=[]
        self.colormaps=colormaps()

        self.viewer = napari_viewer
        layout = QGridLayout()
        ######################### data_control_tab
        self.Bopen = QPushButton()
        self.Bopen.clicked.connect(lambda: open_STORM_data(self))
        self.Bopen.setText("Import File Dialog")
        layout.addWidget(self.Bopen, 0, 1,1,4)

        self.Limport = QLabel()
        self.Limport.setText("Import \nSTORM Data")
        layout.addWidget(self.Limport, 0, 0)

        self.Lnumberoflocs = TestListView(self, parent=self)
        self.Lnumberoflocs.addItem("STATISTICS \nWaiting for Data... \nImport or drag file here")
        self.Lnumberoflocs.itemDoubleClicked.connect(self.Lnumberoflocs.remove_dataset)
        layout.addWidget(self.Lnumberoflocs, 1, 0, 1, 4)

        self.Lresetview = QLabel()
        self.Lresetview.setText("Reset view:")
        layout.addWidget(self.Lresetview, 2, 0)
        """
        self.Baxis = QComboBox()
        self.Baxis.addItems(["XY", "XZ", "YZ"])
        self.Baxis.currentIndexChanged.connect(self.change_camera)
        layout.addWidget(self.Baxis, 2, 1)"""
        self.Baxis_xy=QPushButton()
        self.Baxis_xy.setText("XY")
        self.Baxis_xy.clicked.connect(lambda: self.change_camera(type="XY"))
        layout.addWidget(self.Baxis_xy,2,1)

        self.Baxis_yz = QPushButton()
        self.Baxis_yz.setText("YZ")
        self.Baxis_yz.clicked.connect(lambda: self.change_camera(type="YZ"))
        layout.addWidget(self.Baxis_yz, 2, 2)

        self.Baxis_xz = QPushButton()
        self.Baxis_xz.setText("XZ")
        self.Baxis_xz.clicked.connect(lambda: self.change_camera(type="XZ"))
        layout.addWidget(self.Baxis_xz, 2, 3)


        self.Lrenderoptions = QLabel()
        self.Lrenderoptions.setText("Rendering options:")
        layout.addWidget(self.Lrenderoptions, 3, 0)

        self.Brenderoptions = QComboBox()
        self.Brenderoptions.addItems(["fixed gaussian", "variable gaussian"])
        self.Brenderoptions.currentIndexChanged.connect(self.render_options_changed)
        layout.addWidget(self.Brenderoptions, 3, 1,1,3)

        self.Lsigma = QLabel()
        self.Lsigma.setText("PSF FWHM in XY [nm]:")
        layout.addWidget(self.Lsigma, 4, 0)

        self.Lsigma2 = QLabel()
        self.Lsigma2.setText("PSF FWHM in Z [nm]:")
        layout.addWidget(self.Lsigma2, 5, 0)

        self.Esigma = QLineEdit()
        self.Esigma.setText("10")
        self.Esigma.textChanged.connect(lambda: self.start_typing_timer(self.typing_timer_sigma))
        layout.addWidget(self.Esigma, 4, 1,1,3)
        self.typing_timer_sigma = QtCore.QTimer()
        self.typing_timer_sigma.setSingleShot(True)
        self.typing_timer_sigma.timeout.connect(lambda: update_layers(self))

        self.Esigma2 = QLineEdit()
        self.Esigma2.setText("750")
        self.Esigma2.textChanged.connect(lambda: self.start_typing_timer(self.typing_timer_sigma))
        layout.addWidget(self.Esigma2, 5, 1,1,3)

        self.Lrangex = QLabel()
        self.Lrangex.setText("X-range")
        layout.addWidget(self.Lrangex, 6, 0)

        self.Lrangey = QLabel()
        self.Lrangey.setText("Y-range")
        layout.addWidget(self.Lrangey, 7, 0)

        self.Lrangez = QLabel()
        self.Lrangez.setText("Z-range")
        layout.addWidget(self.Lrangez, 8, 0)

        from .RangeSlider2 import RangeSlider2
        self.Srender_rangex = RangeSlider2(parent=self,type='x')
        # self.Srender_rangex.mouseReleaseEvent.connect()
        layout.addWidget(self.Srender_rangex, 6, 1,1,3)

        self.Srender_rangey = RangeSlider2(parent=self,type='y')
        layout.addWidget(self.Srender_rangey, 7, 1,1,3)

        self.Srender_rangez = RangeSlider2(parent=self,type='z')
        layout.addWidget(self.Srender_rangez, 8, 1,1,3)

        self.Lscalebar = QLabel()
        self.Lscalebar.setText("Scalebar?")
        layout.addWidget(self.Lscalebar, 9, 0)

        self.Cscalebar = QCheckBox()
        self.Cscalebar.stateChanged.connect(self.scalebar)
        layout.addWidget(self.Cscalebar, 9, 1,1,3)

        self.Lscalebarsize = QLabel()
        self.Lscalebarsize.setText("Size of Scalebar [nm]:")
        layout.addWidget(self.Lscalebarsize, 10, 0)

        self.Esbsize = QLineEdit()
        self.Esbsize.setText("500")
        self.Esbsize.textChanged.connect(lambda: self.start_typing_timer(self.typing_timer_sbscale))
        layout.addWidget(self.Esbsize, 10, 1,1,3)
        self.typing_timer_sbscale = QtCore.QTimer()
        self.typing_timer_sbscale.setSingleShot(True)
        self.typing_timer_sbscale.timeout.connect(self.scalebar)

        self.Bspecial_colorcoding = QCheckBox()
        self.Bspecial_colorcoding.setText("Activate Rainbow colorcoding in Z")
        self.Bspecial_colorcoding.stateChanged.connect(self.colorcoding)
        layout.addWidget(self.Bspecial_colorcoding,13,0,1,4)

        self.Bmerge_with_additional_file = QPushButton()
        self.Bmerge_with_additional_file.setText("Merge with additional file")
        layout.addWidget(self.Bmerge_with_additional_file,14,0,1,4)
        self.Bmerge_with_additional_file.clicked.connect(lambda: open_STORM_data(self,merge=True))

        ########################################## visual_control_tab
        self.layout2 = QFormLayout()

        self.BAutoContrast = QPushButton()
        self.BAutoContrast.setText("reset contrast")
        self.layout2.addRow(self.BAutoContrast)
        self.BAutoContrast.clicked.connect(self.auto_contrast)

        ##############################################
        self.layout=QGridLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.custom_controlls = MouseControlls()
        layout.setColumnStretch(0, 4)
        self.data_control_tab.setLayout(layout)
        self.visual_control_tab.setLayout(self.layout2)
        self.right_click_pan()
        custom_keys_and_scalebar(self)
        self.hide_stuff()

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
            open_STORM_data(self, file_path=file)
        else:
            event.ignore()
        #####

    def right_click_pan(self):
        self.copy_on_mouse_press = self.viewer.window.qt_viewer.on_mouse_press
        print(self.copy_on_mouse_press)
        self.mouse_down=False
        def our_mouse_press(event=None):
            print(event.type,QMouseEvent,event.button)
            if event.type == "mouse_press":
                print("almost")
                if event.button == 3:
                    self.start_x = event.native.x()
                    self.start_y = event.native.y()
                    self.zoom = self.viewer.camera.zoom
                    self.mouse_down=True
                    print("we're in")
                else:
                    pass
                    #super().mousePressEvent(self,event)

        def our_mouse_move(event=None):
            if not self.mouse_down:
                return
            #print("mouse move", event.native.x(), event.native.y(), event.native.button())
            self._handle_move(event.native.x(), event.native.y())

        def our_mouse_release(event=None):
            print("release")
            if event.type == "mouse_press":
                if event.button == "mouse_press":
                    if not self.mouse_down:
                        return
                    #print("mouse release", event.native.x(), event.native.y(), event.native.button())
                    self._handle_move(event.native.x(), event.native.y())
                    self.mouse_down = False

        self.viewer.window.qt_viewer.on_mouse_press = our_mouse_press
        self.viewer.window.qt_viewer.on_mouse_move = our_mouse_move
        self.viewer.window.qt_viewer.on_mouse_release = our_mouse_release

    def _handle_move(self, x, y):
        delta_x = x - self.start_x
        delta_y = y - self.start_y
        alpha, beta, gamma = self.viewer.camera.angles
        relative_x = delta_x / self.viewer.window.qt_viewer.width() * 7.5
        relative_y = delta_y / self.viewer.window.qt_viewer.height() * 7.5
        gamma -= relative_y
        beta -= relative_x
        z, y, x = self.viewer.camera.center
        y -= np.cos(2 * 3.14145 * gamma / 360) * self.viewer.window.qt_viewer.height()
        x += np.sin(2 * 3.14145 * beta / 360) * self.viewer.window.qt_viewer.width()
        print((z, y, x))
        self.viewer.camera.center = (z, y, x)
        self.viewer.camera.zoom=self.zoom
        # print(alpha,beta,gamma)
        # print(self.viewer.camera.center)


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
        self.Esigma.hide()
        self.Lsigma2.hide()
        self.Esigma2.hide()
        self.Bspecial_colorcoding.hide()
        self.Lscalebarsize.hide()
        self.Esbsize.hide()
        self.Bmerge_with_additional_file.hide()
        self.Srender_rangez.hide()
        self.Lrangez.hide()
        self.Lresetview.hide()
        #self.Baxis.hide()
        self.Baxis_xy.hide()
        self.Baxis_yz.hide()
        self.Baxis_xz.hide()
        self.BAutoContrast.hide()

    def show_stuff(self):
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
        self.Esigma.show()
        self.Lscalebar.show()
        self.Lscalebarsize.show()
        self.Esbsize.show()
        self.BAutoContrast.show()
        self.Bmerge_with_additional_file.show()

    def add_channel(self,name="Channel"):
        """Adds a Channel in the visual controlls"""
        self.channel.append(ChannelControls(parent=self,name=name,idx=len(self.channel)))
        self.layout2.addRow(self.channel[-1])

    def auto_contrast(self):
        """Actually atm just resets the contrast"""
        for i in range(len(self.list_of_datasets)):
            self.channel[i].Slider.setValue(100)

    def colorcoding(self):
        """Check if Colorcoding is choosen"""
        if self.Bspecial_colorcoding.isChecked():
            for i in range(len(self.channel)):
                self.channel[i].Colormap.hide()
        else:
            for i in range(len(self.channel)):
                self.channel[i].Colormap.show()
        update_layers(self)

    def render_options_changed(self):
        if self.Brenderoptions.currentText() == "variable gaussian":
            self.Bspecial_colorcoding.setText("normalize Gaussian Intensity")
            self.Lsigma2.show()
            self.Esigma2.show()
            self.Lsigma.setText("PSF FWHM in XY [nm]")
            self.Esigma.setText("300")
        else:
            self.Bspecial_colorcoding.setText("Activate Rainbow colorcoding in Z")
            self.Lsigma2.hide()
            self.Esigma2.hide()
            self.Lsigma.setText("FWHM in XY [nm]")
            self.Esigma.setText("10")
        update_layers(self)

    def scalebar(self):
        """Creating/Removing/Updating the custom Scalebar in 2 and 3D"""
        v = napari.current_viewer()
        cpos=v.camera.center
        l = int(self.Esbsize.text())
        if self.Cscalebar.isChecked() and not not all(self.list_of_datasets[-1].locs):
            if self.list_of_datasets[-1].zdim:
                list=[l,.125*l/2,.125*l/2]
                faces =np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [0, 2, 4], [4, 6, 2], [1, 3, 7], [1, 5, 7], [2, 3, 6],
                     [3, 6, 7], [4, 5, 0], [0, 1, 5]])
                vertices=[]
                """
                for c in range(3):
                    i=list[c%3]
                    j=list[(c+1)%3]
                    k=list[(c+2)%3]
                    m=0
                    verts = [[-i,-k,-j],[-i,k,-j],[-i,-k,j],[-i,k,j],[i,-k,-j],[i,k,-j],[i,-k,j],[i,k,j]]
                    vertices.append(np.asarray(verts + cpos*np.ones_like(verts)))"""
                vertices=np.asarray([[0,list[1],list[2]],[0,-list[1],list[2]],[0,list[1],-list[2]],[0,-list[1],-list[2]],
                       [l,list[1],list[2]],[ l,-list[1],list[2]],[ l,list[1],-list[2]],[ l,-list[1],-list[2]],
                       [list[1],0,list[2]],[-list[1],0,list[2]],[list[1],0,-list[2]],[-list[1],0,-list[2]],
                       [list[1],l,list[2]],[-list[1],l,list[2]],[list[1],l,-list[2]],[-list[1],l,-list[2]],
                       [list[1],list[2],0],[-list[1],list[2],0],[list[1],-list[2],0],[-list[1],-list[2],0],
                       [list[1],list[2],l],[-list[1],list[2],l],[list[1],-list[2],l],[-list[1],-list[2],l]])
                for i in range(len(vertices)):
                    vertices[i]=vertices[i]+cpos-(l/2,0,0)

                faces=np.asarray(np.vstack((faces,faces+8,faces+16)))
                #vertices=np.reshape(np.asarray(vertices),(24,3))
            else:
                list=[l,0.05*l]

                faces=np.asarray([[0,1,3],[1,2,3],[4,5,7],[5,6,7]])
                verts=[[cpos[1] ,cpos[2]-list[1]],[cpos[1]+list[0],cpos[2]-list[1]],
                       [cpos[1]+list[0],cpos[2]+list[1]],[cpos[1] ,cpos[2]+list[1]],
                       [cpos[1]-list[1],cpos[2] ],[cpos[1]+list[1],cpos[2] ],
                       [cpos[1]+list[1],cpos[2]+list[0]],[cpos[1]-list[1],cpos[2]+list[0]]]
                vertices=np.reshape(np.asarray(verts),(8,2))
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_layer = v.add_surface((vertices,faces),name='scalebar',shading='none')
            else:
                self.scalebar_layer = v.add_surface((vertices,faces),name='scalebar',shading='none')
                self.scalebar_exists = True
        else:
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_exists = False


    def threed(self):
        """3D or 2D Mode"""
        v = napari.current_viewer()
        # print(v.camera.view_direction)
        if self.list_of_datasets[-1].zdim:
            self.Lrangez.show()
            self.Srender_rangez.show()
            #self.Baxis.show()
            self.Baxis_xy.show()
            self.Baxis_xz.show()
            self.Baxis_yz.show()
            self.Lresetview.show()
            self.Bspecial_colorcoding.show()
            v.dims.ndisplay = 3
        else:
            self.Bspecial_colorcoding.show()
            self.Lrangez.hide()
            self.Srender_rangez.hide()
            v.dims.ndisplay = 2

    def start_typing_timer(self, timer):
        timer.start(500)

    def change_camera(self,type="XY"):
        v = napari.current_viewer()
        values= {}
        if type == "XY":
            v.camera.angles=(0,0,90)
        elif type == "XZ":
            v.camera.angles=(0,0,180)
        else:
            v.camera.angles=(-90,-90,-90)
        v.camera.center = self.list_of_datasets[-1].camera_center[0]
        v.camera.zoom =self.list_of_datasets[-1].camera_center[1]
        v.camera.update(values)




@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return napari_storm


import h5py
import numpy as np
import yaml as _yaml
import os.path as _ospath
from tkinter import filedialog as fd
from tkinter import Tk
import napari
import matplotlib.pyplot as plt
from .particles import Particles, BillboardsFilter


##### Highest Order Function
def colormaps():
    """Creating the Custom Colormaps"""
    cmaps=[]
    names=["red","green","blue"]
    for i in range(3):
        colors=np.zeros((2,4))
        colors[-1][i]=1
        colors[-1][-1]=1
        cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors,name=names[i]))
    names = ["yellow", "cyan","pink"]
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i+1)%3] = 1
        colors[-1][-1] = 1
        cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i]))
    names=["orange","mint","purple"]
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i+1)%3] = .5
        colors[-1][-1] = 1
        cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i]))
    colors=np.ones((2, 4))
    cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name="gray"))
    return cmaps


def open_STORM_data(self, file_path=None, merge=False):
    """Deciding which filetype and the calling the corresponding Importer"""
    self.show_stuff()
    if len(self.list_of_datasets)!=0 and merge==False:
        reset(self)
    from .importer import load_hdf5,load_csv,load_SMLM,load_h5,load_mfx_json,load_mfx_npy,start_testing
    v=napari.current_viewer()
    if not self.list_of_datasets:
        self.Lnumberoflocs.clear()
    elif merge == False:
        for i in range(len(self.list_of_datasets)):
            v.layers.remove(self.list_of_datasets[i].name)
        self.list_of_datasets = []
        self.Lnumberoflocs.clear()
    Tk().withdraw()
    if not file_path:
        file_path = fd.askopenfilename()
    filetype = file_path.split(".")[-1]
    filename = file_path.split("/")[-1]
    if filetype == "hdf5":
        load_hdf5(self, file_path)
    elif filetype == "yaml":
        file_path = file_path[:-(len(filetype))] + "hdf5"
        load_hdf5(self, file_path)
    elif filetype == "csv":
        load_csv(self, file_path)
    elif filetype == "smlm":
        load_SMLM(self, file_path)
    elif filetype == "h5":
        load_h5(self, file_path)
    elif filetype == "json":
        load_mfx_json(self,file_path)
    elif filetype == "npy":
        load_mfx_npy(self,file_path)
    elif filetype == "test":
        start_testing(self)
    else:
        self.hide_stuff()
        raise TypeError("unknown File extension for STORM Data files...")

def reset(self):
    """Funktion which is called when a file has aready been opened and a new one ovverrides it"""
    if not len(self.channel) == 0:  # Remove Channel of older files
        for i in range(len(self.channel)):
            self.channel[i].hide()
        self.channel = []
    if self.Bspecial_colorcoding.isChecked():  # Remove Colorcoding
        self.Bspecial_colorcoding.setCheckState(False)
    if self.Cscalebar.isChecked():  # Remove Scalebar
        self.Cscalebar.setCheckState(False)
        self.scalebar()

def create_new_layer(self, aas=0, layer_name="SMLM Data", idx=-1):
    """Creating a Particle Layer """
    if not self.list_of_datasets[idx].zdim: # Hide 3D Options if 2D Dataset
        self.Srender_rangez.hide()
        self.Lrangez.hide()
        #self.Baxis.hide()
        self.Baxis_xy.hide()
        self.Baxis_yz.hide()
        self.Baxis_xz.hide()
        self.Lresetview.hide()
        self.Bspecial_colorcoding.hide()
    else:
        self.Srender_rangez.show()
        self.Lrangez.show()
        #self.Baxis.show()
        self.Baxis_xy.show()
        self.Baxis_yz.show()
        self.Baxis_xz.show()
        self.Lresetview.show()
        self.Bspecial_colorcoding.show()
    self.list_of_datasets[idx].update_locs()
    coords = get_coords_from_locs(self, self.list_of_datasets[idx].pixelsize, idx)
    v = napari.current_viewer()  # Just to get the sigmas
    """print(f"Create:\n #################\n"
          f"size = {self.list_of_datasets[idx].size}\n values = {self.list_of_datasets[idx].values}\n "
          f"sigmas {self.list_of_datasets[idx].sigma}\n Pixelsize {self.list_of_datasets[idx].pixelsize}"
          f"\n coords ={coords}\n##############")"""
    self.list_of_datasets[idx].layer = Particles(coords, size=self.list_of_datasets[idx].size,
                                                 values=self.list_of_datasets[idx].values,
                                                 antialias=self.list_of_datasets[idx].aas,
                                                 colormap=self.colormaps[idx],
                                                 sigmas=self.list_of_datasets[idx].sigma,
                                                 filter=None,
                                                 name=layer_name,
                                                 )
    self.list_of_datasets[idx].name = layer_name
    self.list_of_datasets[idx].layer.add_to_viewer(v)
    # v.window.qt_viewer.layer_to_visual[self.layer[-1]].node.canvas.measure_fps()
    if self.list_of_datasets[idx].zdim:
        v.dims.ndisplay = 3
    else:
        v.dims.ndisplay = 2
    v.camera.perspective = 50
    self.list_of_datasets[idx].layer.shading = 'gaussian'
    show_infos(self, layer_name, idx)
    self.list_of_datasets[idx].camera_center=[v.camera.center,v.camera.zoom, v.camera.angles]
    self.add_channel(name=layer_name)
    self.channel[-1].adjust_cmap()
    self.channel[-1].adjust_contrast()
    #print(len(self.list_of_datasets[-1].index),"idx,locs",len(self.list_of_datasets[-1].locs.x))

def update_layers(self, aas=0,  layer_name="SMLM Data"):
    """Updating a Particle Layer"""
    v = napari.current_viewer()
    self.list_of_datasets[-1].camera = [v.camera.zoom, v.camera.center, v.camera.angles]
    for i in range(len(self.list_of_datasets)):
        self.list_of_datasets[i].update_locs()
        v.layers.remove(self.list_of_datasets[i].name)
        coords = get_coords_from_locs(self, self.list_of_datasets[i].pixelsize, i)
        """print(f"Update:\n #################\n"
          f"size = {self.list_of_datasets[i].size}\n values = {self.list_of_datasets[i].values}\n "
          f"sigmas ={self.list_of_datasets[i].sigma}\n Pixelsize ={self.list_of_datasets[i].pixelsize}"
          f"\n coords ={coords}\n##############")"""
        self.list_of_datasets[i].layer = Particles(coords, size=self.list_of_datasets[i].size,
                                                   values=self.list_of_datasets[i].values,
                                                   antialias=aas,
                                                   sigmas=self.list_of_datasets[i].sigma,
                                                   filter=None,
                                                   name=self.list_of_datasets[i].name,
                                                   )
        self.list_of_datasets[i].layer.add_to_viewer(v)
        self.channel[i].adjust_contrast()
        self.channel[i].adjust_cmap()
        #if np.min(self.list_of_datasets[i].values) != np.max(self.list_of_datasets[i].values):
        #    self.list_of_datasets[i].layer.contrast_limits = (np.min(self.list_of_datasets[i].values),
        #                                                        np.max(self.list_of_datasets[i].values))
        self.list_of_datasets[i].layer.shading = 'gaussian'
    v.camera.angles = self.list_of_datasets[-1].camera[2]
    v.camera.zoom = self.list_of_datasets[-1].camera[0]
    v.camera.center = self.list_of_datasets[-1].camera[1]
    v.camera.update({})

def update_layers2(self):
    """Still doesn't work"""
    v= napari.current_viewer()
    for i in range(len(self.list_of_datasets)):
        self.list_of_datasets[i].update_locs()
        coords = get_coords_from_locs(self, self.list_of_datasets[i].pixelsize, i)
        values = self.list_of_datasets[i].values
        size = self.list_of_datasets[i].size

        coords = np.asarray(coords)
        if np.isscalar(values):
            values = values * np.ones(len(coords))
        values = np.broadcast_to(values, len(coords))
        size = np.broadcast_to(size, len(coords))
        if coords.shape[1] == 2:
            coords = np.concatenate([np.zeros((len(coords),1)), coords], axis=-1)


        vertices, faces, texcoords = generate_billboards_2d(coords, size=size)


        values = np.repeat(values, 4, axis=0)
        vertices_old,faces_old,values_old=v.layers[self.list_of_datasets[i].name].data
        print(f"before: {len(vertices_old),len(faces_old),len(values_old)}")
        print(f"after: {len(vertices), len(faces), len(values)}")
        v.layers[self.list_of_datasets[i].name].data=(vertices,faces,values)


def get_coords_from_locs(self, pixelsize, idx):
    """Calculating Particle Coordinates from Locs"""
    if self.list_of_datasets[idx].zdim:
        num_of_locs = len(self.list_of_datasets[idx].locs.x)
        coords = np.zeros([num_of_locs, 3])
        coords[:, 0] = self.list_of_datasets[idx].locs.z * pixelsize
        coords[:, 1] = self.list_of_datasets[idx].locs.x * pixelsize
        coords[:, 2] = self.list_of_datasets[idx].locs.y * pixelsize
    else:
        num_of_locs = len(self.list_of_datasets[idx].locs.x)
        coords = np.zeros([num_of_locs, 2])
        coords[:, 0] = self.list_of_datasets[idx].locs.x * pixelsize
        coords[:, 1] = self.list_of_datasets[idx].locs.y * pixelsize
    return coords

##### Semi Order Functions
def show_infos(self, filename, idx):
    """Print Infos about files in Log"""
    if self.list_of_datasets[idx].zdim:
        self.Lnumberoflocs.addItem(
            "Statistics\n" + f"File: {filename}\n" + f"Number of locs: {len(self.list_of_datasets[idx].locs.x)}\n"
                                                     f"Imagewidth: {np.round((max(self.list_of_datasets[idx].locs.x) - min(self.list_of_datasets[idx].locs.x)) * self.list_of_datasets[idx].pixelsize / 1000,3)} µm\n" +
            f"Imageheigth: {np.round((max(self.list_of_datasets[idx].locs.y) - min(self.list_of_datasets[idx].locs.y)) * self.list_of_datasets[idx].pixelsize / 1000,3)} µm\n" +
            f"Imagedepth: {np.round((max(self.list_of_datasets[idx].locs.z) - min(self.list_of_datasets[idx].locs.z)) * self.list_of_datasets[idx].pixelsize / 1000,3)} µm\n" +
            f"Intensity per localisation\nmean: {np.round(np.mean(self.list_of_datasets[idx].locs.photons),3)}\nmax: " + f"{np.round(max(self.list_of_datasets[idx].locs.photons),3)}\nmin:" +
            f" {np.round(min(self.list_of_datasets[idx].locs.photons),3)}\n")
    else:
        self.Lnumberoflocs.addItem(
            "Statistics\n" + f"File: {filename}\n" + f"Number of locs: {len(self.list_of_datasets[idx].locs.x)}\n"
                                                     f"Imagewidth: {np.round((max(self.list_of_datasets[idx].locs.x) - min(self.list_of_datasets[idx].locs.x)) * self.list_of_datasets[idx].pixelsize / 1000,3)} µm\n" +
            f"Imageheigth: {np.round((max(self.list_of_datasets[idx].locs.y) - min(self.list_of_datasets[idx].locs.y)) * self.list_of_datasets[idx].pixelsize / 1000,3)} µm\n" +
            f"Intensity per localisation\nmean: {np.round(np.mean(self.list_of_datasets[idx].locs.photons),3)}\nmax: " + f"{np.round(max(self.list_of_datasets[idx].locs.photons),3)}\nmin:" +
            f" {np.round(min(self.list_of_datasets[idx].locs.photons),3)}\n")


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
            open_STORM_data(self.parent, file_path=file)
        else:
            event.ignore()

    def remove_dataset(self, item):
        print("Dataset removal not implemented yet...", item)