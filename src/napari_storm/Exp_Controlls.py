from qtpy import QtGui
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QComboBox, QListWidget, QWidget
import napari
import numpy as np
from PyQt5.QtCore import Qt



class RightClickPaninng(QWidget):
    def __init__(self,parent,viewer):
        super().__init__()
        self.mouse_down = False
        self.start_x=0
        self.start_y=0
        self.parent=parent
        self.viewer=viewer
        self.right_click_pan()

        self.anker=None
        self.anker_coords=None
        self.anker_faces=None


    def create_anker(self):
        center = self.viewer.camera.center
        h=200
        self.anker_coords=np.asarray([[h,0,0],[0,h/2,h/2],[0,h/2,-h/2],[0,-h/2,h/2],[0,-h/2,-h/2],[-h,0,0]])
        #self.anker_faces=np.asarray([[1,2,3],[1,4,5],[1,3,4],[1,5,2],[2,3,6],[4,5,6],[3,4,6],[5,2,6]])-1
        self.anker_faces=np.asarray([[0,1,2],[0,2,4],[0,4,3],[0,3,1],[5,1,2],[5,2,4],[5,4,3],[5,3,1]])
        verts=np.reshape(self.anker_coords+center,(6,3))
        self.anker = self.viewer.add_surface((verts, self.anker_faces), name='anker', shading='smooth',blending='opaque')

    def move_anker(self):
        center = self.viewer.camera.center
        self.viewer.layers['anker'].data=(np.reshape(self.anker_coords+center,(6,3)),self.anker_faces,np.ones(6))

    def remove_anker(self):
        self.viewer.layers.remove('anker')

    def right_click_pan(self):
        #self.copy_on_mouse_press = self.viewer.window.qt_viewer.on_mouse_press
        #print(self.copy_on_mouse_press)
        self.mouse_down=False
        def our_mouse_press(event=None):
            #print(event.type,QMouseEvent,event.button)
            if event.type == "mouse_press":
                if event.button == 2:
                    self.start_x = event.native.x()
                    self.start_y = event.native.y()
                    self.zoom = self.viewer.camera.zoom
                    self.mouse_down=True
                    self.create_anker()
                else:
                    pass


        def our_mouse_move(event : QtGui.QMouseEvent) -> None:
            if not self.mouse_down:
                return
            event.blocked=True
            #print("mouse move", event.native.x(), event.native.y(), event.native.button())
            self._handle_move(event.native.x(), event.native.y())
            self.start_x = event.native.x()
            self.start_y = event.native.y()
            self.move_anker()



        def our_mouse_release(event=None):
            if event.type == "mouse_release":
                if event.button == 2:
                    if not self.mouse_down:
                        return
                    #print("mouse release", event.native.x(), event.native.y(), event.native.button())
                    self.mouse_down = False
                    self.start_x = event.native.x()
                    self.start_y = event.native.y()
                    self.remove_anker()

        self.viewer.window.qt_viewer.on_mouse_press = our_mouse_press
        self.viewer.window.qt_viewer.on_mouse_move = our_mouse_move
        self.viewer.window.qt_viewer.on_mouse_release = our_mouse_release
    def _handle_move(self, x, y):
        v = napari.current_viewer()
        delta_x = (x - self.start_x)/v.window.qt_viewer.width()
        delta_y = (y - self.start_y)/v.window.qt_viewer.height()
        #print(delta_x,delta_y)
        alpha, beta, gamma = self.viewer.camera.angles
        alpha=alpha/360*2*np.pi
        beta=beta/360*2*np.pi
        gamma=gamma/360*2*np.pi
        rx=delta_x*np.cos(alpha)*np.cos(beta)+delta_y*(-np.sin(alpha))*np.cos(beta)*np.sin(gamma)
        ry=delta_x*np.sin(alpha)*np.cos(beta)*np.sin(gamma)+delta_x*np.cos(alpha)*np.sin(beta)+\
           delta_y*np.cos(alpha)*np.cos(beta)*np.sin(gamma)
        rz=-delta_x*np.sin(alpha)*np.cos(gamma)-delta_y*np.cos(alpha)*np.cos(gamma)
        e = v.layers.extent[1][1]
        rx=rx*e[2]
        ry=ry*e[1]
        rz=rz*e[0]
        z, y, x = self.viewer.camera.center
        #print(rx,ry,rz)
        y -= ry
        x -= rx
        z -= rz
        self.viewer.camera.center = (z, y, x)


def custom_keys_and_scalebar(self):
# Custom Keys : w and s for zoom
        # q and e to switch trough axis
        # a and d to rotate view
        v = napari.current_viewer()
        @v.bind_key('w')
        def fly_ahead(v):
            v.camera.zoom *= 1.1
        @v.bind_key('s')
        def fly_back(v):
            self.viewer.camera.zoom *= 0.9
        @v.bind_key('a')
        def fly_rotate_l(v):
            alpha,beta,gamma=v.camera.angles
            #print(alpha)
            alpha += 30
            if alpha >180:
                alpha -= 360
            self.viewer.camera.angles = (alpha, beta, gamma)
        @v.bind_key('d')
        def fly_rotate_d(v):
            alpha, beta, gamma = v.camera.angles
            alpha -= 30
            if alpha < -180:
                alpha += 360
            self.viewer.camera.angles = (alpha, beta, gamma)
        @v.bind_key('q')
        def fly_rotate_l(v):
            alpha, beta, gamma = v.camera.angles
            beta += 30
            if beta > 90:
                beta -= 180
                gamma -=90
            self.viewer.camera.angles = (alpha, beta, gamma)
        @v.bind_key('e')
        def fly_rotate_d(v):
            alpha, beta, gamma = v.camera.angles
            beta -= 30
            if beta < -90:
                beta += 180
                gamma += 90
            self.viewer.camera.angles = (alpha, beta, gamma)
        @v.bind_key('r')
        def fly_reset(v):
            self.change_camera()
            #v.camera.angles=(0,0,90)
            #v.camera.center = self.list_of_datasets[-1].camera_center[0]
            #v.camera.zoom = self.list_of_datasets[-1].camera_center[1]
        @v.bind_key('Up')
        def translate_up(v):
            for layer in v.layers:
                if layer.name !="scalebar":
                    layer.translate+=[0,-50,0]

        @v.bind_key('Down')
        def translate_down(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 50, 0]

        @v.bind_key('Left')
        def translate_left(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 0, -50]

        @v.bind_key('Right')
        def translate_right(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 0, -50]
        v.scale_bar.visible = True

        #placeholder=RightClickPaninng(parent=self,viewer=v)

class MouseControls(QWidget):  # Experimental Fly through mode
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.mouse_down = False
        self.mode = None
        self.active = False

    def _handle_moveL(self, x, y):

        #self.viewer.camera.zoom *= 1.0025
        delta_x = x - self.start_x
        delta_y = y - self.start_y
        alpha, beta, gamma = self.viewer.camera.angles
        relative_x = delta_x / self.viewer.window.qt_viewer.width() * 50
        relative_y = delta_y / self.viewer.window.qt_viewer.height() * 15
        gamma -= relative_y
        beta -= relative_x
        z, y, x = self.viewer.camera.center
        y += np.cos(2 * 3.14145 * gamma / 360) * self.viewer.window.qt_viewer.height() * 0.05
        x -= np.sin(2 * 3.14145 * beta / 360) * self.viewer.window.qt_viewer.width() * 0.05
        self.viewer.camera.center = (z, y, x)
        self.viewer.camera.angles = (alpha, beta, gamma)

    def _handle_moveR(self, x, y):
        #self.viewer.camera.zoom *= 0.9975
        delta_x = x - self.start_x
        delta_y = y - self.start_y
        alpha, beta, gamma = self.viewer.camera.angles
        relative_x = delta_x / self.viewer.window.qt_viewer.width() * 7.5
        relative_y = delta_y / self.viewer.window.qt_viewer.height() * 7.5
        gamma -= relative_y
        beta -= relative_x
        z, y, x = self.viewer.camera.center
        y -= np.cos(2 * 3.14145 * gamma / 360) * self.viewer.window.qt_viewer.height() * 0.05
        x += np.sin(2 * 3.14145 * beta / 360) * self.viewer.window.qt_viewer.width() * 0.05
        self.viewer.camera.center = (z, y, x)
        # print(alpha,beta,gamma)
        self.viewer.camera.angles = (alpha, beta, gamma)
        # print(self.viewer.camera.center)

    def _activate(self):
            print("Custom controlls active")

            self.copy_on_mouse_press = self.viewer.window.qt_viewer.on_mouse_press
            self.copy_on_mouse_move = self.viewer.window.qt_viewer.on_mouse_move
            self.copy_on_mouse_release = self.viewer.window.qt_viewer.on_mouse_release
            self.copy_on_key_press = self.viewer.window.qt_viewer.keyPressEvent

            def our_key_press(event=None): # not used atm
                z,y,x = self.viewer.camera.center
                alpha,beta,gamma= self.viewer.camera.angles
                #print(event.key())
                if event.key()==87: #W
                    self.viewer.camera.zoom *= 1.1
                elif event.key()==83: #S
                    self.viewer.camera.zoom *= 0.9
                elif  event.key()==65: #A
                    alpha+=2.5
                elif event.key()==68:
                    alpha-=2.5
                self.viewer.camera.center=(z,y,x)
                self.viewer.camera.angles=(alpha,beta,gamma)



            def our_mouse_wheel(event=None):
                #print(event.delta)
                if event.delta[-1]>0:
                    self.viewer.camera.zoom *= 1.1
                else:
                    self.viewer.camera.zoom *= 0.9
                #print(self.viewer.camera.zoom)

            def our_mouse_press(event=None):
                #print("mouse press", event.native.x(), event.native.y(), event.native.button())
                self.mouse_down = True
                self.start_x = event.native.x()
                self.start_y = event.native.y()

                self.current_step = list(self.viewer.dims.current_step)

                self._start_zoom = self.viewer.camera.zoom

            def our_mouse_move(event=None):
                if event.button == Qt.MouseButton.RightButton:
                    if not self.mouse_down:
                        return
                    #print("mouse move", event.native.x(), event.native.y(), event.native.button())
                    self._handle_moveR(event.native.x(), event.native.y())
                else:
                    if not self.mouse_down:
                        return
                    #print("mouse move", event.native.x(), event.native.y(), event.native.button())
                    self._handle_moveL(event.native.x(), event.native.y())

            def our_mouse_release(event=None):
                if event.button == Qt.MouseButton.RightButton:
                    if not self.mouse_down:
                        return
                    #print("mouse release", event.native.x(), event.native.y(), event.native.button())
                    self._handle_moveR(event.native.x(), event.native.y())
                    self.mouse_down = False
                else:
                    if not self.mouse_down:
                        return
                    #print("mouse release", event.native.x(), event.native.y(), event.native.button())
                    self._handle_moveL(event.native.x(), event.native.y())
                    self.mouse_down = False



            self.viewer.window.qt_viewer.on_mouse_wheel = our_mouse_wheel
            self.viewer.window.qt_viewer.on_mouse_press = our_mouse_press
            self.viewer.window.qt_viewer.on_mouse_move = our_mouse_move
            self.viewer.window.qt_viewer.on_mouse_release = our_mouse_release
            #self.viewer.window.qt_viewer.keyPressEvent = our_key_press
            self.viewer.camera.interactive = False
            self.active = True

    def _deactivate(self):
        if not self.active:
            return
        self.viewer.window.qt_viewer.on_mouse_press = self.copy_on_mouse_press
        self.viewer.window.qt_viewer.on_mouse_move = self.copy_on_mouse_move
        self.viewer.window.qt_viewer.on_mouse_release = self.copy_on_mouse_release
        #self.viewer.window.qt_viewer.keyPressEvent = self.copy_on_key_press
        self.viewer.camera.interactive = True
        self.active = False