import napari
from napari_storm import *
if __name__ == "__main__":
    v = napari.Viewer()
    widget=SMLMQW(v)
    #widget2=STORM_CON(v)
    print(widget)
    v.window.qt_viewer.dockLayerControls.setVisible(False)
    v.window.add_dock_widget(widget,area='left',name='napari-STORM')
    #v.window.add_dock_widget(widget2,area='left',name='STORM Controls')


    #for wid in v.window():
    #    print(wid)
    napari.run()