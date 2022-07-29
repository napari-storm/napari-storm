import napari
from napari_storm import *
if __name__ == "__main__":
    v = napari.Viewer()
    widget = napari_storm(v)

    v.window.qt_viewer.dockLayerControls.setVisible(False)
    v.window.qt_viewer.dockLayerList.setVisible(False)
    v.window.add_dock_widget(widget, area='right', name='napari-STORM')

    napari.run()
