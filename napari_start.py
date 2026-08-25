import napari
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from napari_storm import *

def main():
    from napari_storm.napari_particles._napari_compat import \
        enable_instanced_backend

    if not enable_instanced_backend():
        print(
            "instanced rendering is unavailable: VisPy's 'gl+' backend "
            "could not be selected. Is PyOpenGL installed?"
        )
        return 2
    print("GL backend: gl+ (instancing available)")
    v = napari.Viewer()
    widget = napari_storm(v)
    v.window.qt_viewer.dockLayerControls.setVisible(False)
    v.window.qt_viewer.dockLayerList.setVisible(False)
    v.window.add_dock_widget(widget, area="right", name="napari-STORM")

    napari.run()
    return 0

if __name__ == "__main__":
    main()
