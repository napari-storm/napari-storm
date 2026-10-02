"""``napari-storm``: napari with the napari-storm dock, ready to load data.

Installed as a console command by ``setup.cfg`` (``console_scripts``), so
``napari-storm`` starts the viewer in any terminal where the environment is
active -- an Anaconda Prompt included.  ``python -m napari_storm`` does the
same.
"""

import sys
import warnings


def main():
    # Before napari is imported, deliberately: importing it is what emits the
    # FutureWarnings being silenced.
    warnings.simplefilter(action="ignore", category=FutureWarning)

    from napari_storm.napari_particles._napari_compat import (
        enable_instanced_backend,
        get_qt_viewer,
    )

    # Before any GL context exists, which means before the viewer.  Without
    # instancing the dock falls back to the billboard renderer and says so;
    # a command a user types should start either way.
    if enable_instanced_backend():
        print("GL backend: gl+ (instancing available)")
    else:
        print(
            "Instanced rendering is unavailable: VisPy's 'gl+' backend could "
            "not be selected. Is PyOpenGL installed? Continuing with the "
            "slower billboard renderer."
        )

    import napari

    from napari_storm._dock_widget import napari_storm

    viewer = napari.Viewer()
    widget = napari_storm(viewer)
    qt_viewer = get_qt_viewer(viewer)
    qt_viewer.dockLayerControls.setVisible(False)
    qt_viewer.dockLayerList.setVisible(False)
    viewer.window.add_dock_widget(widget, area="right", name="napari-STORM")

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
