"""``napari-storm``: napari with the napari-storm dock, ready to load data.

Installed as a console command by ``setup.cfg`` (``console_scripts``), so
``napari-storm`` starts the viewer in any terminal where the environment is
active -- an Anaconda Prompt included.  ``python -m napari_storm`` does the
same.  Localization files named on the command line are opened once the
viewer is up: ``napari-storm data.h5``.

The desktop installers freeze this module too (``installer/launcher.py``):
:func:`launch` is what they call, so the installed app and the command start
the same way.
"""

import argparse
import sys
import warnings


def launch(paths=()):
    """Open napari with the napari-storm dock and return ``(viewer, widget)``.

    Selects VisPy's ``gl+`` backend first and prints which renderer the dock
    will use.  Each of *paths* is opened as the Open button would, the first
    replacing the (empty) session and the rest merged into it, one after
    another once the event loop runs.  Does not start the event loop.
    """
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
    from qtpy.QtCore import QTimer

    from napari_storm._dock_widget import napari_storm

    viewer = napari.Viewer()
    widget = napari_storm(viewer)
    qt_viewer = get_qt_viewer(viewer)
    qt_viewer.dockLayerControls.setVisible(False)
    qt_viewer.dockLayerList.setVisible(False)
    viewer.window.add_dock_widget(widget, area="right", name="napari-STORM")

    paths = [str(path) for path in paths]
    if paths:
        # Once the event loop runs, so the progress dialog has a window to
        # sit on.
        QTimer.singleShot(0, lambda: _open_in_turn(widget, paths, merge=False))
    return viewer, widget


def _open_in_turn(widget, paths, merge):
    """Open *paths* one after another, each once the one before has loaded.

    Background loads must not overlap, so the next starts from the previous
    one's ``on_loaded``.  A file that fails or is cancelled ends the sequence:
    the dock has already said why.
    """
    if not paths:
        return
    first, rest = paths[0], paths[1:]
    widget.open_localization_data_file_and_get_dataset(
        file_path=first,
        merge=merge,
        background=True,
        on_loaded=lambda _datasets: _open_in_turn(widget, rest, merge=True),
    )


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="napari-storm",
        description="napari with the napari-storm dock, ready to load data.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="localization files to open; the second and later are merged in",
    )
    # Known arguments only: Qt and macOS may add their own (-psn_..., -style),
    # and a launcher should start rather than refuse them.
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    launch(args.files)

    import napari

    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
