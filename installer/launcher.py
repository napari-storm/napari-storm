"""The script PyInstaller freezes into the napari-storm desktop app.

It starts the app the way the ``napari-storm`` command does: through
:func:`napari_storm.__main__.main`, so files named on the command line open
too.

With the environment variable ``NAPARI_STORM_SMOKE_TEST`` set to a file path
it runs the installer's smoke test instead (``installer/build.py`` sets it):
it starts the same viewer and opens the files named on the command line the
way ``napari-storm FILE`` does, waits for them to load, checks that napari
sees the napari-storm plugin, draws one frame, writes what it found to that
file and exits -- 0 when the instanced (gl+) renderer drew the data, non-zero
otherwise.  A watchdog ends it
if anything hangs, a modal dialog included.  The report goes to a file rather
than the console because the Windows app is built without one: there,
``sys.stdout`` is ``None``.
"""

import multiprocessing
import os
import sys
import threading
import time
import traceback

SMOKE_ENV = "NAPARI_STORM_SMOKE_TEST"
#: What launch() prints when instancing is available; build.py looks for it.
GL_LINE = "GL backend: gl+ (instancing available)"
PASSED = "SMOKE TEST PASSED"
#: Seconds before the smoke test gives up.  Generous: a cold start of a
#: frozen app on a CI runner, with software OpenGL on Windows, is slow.
SMOKE_TIMEOUT = 300


def _log_path():
    """Where a windowed app's output goes when there is no console."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        folder = os.path.join(base, "napari-storm")
    elif sys.platform == "darwin":
        folder = os.path.expanduser("~/Library/Logs/napari-storm")
    else:
        folder = os.path.expanduser("~/.cache/napari-storm")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "napari-storm.log")


def _give_output_a_home():
    """Send stdout and stderr to a log file when the app has no console.

    A windowed Windows build starts with both ``None``: print() then does
    nothing, but libraries that write to ``sys.stderr`` directly fail, and a
    traceback would be lost.  The log is overwritten at each start.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log = open(_log_path(), "w", encoding="utf-8", buffering=1)
    except OSError:
        log = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def _smoke_test(report_path, files):
    report = open(report_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = report

    def _timed_out():
        print(f"SMOKE TEST FAILED: no result after {SMOKE_TIMEOUT} s")
        report.flush()
        os._exit(3)

    watchdog = threading.Timer(SMOKE_TIMEOUT, _timed_out)
    watchdog.daemon = True
    watchdog.start()

    print(f"python {sys.version.split()[0]}, frozen={getattr(sys, 'frozen', False)}")

    import napari
    from napari.utils.notifications import notification_manager
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from napari_storm.__main__ import launch
    from napari_storm.napari_particles._napari_compat import get_qt_viewer

    # The dock reports a file it could not open as a napari notification, not
    # an exception; copy them all into the report, and stop on that one.
    notes = []

    def _noted(notification):
        print(f"notification ({notification.severity}): {notification.message}")
        notes.append(str(notification.message))

    notification_manager.notification_ready.connect(_noted)

    # The command-line path, as a user's `napari-storm FILE` takes it: launch()
    # prints the GL line and opens the files in the background, behind the
    # progress dialog.
    viewer, widget = launch(files)
    wanted = {os.path.abspath(path) for path in files}
    deadline = time.monotonic() + SMOKE_TIMEOUT / 2
    outcome = {"code": 1}
    poll = QTimer()
    poll.setInterval(250)

    def _loaded():
        return wanted <= {
            os.path.abspath(str(getattr(dataset, "file_path", "")))
            for dataset in widget.localization_datasets
        }

    def _poll():
        failed = [note for note in notes if "could not open" in note]
        if not (failed or _loaded() or time.monotonic() > deadline):
            return
        poll.stop()
        try:
            _check(failed)
            print(PASSED)
            outcome["code"] = 0
        except Exception:  # noqa: BLE001 - reported, and the exit code says it
            traceback.print_exc()
            print("SMOKE TEST FAILED")
        finally:
            QApplication.instance().quit()

    def _check(failed):
        if failed:
            raise RuntimeError(f"a file did not open: {failed}")
        if not _loaded():
            raise RuntimeError("the files did not finish loading in time")
        datasets = widget.localization_datasets
        print(f"datasets: {len(datasets)}")
        layer_types = [type(layer).__name__ for layer in viewer.layers]
        print(f"layers: {layer_types}")
        if files and "InstancedParticles" not in layer_types:
            raise RuntimeError("the data was not drawn by the instanced renderer")

        # Plugin discovery: napari finds napari-storm's reader (dropping a
        # file on the viewer) through the distribution's entry points, which
        # a frozen app has only if their metadata was bundled.
        from npe2 import PluginManager

        manager = PluginManager.instance()
        plugins = sorted(manifest.name for manifest in manager.iter_manifests())
        print(f"npe2 plugins: {plugins}")
        if "napari-storm" not in plugins:
            raise RuntimeError("napari does not see the napari-storm plugin")
        for path in files:
            readers = [r.command for r in manager.iter_compatible_readers([path])]
            print(f"readers for {os.path.basename(path)}: {readers}")

        from vispy.gloo import gl

        print(f"vispy GL backend: {gl.current_backend.__name__}")
        canvas = get_qt_viewer(viewer).canvas
        scene_canvas = getattr(canvas, "_scene_canvas", canvas)
        frame = scene_canvas.render()
        print(f"GL_VERSION: {gl.glGetParameter(gl.GL_VERSION)}")
        print(f"GL_RENDERER: {gl.glGetParameter(gl.GL_RENDERER)}")
        lit = int((frame[..., :3].max(axis=-1) > 0).sum())
        print(f"frame: {frame.shape}, {lit} lit pixels")
        if files and lit == 0:
            raise RuntimeError("the rendered frame is blank")

    poll.timeout.connect(_poll)
    poll.start()
    napari.run()
    watchdog.cancel()
    report.flush()
    return outcome["code"]


def run():
    # Harmless when nothing spawns processes, and without it a library that
    # does (joblib's loky, say) would start another copy of the app instead.
    multiprocessing.freeze_support()
    report_path = os.environ.get(SMOKE_ENV)
    if report_path:
        return _smoke_test(report_path, sys.argv[1:])
    _give_output_a_home()
    from napari_storm.__main__ import main

    return main()


if __name__ == "__main__":
    sys.exit(run())
