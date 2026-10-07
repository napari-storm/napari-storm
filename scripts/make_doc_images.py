"""Regenerate the screenshots in docs/images from the files in sample_data.

    python scripts/make_doc_images.py            # every image
    python scripts/make_doc_images.py hero tabs  # only the named shots

Opens a real napari window -- the canvas has to be drawn by the GPU to be
captured -- drives the dock the way a user would, and saves each shot.  Run
it again whenever the interface changes, so the documentation shows the
plugin as it is rather than as it was.

The MINFLUX trace images use synthetic tracks generated here: sample_data
has no tracking measurement, and the point is to show what the overlay does.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
SPECTRIN = ROOT / "sample_data" / "beta_ii_spectrin.storm_4pi.h5"

WINDOW = (1500, 950)


def _write(name, image, shrink=2):
    """Save opaque RGB, *shrink* times smaller: captures are at device
    pixels, twice what a docs page shows on a Retina screen."""
    import imageio.v3 as iio

    image = np.asarray(image)[..., :3].astype(np.float32)
    if shrink > 1:
        h, w = (d // shrink * shrink for d in image.shape[:2])
        image = image[:h, :w].reshape(h // shrink, shrink, w // shrink, shrink, 3)
        image = image.mean(axis=(1, 3))
    iio.imwrite(OUT / f"{name}.png", image.round().astype(np.uint8))
    print("wrote", name)


def settle(n=30):
    from qtpy.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


class Session:
    """One napari window with the dock, as ``napari-storm`` opens it."""

    def __init__(self):
        from napari_storm.napari_particles._napari_compat import (
            enable_instanced_backend,
            get_qt_viewer,
        )

        enable_instanced_backend()
        import napari

        from napari_storm._dock_widget import napari_storm

        self.viewer = napari.Viewer()
        self.dock = napari_storm(self.viewer)
        qt_viewer = get_qt_viewer(self.viewer)
        qt_viewer.dockLayerControls.setVisible(False)
        qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.add_dock_widget(self.dock, area="right", name="napari-STORM")
        self.window = self.viewer.window._qt_window
        self.window.resize(*WINDOW)
        settle()

    # --- loading ---------------------------------------------------------

    def open(self, path, merge=False):
        itf = self.dock.file_to_data_itf
        datasets = itf.open_known_filetype_and_import_dataset(str(path))
        self.dock._apply_loaded_datasets(datasets, merge=merge)
        settle()
        return datasets

    def add(self, datasets, merge=False):
        self.dock._apply_loaded_datasets(list(datasets), merge=merge)
        settle()

    # --- settings --------------------------------------------------------

    def fwhm(self, xy, z=None):
        self.dock.Esigma_xy.setText(f"{xy:g}")
        self.dock.Esigma_z.setText(f"{(z if z is not None else xy):g}")
        self.dock.update_sigma()
        settle()

    def contrast(self, cutoff=None, top=None, channel=0):
        controls = self.dock.channel[channel]
        if cutoff is not None:
            controls.cutoff_spin.setValue(cutoff)
        if top is not None:
            controls.factor_spin.setValue(top)
        settle()

    def colormap(self, name, channel=0):
        self.dock.channel[channel].Colormap_selector.setCurrentText(name)
        settle()

    def view(self, plane, zoom=None, center=None):
        self.dock.change_camera(plane)
        camera = self.viewer.camera
        if center is not None:
            camera.center = center
        if zoom is not None:
            camera.zoom = zoom
        settle()

    def tab(self, label):
        tabs = self.dock.tabs
        for i in range(tabs.count()):
            if tabs.tabText(i) == label:
                tabs.setCurrentIndex(i)
                settle()
                return
        raise KeyError(label)

    # --- capture ---------------------------------------------------------

    def canvas(self):
        """What the canvas draws, as RGBA.

        Not ``viewer.screenshot``: its offscreen path comes back empty for
        these layers, and a grab of the window composites the GL surface
        through its alpha, which turns every additive Gaussian into a ring.
        ``render()`` on the scene canvas is what the pixel tests trust too.
        """
        settle()
        canvas = self.viewer.window._qt_viewer.canvas._scene_canvas
        return np.asarray(canvas.render())

    def save_window(self, name):
        """The whole window, with the real canvas render in place."""
        from qtpy.QtCore import QPoint

        settle()
        window = np.array(self.viewer.screenshot(canvas_only=False, flash=False))
        canvas = self.canvas()
        native = self.viewer.window._qt_viewer.canvas.native
        ratio = window.shape[1] / self.window.width()
        corner = native.mapTo(self.window, QPoint(0, 0))
        top, left = round(corner.y() * ratio), round(corner.x() * ratio)
        height = min(canvas.shape[0], window.shape[0] - top)
        width = min(canvas.shape[1], window.shape[1] - left)
        window[top : top + height, left : left + width] = canvas[:height, :width]
        _write(name, window)

    def save_canvas(self, name, crop=None):
        """The canvas alone; *crop* is (top, bottom, left, right) fractions."""
        image = self.canvas()
        if crop is not None:
            h, w = image.shape[:2]
            t, b, l, r = crop
            image = image[int(t * h) : int(b * h), int(l * w) : int(r * w)]
        _write(name, image)

    def save_widget(self, widget, name):
        settle()
        widget.grab().save(str(OUT / f"{name}.png"))
        print("wrote", name)

    def save_dialog(self, dialog, name):
        dialog.show()
        settle()
        self.save_widget(dialog, name)
        dialog.close()

    def save_tab(self, label, name):
        """The tab's whole content, not the part its scroll area shows."""
        self.tab(label)
        page = self.dock.tabs.currentWidget()
        from qtpy.QtWidgets import QScrollArea

        scroll = page if isinstance(page, QScrollArea) else page.findChild(QScrollArea)
        content = scroll.widget() if scroll is not None else page
        content.adjustSize()
        self.save_widget(content, name)

    def close(self):
        self.viewer.close()
        settle()


# --- the shots ---------------------------------------------------------------


def spectrin(session, fwhm=40, top=4.0):
    session.open(SPECTRIN)
    session.fwhm(fwhm)
    session.contrast(cutoff=0.0, top=top)
    session.view("XY")
    return session


def zoom_on_densest(session, factor):
    """Centre the XY view on the densest part of the canvas, then zoom."""
    image = session.canvas()[..., :3].sum(axis=2).astype(float)
    h, w = image.shape
    block = max(h, w) // 12
    trimmed = image[: h // block * block, : w // block * block]
    tiles = trimmed.reshape(h // block, block, w // block, block).sum(axis=(1, 3))
    row, col = np.unravel_index(np.argmax(tiles), tiles.shape)
    py, px = (row + 0.5) * block, (col + 0.5) * block
    camera = session.viewer.camera
    ratio = w / session.viewer.window._qt_viewer.canvas.native.width()
    scale = camera.zoom * ratio  # physical pixels per world unit
    z, y, x = camera.center
    camera.center = (z, y + (py - h / 2) / scale, x + (px - w / 2) / scale)
    camera.zoom = camera.zoom * factor
    settle()


def minflux_tracks(directory, n_traces=14, n_cycles=160, seed=3):
    """A synthetic MINFLUX tracking run in the Imspector >= 24.10 layout.

    Random walks with a little drift, interleaved in time as concurrent
    traces are in a real export, written as .npy and read by the real reader.
    """
    template = np.load(ROOT / "sample_data" / "imspector_v2.npy")
    rng = np.random.default_rng(seed)
    rows = n_traces * n_cycles
    a = np.zeros(rows, dtype=template.dtype)
    cycle = np.repeat(np.arange(n_cycles), n_traces)
    trace = np.tile(np.arange(n_traces), n_cycles)
    starts = rng.uniform(0, 3000, size=(n_traces, 3)) * [1, 1, 0.1]
    drift = rng.normal(0, 4, size=(n_traces, 3)) * [1, 1, 0.2]
    steps = rng.normal(0, 18, size=(n_cycles, n_traces, 3)) * [1, 1, 0.3]
    paths = starts + np.cumsum(steps + drift, axis=0)  # cycle, trace, axis
    a["vld"] = True
    a["fnl"] = True
    a["bot"] = cycle == 0
    a["eot"] = cycle == n_cycles - 1
    a["itr"] = 4
    a["tid"] = trace + 1
    a["tim"] = np.arange(rows) * 2e-3
    a["loc"] = paths[cycle, trace] * 1e-9
    a["lnc"] = a["loc"]
    a["efo"] = rng.gamma(4, 20000, rows)
    a["cfr"] = rng.uniform(0, 0.8, rows)
    a["eco"] = rng.poisson(120, rows)
    path = Path(directory) / "minflux_tracking.npy"
    np.save(path, a)
    return path


def shot_overview(session):
    spectrin(session)
    session.tab("Data Controls")
    session.save_window("overview")


def shot_hero(session):
    spectrin(session)
    session.dock.Bz_color_coding.setChecked(True)
    settle()
    session.save_canvas("hero")


def shot_tabs(session):
    spectrin(session)
    for label, name in [
        ("Data Controls", "tab-data-controls"),
        ("File Infos", "tab-file-infos"),
        ("Decorators", "tab-decorators"),
        ("Data Filter", "tab-data-filter"),
        ("Data adjustment", "tab-data-adjustment"),
    ]:
        session.save_tab(label, name)


def shot_contrast(session):
    spectrin(session)
    zoom_on_densest(session, 2.5)
    # The default, then the top raised so dense bands resolve, then a cutoff
    # on top of that to drop the lone localizations between them.
    for cutoff, top, name in [
        (0.0, 1.0, "contrast-default"),
        (0.0, 10.0, "contrast-top"),
        (2.0, 10.0, "contrast-cutoff"),
    ]:
        session.contrast(cutoff=cutoff, top=top)
        session.save_canvas(name, crop=(0.15, 0.85, 0.15, 0.85))


def shot_fwhm(session):
    spectrin(session, top=10.0)
    zoom_on_densest(session, 4)
    for fwhm in (15, 40, 100):
        session.fwhm(fwhm)
        session.save_canvas(f"fwhm-{fwhm}", crop=(0.2, 0.8, 0.2, 0.8))


def shot_views(session):
    spectrin(session)
    session.dock.Bz_color_coding.setChecked(True)
    settle()
    session.save_canvas("view-xy")
    for plane in ("XZ", "YZ"):
        # The view buttons restore the dock's own zoom, so zoom afterwards.
        session.view(plane)
        zoom_on_densest(session, 6)
        session.save_canvas(f"view-{plane.lower()}", crop=(0.25, 0.75, 0.0, 1.0))


def shot_styles(session):
    from napari_storm.core import PALETTE

    # A high top, so that the densest region is found among unsaturated
    # pixels rather than at the first of many saturated ones.
    spectrin(session, top=15.0)
    zoom_on_densest(session, 20)
    combo = session.dock.Bfootprint
    for footprint in PALETTE:
        combo.setCurrentIndex(combo.findData(footprint.name))
        # Opaque markers keep their own contrast, at its default.  The additive
        # ones sum, and this is the densest region, so they get a high top --
        # all but glow, which is faint by design and would vanish under it.
        if footprint.blend == "additive":
            session.contrast(top=2.0 if footprint.name == "glow" else 15.0)
        settle()
        session.save_canvas(f"style-{footprint.name}", crop=(0.25, 0.75, 0.25, 0.75))


def shot_grid(session):
    spectrin(session)
    session.dock.Cgrid_plane.setChecked(True)
    settle()
    session.viewer.camera.angles = (-20, 35, 125)
    settle()
    session.save_canvas("grid-plane")


def shot_export(session):
    from napari_storm.image_export import plan_from_widget
    from napari_storm.pyqt.export_dialog import ExportImageDialog

    spectrin(session)
    dialog = ExportImageDialog(
        lambda chosen: plan_from_widget(session.dock, chosen), parent=session.dock
    )
    session.save_dialog(dialog, "export-dialog")


def shot_traces(session):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        session.open(minflux_tracks(directory))
    session.fwhm(20)
    session.view("XY")
    session.save_canvas("traces-off")
    session.dock.Ctraces.setChecked(True)
    settle()
    combo = session.dock.Btrace_color_by
    for label, name in [
        ("Trace", "traces-by-trace"),
        ("Progress along trace", "traces-by-progress"),
    ]:
        combo.setCurrentText(label)
        settle()
        session.save_canvas(name)
    session.save_tab("Decorators", "tab-decorators-traces")


SHOTS = {
    "overview": shot_overview,
    "hero": shot_hero,
    "tabs": shot_tabs,
    "contrast": shot_contrast,
    "fwhm": shot_fwhm,
    "views": shot_views,
    "styles": shot_styles,
    "grid": shot_grid,
    "export": shot_export,
    "traces": shot_traces,
}


def main(names):
    from qtpy.QtWidgets import QApplication  # noqa: F401  (napari makes the app)

    OUT.mkdir(parents=True, exist_ok=True)
    unknown = [n for n in names if n not in SHOTS]
    if unknown:
        sys.exit(f"unknown shot(s): {', '.join(unknown)}; known: {', '.join(SHOTS)}")
    for name in names or SHOTS:
        session = Session()
        try:
            SHOTS[name](session)
        finally:
            session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
