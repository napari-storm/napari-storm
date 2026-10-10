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
        from qtpy.QtCore import Qt

        from napari_storm._dock_widget import napari_storm

        # Shown without taking keyboard focus: the dock binds plain letter
        # keys (r resets the camera), and whatever you type elsewhere while
        # this runs must not land in the window being photographed.
        self.viewer = napari.Viewer(show=False)
        self.dock = napari_storm(self.viewer)
        qt_viewer = get_qt_viewer(self.viewer)
        qt_viewer.dockLayerControls.setVisible(False)
        qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.add_dock_widget(self.dock, area="right", name="napari-STORM")
        self.window = self.viewer.window._qt_window
        self.window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.window.resize(*WINDOW)
        self.window.show()
        settle()

    # --- loading ---------------------------------------------------------

    def open(self, path, merge=False):
        itf = self.dock.file_to_data_itf
        datasets = itf.open_known_filetype_and_import_dataset(str(path))
        self.dock._apply_loaded_datasets(datasets, merge=merge)
        self._unbind_keys()
        settle()
        return datasets

    def _unbind_keys(self):
        """Drop the camera keys the dock binds on load.

        A key typed while this runs can reach the window even when it was
        shown without focus, and r resets the camera mid-shot.
        """
        for key in ("w", "s", "a", "d", "q", "e", "r", "Up", "Down", "Left", "Right"):
            self.viewer.bind_key(key, None, overwrite=True)

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
        _write(name, _crop(self.canvas(), crop))

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


#: The spectrin sample resolves better than 10 nm in all three axes; drawing
#: it wider than that would misrepresent it.
SPECTRIN_FWHM_NM = 8

#: Share of lit pixels allowed to saturate where Gaussians are about a pixel.
WIDE_LIMIT = 0.15

#: ×Range values tried when exposing an image, from bright to dim.
TOPS = (0.5, 0.7, 1, 1.4, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def spectrin(session, fwhm=SPECTRIN_FWHM_NM, top=1.0):
    session.open(SPECTRIN)
    session.fwhm(fwhm)
    session.contrast(cutoff=0.0, top=top)
    session.view("XY")
    return session


def _crop(image, crop):
    if crop is None:
        return image
    h, w = image.shape[:2]
    t, b, l, r = crop
    return image[int(t * h) : int(b * h), int(l * w) : int(r * w)]


def saturated_fraction(image):
    """Share of the lit pixels that have hit the top of the colormap.

    Colourless pixels are left out: they are napari's scale bar and its
    label, white whatever the contrast.
    """
    rgb = np.asarray(image)[..., :3].astype(int)
    peak = rgb.max(axis=-1)
    coloured = (peak - rgb.min(axis=-1)) > 40
    lit = coloured & (peak > 12)
    return float((peak[lit] >= 250).mean()) if lit.any() else 0.0


def expose(session, crop=None, limit=0.05, cutoff=0.0):
    """Set the brightest ×Range at which at most *limit* of the lit pixels
    in *crop* saturate, and return it -- an exposure, not a guess.

    Zoomed out, each Gaussian is smaller than a pixel and a few dense pixels
    carry the sum; there a strict limit leaves everything else near black,
    so whole-dataset views pass ``WIDE_LIMIT``.
    """
    for top in TOPS:
        session.contrast(cutoff=cutoff, top=top)
        if saturated_fraction(_crop(session.canvas(), crop)) <= limit:
            return top
    return TOPS[-1]


def zoom_on(session, factor, quantile=1.0):
    """Centre the XY view on a region of the canvas, then zoom by *factor*.

    The canvas is cut into tiles the size of the view after zooming, each
    scored by how much of it holds data.  *quantile* picks among the tiles
    of the structure itself -- at least 30 % as full as the fullest, which
    leaves out the scattered background: 1 is the fullest, 0 the sparsest.
    """
    image = session.canvas()[..., :3].sum(axis=2).astype(float)
    h, w = image.shape
    block = max(8, int(min(h, w) / factor))
    trimmed = image[: h // block * block, : w // block * block] > 30
    tiles = trimmed.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    filled = np.flatnonzero(tiles.ravel() > 0.3 * tiles.max())
    ranked = filled[np.argsort(tiles.ravel()[filled])]
    pick = ranked[min(len(ranked) - 1, int(quantile * (len(ranked) - 1)))]
    row, col = np.unravel_index(pick, tiles.shape)
    # Centre on the data within the tile, not the tile: a partly filled one
    # would otherwise leave its structure in a corner of the view.
    window = trimmed[row * block : (row + 1) * block, col * block : (col + 1) * block]
    py, px = _centroid(window.astype(float)) + (row * block, col * block)
    camera = session.viewer.camera
    z, y, x = camera.center
    # How far, and which way, content moves on screen per world unit along
    # y and x: measured rather than assumed, since the camera's angles flip
    # the screen axes against the world's.
    step = (
        0.02
        * w
        / camera.zoom
        / (w / session.viewer.window._qt_viewer.canvas.native.width())
    )
    before = _centroid(image)
    camera.center = (z, y + step, x)
    moved_y = _centroid(session.canvas()[..., :3].sum(axis=2).astype(float)) - before
    camera.center = (z, y, x + step)
    moved_x = _centroid(session.canvas()[..., :3].sum(axis=2).astype(float)) - before
    # Solve for the world shift that brings the tile to the centre.
    jacobian = (
        np.column_stack((moved_y, moved_x)) / step
    )  # screen (r, c) per world (y, x)
    wanted = np.array((h / 2 - py, w / 2 - px))
    dy, dx = np.linalg.solve(jacobian, wanted)
    camera.center = (z, y + dy, x + dx)
    camera.zoom = camera.zoom * factor
    settle()


def _centroid(image):
    """(row, column) of the image's brightness."""
    total = image.sum()
    rows, cols = np.indices(image.shape)
    return np.array(((rows * image).sum() / total, (cols * image).sum() / total))


def zoom_on_densest(session, factor):
    zoom_on(session, factor, quantile=1.0)


def depth_coloured(session, opacity):
    """Rainbow depth colouring, every channel at *opacity* percent."""
    session.dock.Bz_color_coding.setChecked(True)
    settle()
    for controls in session.dock.channel:
        controls.Slider_opacity.setValue(opacity)
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


#: Crop of a zoomed canvas kept for the side-by-side comparisons.
DETAIL = (0.2, 0.8, 0.2, 0.8)

#: Opacity of each channel under rainbow depth colouring: dense, overlapping
#: data reads as colour rather than white at full opacity.
DEPTH_OPACITY = 35


def shot_overview(session):
    spectrin(session)
    expose(session, limit=WIDE_LIMIT)
    session.tab("Data Controls")
    session.save_window("overview")


def shot_tabs(session):
    spectrin(session)
    expose(session, limit=WIDE_LIMIT)
    for label, name in [
        ("Data Controls", "tab-data-controls"),
        ("File Infos", "tab-file-infos"),
        ("Decorators", "tab-decorators"),
        ("Data Filter", "tab-data-filter"),
        ("Data adjustment", "tab-data-adjustment"),
    ]:
        session.save_tab(label, name)


def shot_detail(session):
    """Getting started's close-up: the spectrin rings at the data's own
    resolution."""
    spectrin(session)
    zoom_on_densest(session, 8)
    expose(session, crop=DETAIL)
    session.save_canvas("spectrin-detail", crop=DETAIL)


def shot_contrast(session):
    spectrin(session)
    zoom_on_densest(session, 4)
    # Exposed; then the same with full brightness 2.5 times further up, so
    # sparse localizations fade; then exposed again with a cutoff that drops
    # the lone localizations between the bands.
    top = expose(session, crop=DETAIL)
    for cutoff, scale, name in [
        (0.0, 1, "contrast-exposed"),
        (0.0, 2.5, "contrast-dim"),
        (1.2, 1, "contrast-cutoff"),
    ]:
        session.contrast(cutoff=cutoff, top=top * scale)
        session.save_canvas(name, crop=DETAIL)


def shot_fwhm(session):
    spectrin(session)
    # Close enough that a pixel is under 2 nm, or 2 and 8 nm look the same.
    zoom_on_densest(session, 12)
    # As bright as they go for the narrow widths, whose Gaussians are a
    # pixel or two; strictly exposed for the wide one, which saturates first.
    for fwhm, limit in ((2, WIDE_LIMIT), (8, WIDE_LIMIT), (20, 0.02)):
        session.fwhm(fwhm)
        expose(session, crop=DETAIL, limit=limit)
        session.save_canvas(f"fwhm-{fwhm}", crop=DETAIL)


def shot_views(session):
    spectrin(session)
    depth_coloured(session, DEPTH_OPACITY)
    session.save_canvas("view-xy")
    for plane in ("XZ", "YZ"):
        # The view buttons restore the dock's own zoom, so zoom afterwards.
        session.view(plane)
        session.viewer.camera.zoom *= 5
        settle()
        session.save_canvas(f"view-{plane.lower()}", crop=(0.3, 0.7, 0.1, 0.9))


def shot_styles(session):
    from napari_storm.core import PALETTE

    spectrin(session)
    # A sparse region rather than the densest, so single localizations
    # stand apart, about 1.1 um across.
    zoom_on(session, 20, quantile=0.3)
    crop = (0.15, 0.85, 0.15, 0.85)
    # At 8 nm a marker is a few pixels across; the dock's own floor keeps
    # each style's shape readable.  The scientific Gaussian ignores it.
    session.dock.Sfootprint_min_size.setValue(10)
    combo = session.dock.Bfootprint
    for footprint in PALETTE:
        combo.setCurrentIndex(combo.findData(footprint.name))
        settle()
        expose(session, crop=crop)
        session.save_canvas(f"style-{footprint.name}", crop=crop)


def shot_grid(session):
    spectrin(session)
    expose(session, limit=WIDE_LIMIT)
    session.dock.Cgrid_plane.setChecked(True)
    settle()
    opacity = session.dock.Sgrid_plane_opacity
    opacity.setValue(round(opacity.value() * 0.65))
    # The top view, tilted 25 degrees so the plane recedes like a table top.
    alpha, beta, gamma = session.viewer.camera.angles
    session.viewer.camera.angles = (alpha, beta, gamma + 25)
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
    "tabs": shot_tabs,
    "detail": shot_detail,
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
