import logging
import warnings

import napari
import numpy as np
from qtpy.QtWidgets import QWidget

logger = logging.getLogger(__name__)


def custom_keys_and_scalebar(self):
    # Custom Keys : w and s for zoom
    # q and e to switch trough axis
    # a and d to rotate view
    v = napari.current_viewer()
    try:

        @v.bind_key("w")
        def fly_ahead(v):
            v.camera.zoom *= 1.1

        @v.bind_key("s")
        def fly_back(v):
            self.viewer.camera.zoom *= 0.9

        @v.bind_key("a")
        def fly_rotate_l(v):
            alpha, beta, gamma = v.camera.angles
            alpha += 30
            if alpha > 180:
                alpha -= 360
            self.viewer.camera.angles = (alpha, beta, gamma)

        @v.bind_key("d")
        def fly_rotate_d(v):
            alpha, beta, gamma = v.camera.angles
            alpha -= 30
            if alpha < -180:
                alpha += 360
            self.viewer.camera.angles = (alpha, beta, gamma)

        @v.bind_key("q")
        def fly_rotate(v):
            alpha, beta, gamma = v.camera.angles
            beta = min(beta + 30, 90)
            self.viewer.camera.angles = (alpha, beta, gamma)

        @v.bind_key("e")
        def fly_rotate2(v):
            alpha, beta, gamma = v.camera.angles
            beta = max(beta - 30, -90)
            self.viewer.camera.angles = (alpha, beta, gamma)

        @v.bind_key("r")
        def fly_reset(v):
            self.change_camera()

        @v.bind_key("Up")
        def translate_up(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, -50, 0]

        @v.bind_key("Down")
        def translate_down(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 50, 0]

        @v.bind_key("Left")
        def translate_left(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 0, -50]

        @v.bind_key("Right")
        def translate_right(v):
            for layer in v.layers:
                if layer.name != "scalebar":
                    layer.translate += [0, 0, 50]

    except Exception as exc:
        # Reinitializing a dock can encounter keys already bound by the prior
        # instance.  Report it through logging rather than writing to stdout.
        logger.warning("Could not install custom key bindings: %s", exc)

    v.scale_bar.visible = True
    # Localizations are placed in world coordinates measured in nanometres,
    # and nothing said so, so the scale bar named the one thing they are not:
    # pixels.  One world unit is one nanometre; given that, napari picks a
    # readable multiple itself.
    #
    # Deliberately not `Layer.units`, which is where napari 0.8 wants this and
    # which emits a FutureWarning here.  Setting it rescales the world: with
    # the two-spot fixture, `reset_view()` then framed a view that pushed both
    # splats into the bottom-right corner, most of the data off-screen.  This
    # spelling has no effect on geometry -- measured, same rendered frame to
    # the pixel.  The package pins napari<0.8, where it still works; whoever
    # lifts that pin has to move to Layer.units and re-check the framing.
    with warnings.catch_warnings():
        # Silenced only here, and only this one: it fires on every dataset
        # load, and 200-odd copies of a decision already recorded above would
        # bury the warnings worth reading.
        warnings.simplefilter("ignore", FutureWarning)
        v.scale_bar.unit = "nm"

    Panning_create_anker(parent=self, viewer=v)


class Panning_create_anker(QWidget):
    def __init__(self, parent, viewer):
        super().__init__()
        self.viewer = viewer
        self.mouse_down = False
        self.mode = None
        self.active = False
        self.parent = parent

        # napari 0.5+ uses mouse_drag_callbacks (generator-based) instead of
        # the old mouse_press/move/release_callbacks API.
        self.viewer.mouse_drag_callbacks.append(self._drag_handler)

    def _drag_handler(self, viewer, event):
        """Generator-based handler combining press, move, and release."""
        # on press — left button (1) + Shift
        if event.button == 1 and "Shift" in event.modifiers:
            # self.create_anker()
            pass  # currently not needed, if necessary needs to be created more efficiently.
        yield
        # on move
        while event.type == "mouse_move":
            if "anker" in self.viewer.layers:
                self.move_anker()
            yield
        # on release
        if "anker" in self.viewer.layers:
            self.remove_anker()

    def create_anker(self):
        center = self.viewer.camera.center
        h = 100 * 0.06 / self.viewer.camera.zoom
        coords = []
        for i in [-h / 2, h / 2]:
            for j in [-h / 2, h / 2]:
                for k in [-h / 2, h / 2]:
                    coords.append([i, j, k])
        self.anker_coords = np.asarray(coords)
        # self.anker_coords=np.asarray([[h,0,0],[0,h,h],[0,h,-h],[0,-h,h],
        # [0,-h,-h],[-h,0,0]])
        self.anker_faces = np.asarray(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
                [1, 4, 5],
                [0, 2, 4],
                [2, 4, 6],
                [1, 3, 5],
                [3, 5, 7],
                [2, 3, 6],
                [3, 6, 7],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )
        verts = np.reshape(self.anker_coords + center, (8, 3))
        self.anker = self.viewer.add_surface(
            (verts, self.anker_faces), name="anker", shading="smooth", blending="opaque"
        )

    def move_anker(self):
        center = self.viewer.camera.center
        self.viewer.layers["anker"].data = (
            np.reshape(self.anker_coords + center, (8, 3)),
            self.anker_faces,
            np.ones(8),
        )

    def remove_anker(self):
        self.viewer.layers.remove("anker")
