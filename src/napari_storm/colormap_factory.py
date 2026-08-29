import napari
import numpy as np
from qtpy.QtGui import QIcon, QImage, QPixmap

try:  # Qt6 scoped enums; qtpy also exposes these on Qt5
    _RGBA8888 = QImage.Format.Format_RGBA8888
except AttributeError:  # pragma: no cover - very old bindings
    _RGBA8888 = QImage.Format_RGBA8888


def _rgba_pixmap(rgba):
    """Convert a float RGBA array in [0, 1] to a QPixmap.

    Builds the QImage straight from the array buffer rather than going through
    PIL.  ``PIL.ImageQt.ImageQt`` was removed in Pillow 12, and its replacement
    binds to whichever Qt module PIL happened to import first, which is not
    necessarily the binding qtpy selected.
    """
    buf = np.ascontiguousarray(np.uint8(rgba * 255))
    height, width = buf.shape[:2]
    # .copy() forces the QImage to own its pixels so it outlives ``buf``.
    return QPixmap.fromImage(
        QImage(buf.tobytes(), width, height, 4 * width, _RGBA8888).copy()
    )


def make_colormaps():
    """Build the 11 custom colormaps used for STORM channel display.

    Returns
    -------
    tuple
        (colormaps, colormap_icons) — lists of napari Colormap objects and
        matching QIcon/QPixmap thumbnails.
    """
    cmaps = []
    cmap_icons = []
    names = ["red", "green", "blue"]
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
        icon = np.zeros((128, 128, 4))
        icon[:, :, -1] = 1
        icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon = QIcon(_rgba_pixmap(icon))
        cmap_icons.append(icon)
    names = ["yellow", "cyan", "pink"]
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i + 1) % 3] = 1
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
        icon = np.zeros((128, 128, 4))
        icon[:, :, -1] = 1
        icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon[:, :, (i + 1) % 3] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon = QIcon(_rgba_pixmap(icon))
        cmap_icons.append(icon)
    names = ["orange", "mint", "purple"]
    for i in range(3):
        colors = np.zeros((2, 4))
        colors[-1][i] = 1
        colors[-1][(i + 1) % 3] = 0.5
        colors[-1][-1] = 1
        cmaps.append(
            napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
        )
        icon = np.zeros((128, 128, 4))
        icon[:, :, -1] = 1
        icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon[:, :, (i + 1) % 3] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon = QIcon(_rgba_pixmap(icon))
        cmap_icons.append(icon)
    colors = np.zeros((2, 4))
    colors[-1][:] = 1
    colors[:, -1] = 1
    cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name="gray"))
    icon = np.zeros((128, 128, 4))
    icon[:, :, -1] = 1
    for i in range(4):
        icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
    icon = QIcon(_rgba_pixmap(icon))
    cmap_icons.append(icon)
    colors = np.zeros((4, 4))
    colors[1:, -1] = 1
    colors[1:, 0] = 1
    colors[2:, 1] = 1
    colors[3, 2] = 1
    cmaps.append(
        napari.utils.colormaps.colormap.Colormap(colors=colors, name="red hot")
    )
    icon = np.zeros((128, 128, 4))
    icon[:, :, -1] = 1
    icon[:, 16:48, 0] = np.ones((128, 32)) * np.arange(32) / 32
    icon[:, 48:, 0] = 1
    icon[:, 48:112, 1] = np.ones((128, 64)) * np.arange(64) / 64
    icon[:, 96:, 1] = 1
    icon[:, 96:, 2] = np.ones((128, 32)) * np.arange(32) / 32
    icon = QIcon(_rgba_pixmap(icon))
    cmap_icons.append(icon)
    icon = np.zeros((128, 600, 4))
    icon[:, :100, 0] = 1
    icon[:, :100, 1] = np.ones((128, 100)) * np.arange(100) / 100
    icon[:, 100:200, 0] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
    icon[:, 100:200, 1] = 1
    icon[:, 200:300, 2] = np.ones((128, 100)) * np.arange(100) / 100
    icon[:, 200:300, 1] = 1
    icon[:, 300:400, 2] = 1
    icon[:, 300:400, 1] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
    icon[:, 400:500, 0] = np.ones((128, 100)) * np.arange(100) / 100
    icon[:, 400:500, 2] = 1
    icon[:, 500:, 0] = 1
    icon[:, 500:, 2] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
    icon[:, :, 3] = 1
    qpm = _rgba_pixmap(icon)
    cmap_icons.append(qpm)
    return cmaps, cmap_icons
