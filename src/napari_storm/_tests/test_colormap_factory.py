"""
Tests for colormap_factory.make_colormaps().

Requires a QApplication (provided by the qtbot fixture from pytest-qt).
"""

import napari.utils.colormaps.colormap as ncc
from qtpy.QtGui import QIcon, QPixmap

from napari_storm.colormap_factory import make_colormaps

EXPECTED_NAMES = {
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "pink",
    "orange",
    "mint",
    "purple",
    "gray",
    "red hot",
}


def test_count(qtbot):
    cmaps, icons = make_colormaps()
    assert len(cmaps) == 11
    # 12 icons: 11 channel icons + 1 extra QPixmap for the z-color-encoding bar
    assert len(icons) == 12


def test_colormap_names(qtbot):
    cmaps, _ = make_colormaps()
    names = {cm.name for cm in cmaps}
    assert names == EXPECTED_NAMES


def test_icon_types(qtbot):
    _, icons = make_colormaps()
    # First 11 are QIcon; last is a QPixmap (the z-color-encoding bar)
    for icon in icons[:11]:
        assert isinstance(icon, QIcon)
    assert isinstance(icons[-1], QPixmap)


def test_colormaps_are_valid_napari_colormaps(qtbot):
    cmaps, _ = make_colormaps()
    for cm in cmaps:
        assert isinstance(cm, ncc.Colormap)
