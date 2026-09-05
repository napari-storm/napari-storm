"""
Integration tests for ScalebarRenderer.

ScalebarRenderer calls napari.current_viewer() internally, so the viewer
created by make_napari_viewer() must be current at call time.
"""

import pytest

from napari_storm.render_config import RenderConfig
from napari_storm.scalebar_renderer import ScalebarRenderer


@pytest.fixture
def sbr_and_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    rc = RenderConfig()
    return ScalebarRenderer(viewer, rc), rc, viewer


def test_no_scalebar_when_disabled(sbr_and_viewer, three_d_dataset):
    sbr, rc, viewer = sbr_and_viewer
    rc.scalebar_enabled = False
    sbr.update(three_d_dataset)
    layer_names = [la.name for la in viewer.layers]
    assert "scalebar" not in layer_names


def test_scalebar_created_when_enabled(sbr_and_viewer, three_d_dataset):
    sbr, rc, viewer = sbr_and_viewer
    rc.scalebar_enabled = True
    rc.scalebar_size_nm = 500
    sbr.update(three_d_dataset)
    layer_names = [la.name for la in viewer.layers]
    assert "scalebar" in layer_names


def test_scalebar_removed_when_toggled_off(sbr_and_viewer, three_d_dataset):
    sbr, rc, viewer = sbr_and_viewer
    rc.scalebar_enabled = True
    sbr.update(three_d_dataset)
    rc.scalebar_enabled = False
    sbr.update(three_d_dataset)
    layer_names = [la.name for la in viewer.layers]
    assert "scalebar" not in layer_names


def test_scalebar_update_in_place_does_not_duplicate(sbr_and_viewer, three_d_dataset):
    """Calling update() twice while enabled must not create duplicate layers."""
    sbr, rc, viewer = sbr_and_viewer
    rc.scalebar_enabled = True
    sbr.update(three_d_dataset)
    sbr.update(three_d_dataset)  # second call: update in-place
    count = sum(1 for la in viewer.layers if la.name == "scalebar")
    assert count == 1


def test_scalebar_stale_exists_flag_does_not_crash(sbr_and_viewer, three_d_dataset):
    """If scalebar_exists is False but layer is in viewer, update() must not crash."""
    sbr, rc, viewer = sbr_and_viewer
    rc.scalebar_enabled = True
    sbr.update(three_d_dataset)
    # Simulate stale flag (e.g. after widget rebuild)
    sbr.scalebar_exists = False
    sbr.scalebar_layer = None
    sbr.update(three_d_dataset)  # should not raise AssertionError
    layer_names = [la.name for la in viewer.layers]
    assert "scalebar" in layer_names
