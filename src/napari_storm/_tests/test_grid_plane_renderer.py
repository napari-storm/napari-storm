"""
Integration tests for GridPlaneRenderer.

Require a real napari viewer (make_napari_viewer fixture from napari).
"""
import numpy as np
import pytest

from napari_storm.grid_plane_renderer import GridPlaneRenderer
from napari_storm.render_config import RenderConfig

RX = [0.0, 5000.0]
RY = [0.0, 5000.0]
RZ = [0.0, 0.0]


@pytest.fixture
def gpr(make_napari_viewer):
    viewer = make_napari_viewer()
    rc = RenderConfig()
    rc.range_x_percent[:] = [0, 100]
    rc.range_y_percent[:] = [0, 100]
    rc.range_z_percent[:] = [0, 100]
    return GridPlaneRenderer(viewer, rc), viewer


def test_create_adds_vectors_layer(gpr):
    renderer, viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    layer_names = [la.name for la in viewer.layers]
    assert "Grid_Plane" in layer_names


def test_remove_deletes_vectors_layer(gpr):
    renderer, viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    renderer.create_remove(False, RX, RY, RZ)
    layer_names = [la.name for la in viewer.layers]
    assert "Grid_Plane" not in layer_names


def test_double_create_does_not_duplicate(gpr):
    renderer, viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    original = renderer.grid_plane_layer
    renderer.create_remove(True, RX, RY, RZ)
    assert len(viewer.layers) == 1
    assert renderer.grid_plane_layer is original


def test_update_line_distance_does_not_raise(gpr):
    renderer, viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    renderer.update(RX, RY, RZ, line_distance_nm=2000)


def test_update_changes_color(gpr):
    renderer, viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    renderer.update(RX, RY, RZ, color="red")
    assert renderer.current_grid_plane_color == "red"


def test_asymmetric_nonzero_ranges_place_grid_on_world_axes(gpr):
    renderer, _viewer = gpr
    renderer.render_config.range_x_percent[:] = [25, 75]
    renderer.render_config.range_y_percent[:] = [20, 60]
    renderer.create_remove(
        True, [10_000, 12_000], [40_000, 50_000], [-500, 500]
    )

    # Vectors are (z, y, x): a line running along x carries its direction in
    # column 2 and steps across column 1.
    vectors = renderer.grid_plane_layer.data
    x_lines = vectors[vectors[:, 1, 2] > 0]
    y_lines = vectors[vectors[:, 1, 1] > 0]

    np.testing.assert_allclose(x_lines[:, 0, 2], 10_500)
    np.testing.assert_allclose(x_lines[:, 1, 2], 1_000)
    assert x_lines[:, 0, 1].min() == pytest.approx(42_000)
    assert x_lines[:, 0, 1].max() == pytest.approx(46_000)

    np.testing.assert_allclose(y_lines[:, 0, 1], 42_000)
    np.testing.assert_allclose(y_lines[:, 1, 1], 4_000)
    assert y_lines[:, 0, 2].min() == pytest.approx(10_500)
    assert y_lines[:, 0, 2].max() == pytest.approx(11_500)


def test_z_slider_includes_nonzero_axis_minimum(gpr):
    renderer, _viewer = gpr
    renderer.render_config.zdim = True
    renderer.create_remove(True, RX, RY, [-500, 500])
    renderer.update(RX, RY, [-500, 500], z_pos=25)
    np.testing.assert_allclose(renderer.grid_plane_layer.data[:, 0, 0], -250)


def test_recreated_grid_cannot_keep_a_z_position_outside_new_data(gpr):
    renderer, _viewer = gpr
    renderer.render_config.zdim = True
    renderer.create_remove(True, RX, RY, [-500, 500])
    renderer.create_remove(False, RX, RY, [-500, 500])

    renderer.create_remove(True, RX, RY, [10_000, 12_000])
    np.testing.assert_allclose(renderer.grid_plane_layer.data[:, 0, 0], 11_000)


def test_every_grid_vector_has_an_exact_reverse_partner(gpr):
    renderer, _viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    vectors = renderer.grid_plane_layer.data

    # Six unique lines in each orientation, each drawn in both directions.
    assert vectors.shape == (24, 2, 3)
    for start, direction in vectors:
        reverse_start = start + direction
        matches = np.all(np.isclose(vectors[:, 0, :], reverse_start), axis=1)
        matches &= np.all(np.isclose(vectors[:, 1, :], -direction), axis=1)
        assert np.any(matches)


def test_thickness_value_50_matches_creation_and_scales_smoothly(gpr):
    renderer, _viewer = gpr
    renderer.create_remove(True, RX, RY, RZ)
    initial_width = renderer.grid_plane_layer.edge_width

    renderer.update(RX, RY, RZ, line_thickness=50)
    assert renderer.grid_plane_layer.edge_width == pytest.approx(initial_width)

    renderer.update(RX, RY, RZ, line_thickness=60)
    assert renderer.grid_plane_layer.edge_width == pytest.approx(
        initial_width * np.e, abs=0.01
    )


class _StubViewer:
    """Enough of a viewer for the geometry, which touches no layers."""


def _metrics(margin_percent, render_range_x=RX, render_range_y=RY):
    rc = RenderConfig()
    rc.range_x_percent[:] = [0, 100]
    rc.range_y_percent[:] = [0, 100]
    rc.grid_plane_margin_percent = margin_percent
    renderer = GridPlaneRenderer(_StubViewer(), rc)
    return renderer._grid_metrics(render_range_x, render_range_y, 1000.0)


def test_a_grid_with_no_margin_stops_at_the_render_range():
    """The default has to leave the geometry exactly where it was."""
    x0, y0, x_span, y_span, _, _ = _metrics(0.0)
    assert (x0, y0) == (0.0, 0.0)
    assert (x_span, y_span) == (5000.0, 5000.0)


def test_a_margin_widens_the_grid_past_the_data_at_both_ends():
    """Issue #38: the grid could not be drawn outside the data's limits."""
    x0, y0, x_span, y_span, _, _ = _metrics(10.0)
    assert (x0, y0) == (-500.0, -500.0)
    assert (x_span, y_span) == (6000.0, 6000.0)


def test_the_widened_grid_stays_centred_on_the_data():
    x0, y0, x_span, y_span, _, _ = _metrics(25.0, [1000.0, 3000.0], [1000.0, 3000.0])
    assert x0 + x_span / 2 == pytest.approx(2000.0)
    assert y0 + y_span / 2 == pytest.approx(2000.0)


def test_a_negative_margin_is_ignored_rather_than_shrinking_the_grid():
    """A validator keeps it out of the field; the geometry does not rely on that."""
    assert _metrics(-30.0)[2:4] == _metrics(0.0)[2:4]


def test_a_wider_grid_carries_proportionally_more_lines():
    _, _, _, _, lines_x, lines_y = _metrics(0.0)
    _, _, _, _, wide_x, wide_y = _metrics(100.0)
    # A 100% margin trebles each span, so it fits three times the intervals.
    assert (wide_x, wide_y) == (3 * lines_x, 3 * lines_y)
