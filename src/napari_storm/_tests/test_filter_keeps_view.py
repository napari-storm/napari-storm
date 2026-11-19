"""Filtering keeps the view the user was looking at.

Applying a filter used to re-measure the data's extent and re-centre the
camera on it -- a jump back to the middle of the dataset for anyone zoomed in
on a detail, though a filter cannot change the extent at all: it is measured
over every localization, filtered or not.
"""

import numpy as np
import pytest

from napari_storm._tests.test_density_split import (
    CLUSTER,
    _open,
    _select_density_band,
    _widget,
    _write_points,
)

LOOKING_AT = (1.0, 40_003.0, 10_002.0)  # (z, y, x): inside the cluster
ZOOM = 7.5


def _look_at_a_detail(widget):
    camera = widget.viewer.camera
    camera.center = LOOKING_AT
    camera.zoom = ZOOM


def _assert_view_kept(widget):
    camera = widget.viewer.camera
    assert tuple(camera.center) == pytest.approx(LOOKING_AT)
    assert camera.zoom == pytest.approx(ZOOM)


@pytest.fixture
def widget(make_napari_viewer, tmp_path):
    widget = _widget(make_napari_viewer)
    _open(widget, _write_points(tmp_path / "cells.ns"))
    widget.data_filter_itf.calculate_local_density(35.0)
    _select_density_band(widget.data_filter_itf, 0.5)
    _look_at_a_detail(widget)
    return widget


def test_applying_a_filter_keeps_the_view(widget):
    widget.data_filter_itf.apply_filtering(0)

    assert widget.data_filter_itf.indices_for(widget.localization_datasets[0]).size
    _assert_view_kept(widget)


def test_applying_a_filter_to_all_datasets_keeps_the_view(widget):
    widget.data_filter_itf.apply_filtering_to_all()

    _assert_view_kept(widget)


def test_splitting_keeps_the_view(widget):
    widget.data_filter_itf.apply_filtering(0)

    pieces = widget.data_filter_itf.split_current_dataset()

    assert pieces["in"].number_of_entries() == CLUSTER
    _assert_view_kept(widget)


def test_arithmetic_that_moves_no_localization_keeps_the_view(widget):
    adjust = widget.data_adjustment_itf
    dataset = widget.localization_datasets[0]
    dataset.table.set_side_column("brightness", np.ones(dataset.number_of_entries()))
    adjust.list_of_adjustable_parameters = ["brightness"]
    adjust.current_parameter_idx = 0
    adjust.math_mode_active_idx = 1  # rescale
    adjust.value = 2.0

    adjust.apply_filtering(0)

    _assert_view_kept(widget)


def test_arithmetic_that_moves_the_data_still_follows_it(widget):
    adjust = widget.data_adjustment_itf
    adjust.current_parameter_idx = adjust.list_of_adjustable_parameters.index(
        "x_pos_nm"
    )
    adjust.math_mode_active_idx = 0  # add offset
    adjust.value = 100_000.0

    adjust.apply_filtering(0)

    assert widget.viewer.camera.center[2] > LOOKING_AT[2] + 50_000.0
