"""Local density, and splitting a dataset in two by a filter band.

Density is daxview's measure: for each localization, how many *other*
localizations lie within a radius -- a sphere in 3-D, a circle in 2-D, the
radius itself included -- counting only what the filters applied so far keep.

A split replaces one dataset with two channels, what a band keeps ("in") and
what it removes ("out"), and a scene makes them again from the file.
"""

import h5py
import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (
    DENSITY_COLUMN,
    drop_edits,
    edits_of,
    local_density,
    record_edit,
    split_lineage,
)
from napari_storm.dataset_split import split_by_filters, split_dataset

# ----------------------------------------------------------------- the measure


def test_density_counts_the_others_within_the_radius_its_edge_included():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [40.0, 0.0]])

    assert local_density(points, 10.0).tolist() == [1, 2, 1, 0]


def test_density_is_counted_in_a_sphere_for_3d_data():
    stacked = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 11.0]])

    assert local_density(stacked, 10.0).tolist() == [1, 2, 1]
    # Seen from above they all coincide.
    assert local_density(stacked[:, :2], 10.0).tolist() == [2, 2, 2]


def test_density_agrees_with_counting_every_pair():
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 500, size=(400, 3))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    expected = (distances <= 60.0).sum(axis=1) - 1

    np.testing.assert_array_equal(local_density(points, 60.0), expected)


def test_rows_the_filters_removed_are_counted_but_count_for_no_one():
    points = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    kept = np.array([True, False, True])

    assert local_density(points, 10.0, neighbours=kept).tolist() == [1, 2, 1]


def test_a_localization_without_a_position_neither_counts_nor_is_counted():
    points = np.array([[0.0, 0.0], [np.nan, 0.0], [5.0, 0.0]])

    assert local_density(points, 10.0).tolist() == [1, 0, 1]


@pytest.mark.parametrize("radius", [0.0, -5.0, np.nan])
def test_density_needs_a_positive_radius(radius):
    with pytest.raises(ValueError):
        local_density(np.zeros((3, 2)), radius)


# ------------------------------------------------------------ the edit log


class _Logged:
    pass


def test_resetting_filters_keeps_what_a_split_baked_in():
    part = _Logged()
    for edit in (
        {
            "kind": "filter",
            "parameter": "x",
            "low": 0.0,
            "high": 1.0,
            "mode": "Bandpass",
        },
        {
            "kind": "split",
            "parameter": "x",
            "low": 0.0,
            "high": 1.0,
            "mode": "Bandpass",
            "part": "in",
        },
        {
            "kind": "filter",
            "parameter": "y",
            "low": 0.0,
            "high": 1.0,
            "mode": "Bandpass",
        },
    ):
        record_edit(part, edit)

    drop_edits(part, "filter")

    assert [edit["kind"] for edit in edits_of(part)] == ["filter", "split"]


def test_the_two_parts_of_a_split_have_different_lineages():
    band = {"parameter": "x", "low": 0.0, "high": 1.0, "mode": "Bandpass"}
    inside = [{"kind": "split", **band, "part": "in"}]
    outside = [{"kind": "split", **band, "part": "out"}]

    assert split_lineage([]) == ""
    assert split_lineage(inside) != split_lineage(outside)
    # Later edits are not part of how the part was made.
    later = inside + [{"kind": "density", "radius_nm": 35.0}]
    assert split_lineage(later) == split_lineage(inside)


# ------------------------------------------------------------- in the widget

CLUSTER = 20
SCATTERED = 20


def _write_points(path):
    """A tight cluster of 20 and 20 lone localizations a micrometre apart."""
    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            10_000.0 + rng.uniform(-5, 5, CLUSTER),
            20_000.0 + 1_000.0 * np.arange(SCATTERED),
        ]
    )
    y = np.concatenate(
        [40_000.0 + rng.uniform(-5, 5, CLUSTER), np.full(SCATTERED, 40_000.0)]
    )
    locs = np.rec.array(
        np.zeros(len(x), dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    )
    locs.x_pos_nm = x
    locs.y_pos_nm = y
    with h5py.File(path, "w") as file:
        stored = file.create_dataset("dataset", data=locs)
        stored.attrs["name"] = "cells"
        stored.attrs["zdim_present"] = False
        stored.attrs["dataset_class"] = "LocalizationDataBaseClass"
    return str(path)


def _widget(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.reported = []
    widget._warn_user = widget.reported.append
    return widget


def _open(widget, path):
    assert widget.open_localization_data_file_and_get_dataset(file_path=str(path))
    return widget.localization_datasets[-1]


def _density(dataset):
    return np.asarray(dataset.table.column(DENSITY_COLUMN))


def _select_density_band(itf, low_fraction):
    """Put the sliders on the upper part of the density histogram."""
    itf.current_parameter_idx = itf.list_of_filterable_parameters.index(DENSITY_COLUMN)
    itf.filter_slider_values_decimal = (low_fraction, 1.0)


def _filter_then_split(itf, low_fraction=0.5):
    """What a user does: apply a density threshold, like it, split."""
    _select_density_band(itf, low_fraction)
    itf.apply_filtering(0)
    return itf.split_current_dataset()


def test_calculating_density_adds_a_parameter_to_filter_and_colour_by(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf

    itf.calculate_local_density(35.0)

    counts = _density(dataset)
    assert counts[:CLUSTER].tolist() == [CLUSTER - 1] * CLUSTER
    assert counts[CLUSTER:].tolist() == [0] * SCATTERED
    assert DENSITY_COLUMN in itf.list_of_filterable_parameters
    assert itf.list_of_filterable_parameters[itf.current_parameter_idx] == (
        DENSITY_COLUMN
    )
    assert widget.Bcolor_by.findData(DENSITY_COLUMN) >= 0
    assert edits_of(dataset)[-1] == {"kind": "density", "radius_nm": 35.0}


def test_only_localizations_the_filters_keep_count_as_neighbours(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf._record_indices(dataset, np.arange(10))  # half the cluster filtered out

    itf.calculate_local_density(35.0)

    counts = _density(dataset)
    assert counts[:10].tolist() == [10] * 10  # removed: count the 10 kept
    assert counts[10:CLUSTER].tolist() == [9] * 10  # kept: the 9 others


def test_a_density_threshold_filters_like_any_parameter(make_napari_viewer, tmp_path):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    _select_density_band(itf, 0.5)

    itf.apply_filtering(0)

    np.testing.assert_array_equal(
        itf.indices_for(dataset), np.arange(CLUSTER, CLUSTER + SCATTERED)
    )


def test_the_sliders_move_over_the_density_histogram(make_napari_viewer, tmp_path):
    """The band slider drew its marker by reading the record array, which has
    only the measured columns: dragging it on local_density raised."""
    widget = _widget(make_napari_viewer)
    _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    window = itf.dfw

    window.Sfilter_slider.setValue((500, 1000))  # draws the band marker
    itf.xrange_slider_values_decimal = (0.0, 0.5)
    itf.current_parameter_changed(reset=False)  # what the range slider does
    window.Sfilter_slider.virtual_feedback()

    marker = window.CanvasWidget.Canvas.ax.lines[-1].get_xdata()
    assert marker[0] == pytest.approx((CLUSTER - 1) * 0.5)
    assert marker[-1] == pytest.approx(CLUSTER - 1)


def test_splitting_by_density_makes_two_channels_in_place_of_one(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    original = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    _select_density_band(itf, 0.5)
    itf.apply_filtering(0)  # the sliders spring back to the full range here

    pieces = itf.split_current_dataset()

    assert original not in widget.localization_datasets
    assert [d.name for d in widget.localization_datasets] == ["cells in", "cells out"]
    assert pieces["in"].number_of_entries() == CLUSTER
    assert pieces["out"].number_of_entries() == SCATTERED
    assert _density(pieces["in"]).tolist() == [CLUSTER - 1] * CLUSTER
    assert len(widget.channel) == 2
    colours = [channel.Colormap_selector.currentText() for channel in widget.channel]
    assert colours[0] != colours[1]
    # Each part remembers how it was made.
    assert [edit["kind"] for edit in edits_of(pieces["out"])] == [
        "density",
        "filter",
        "split",
    ]
    assert edits_of(pieces["out"])[-1] == {"kind": "split", "part": "out"}


def test_a_split_recorded_with_a_band_still_divides_what_the_filters_keep(
    make_napari_viewer, tmp_path
):
    """How splits were first recorded; scenes saved then still load."""
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    itf._record_indices(dataset, np.arange(CLUSTER, CLUSTER + 5))
    band = {
        "parameter": DENSITY_COLUMN,
        "low": 10.0,
        "high": 100.0,
        "mode": "Bandpass",
    }

    pieces = split_dataset(widget, dataset, band)

    assert pieces["in"].number_of_entries() == CLUSTER
    assert pieces["out"].number_of_entries() == SCATTERED - 5


def test_out_holds_everything_any_filter_removed(make_napari_viewer, tmp_path):
    """Two filters, then a split: 'out' is what either removed, not the last."""
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    # One filter drops the last five lone localizations...
    itf.replay_band(
        dataset,
        {"parameter": "x_pos_nm", "low": 0.0, "high": 34_500.0, "mode": "Bandpass"},
    )
    itf.calculate_local_density(35.0)

    # ... a density threshold the other fifteen, then the split.
    pieces = _filter_then_split(itf)

    assert pieces["in"].number_of_entries() == CLUSTER
    assert pieces["out"].number_of_entries() == SCATTERED
    assert [edit["kind"] for edit in edits_of(pieces["out"])] == [
        "filter",
        "density",
        "filter",
        "split",
    ]


def test_there_is_nothing_to_split_by_before_a_filter_is_applied(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))

    with pytest.raises(ValueError, match="Apply one or more"):
        split_by_filters(widget, dataset)


def test_filters_that_removed_nothing_are_not_split_by(make_napari_viewer, tmp_path):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    _select_density_band(itf, 0.0)
    itf.apply_filtering(0)

    with pytest.raises(ValueError, match="removed nothing"):
        split_by_filters(widget, dataset)


def test_density_counted_after_a_filter_splits_as_it_was_counted(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.replay_band(
        dataset,
        {"parameter": "x_pos_nm", "low": 0.0, "high": 34_500.0, "mode": "Bandpass"},
    )
    itf.calculate_local_density(35.0)

    pieces = split_by_filters(widget, dataset)

    assert pieces["out"].number_of_entries() == 5
    assert pieces["in"].number_of_entries() == CLUSTER + SCATTERED - 5


def test_a_split_that_would_leave_a_part_empty_is_refused(make_napari_viewer, tmp_path):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))
    band = {"parameter": "x_pos_nm", "low": 0.0, "high": 1e9, "mode": "Bandpass"}

    with pytest.raises(ValueError, match="out part"):
        split_dataset(widget, dataset, band)

    assert widget.localization_datasets[0] is dataset
    assert len(widget.localization_datasets) == 1


def test_a_dataset_is_not_split_while_its_positions_are_locked(
    make_napari_viewer, tmp_path
):
    """Drift correction locks positions; a part would bake the corrected ones in."""
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_points(tmp_path / "cells.ns"))

    with dataset.table.position_write_lock(), pytest.raises(ValueError):
        dataset.subset(np.ones(dataset.number_of_entries(), dtype=bool))


def test_resetting_filters_on_a_part_keeps_its_rows_and_recounts_density(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    inside = _filter_then_split(itf)["in"]
    itf._record_indices(inside, np.arange(10))
    itf.compute_density(inside, 35.0)
    assert _density(inside)[10:].tolist() == [9] * 10

    itf.reset_all_filtering()

    assert inside.number_of_entries() == CLUSTER
    assert _density(inside).tolist() == [CLUSTER - 1] * CLUSTER
    assert [edit["kind"] for edit in edits_of(inside)] == [
        "density",
        "filter",
        "split",
        "density",
    ]


def test_a_scene_makes_the_split_parts_again(make_napari_viewer, tmp_path):
    before = _widget(make_napari_viewer)
    _open(before, _write_points(tmp_path / "cells.ns"))
    itf = before.data_filter_itf
    itf.calculate_local_density(35.0)
    pieces = _filter_then_split(itf)
    # A filter on one part after the split: replayed after the split is made.
    itf.replay_band(
        pieces["out"],
        {"parameter": "x_pos_nm", "low": 0.0, "high": 25_000.0, "mode": "Bandpass"},
    )
    removed = itf.indices_for(pieces["out"]).copy()
    assert removed.size
    before.set_color_coding(color_by=DENSITY_COLUMN)
    path = tmp_path / "split.json"
    before.save_scene_to(path)
    assert before.reported == []

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    assert after.reported == []
    assert [d.name for d in after.localization_datasets] == ["cells in", "cells out"]
    for saved in pieces.values():
        remade = next(d for d in after.localization_datasets if d.name == saved.name)
        assert remade.number_of_entries() == saved.number_of_entries()
        np.testing.assert_array_equal(_density(remade), _density(saved))
        assert edits_of(remade) == edits_of(saved)
    remade_out = after.localization_datasets[1]
    np.testing.assert_array_equal(
        after.data_filter_itf.indices_for(remade_out), removed
    )
    assert after.render_config.color_by == DENSITY_COLUMN


def test_loading_a_split_scene_onto_its_own_parts_changes_nothing(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    _open(widget, _write_points(tmp_path / "cells.ns"))
    itf = widget.data_filter_itf
    itf.calculate_local_density(35.0)
    pieces = _filter_then_split(itf)
    path = tmp_path / "split.json"
    widget.save_scene_to(path)

    widget.load_scene_from(path)

    assert widget.reported == []
    assert list(widget.localization_datasets) == [pieces["in"], pieces["out"]]


def _refuse_reopening(widget, monkeypatch):
    def reopen(*_args, **_kwargs):
        raise AssertionError("the scene read its file again")

    monkeypatch.setattr(widget, "open_localization_data_file_and_get_dataset", reopen)


def test_a_split_scene_cuts_the_file_already_open_rather_than_reopening_it(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Reported: the file was open, and loading the scene read it again."""
    data = _write_points(tmp_path / "cells.ns")
    before = _widget(make_napari_viewer)
    _open(before, data)
    before.data_filter_itf.calculate_local_density(35.0)
    saved = _filter_then_split(before.data_filter_itf)
    path = tmp_path / "split.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    _open(after, data)  # the file, as it opens
    _refuse_reopening(after, monkeypatch)
    after.load_scene_from(path)

    assert after.reported == []
    assert [d.name for d in after.localization_datasets] == ["cells in", "cells out"]
    for part, made in zip(("in", "out"), after.localization_datasets):
        assert made.number_of_entries() == saved[part].number_of_entries()
        assert edits_of(made) == edits_of(saved[part])


def test_a_scene_replays_its_filters_onto_the_file_opened_by_hand(
    make_napari_viewer, tmp_path, monkeypatch
):
    data = _write_points(tmp_path / "cells.ns")
    before = _widget(make_napari_viewer)
    filtered = _open(before, data)
    before.data_filter_itf.replay_band(
        filtered,
        {"parameter": "x_pos_nm", "low": 0.0, "high": 25_000.0, "mode": "Bandpass"},
    )
    removed = before.data_filter_itf.indices_for(filtered).copy()
    path = tmp_path / "filtered.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    opened = _open(after, data)
    _refuse_reopening(after, monkeypatch)
    after.load_scene_from(path)

    assert after.reported == []
    np.testing.assert_array_equal(after.data_filter_itf.indices_for(opened), removed)
