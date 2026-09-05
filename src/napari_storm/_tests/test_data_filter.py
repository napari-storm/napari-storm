"""Parameter filtering: which rows a band excludes, and who owns that record.

The two ~40-line blocks in `apply_filtering` and `apply_filtering_to_all` were
near-identical and untested; they are now one helper, and this pins the
behaviour they had. It also covers the move of the per-dataset record from a
position-indexed list to a dict keyed by stable dataset id.
"""

import numpy as np

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


def _dataset(values, name="ds"):
    """A 2-D dataset whose x column is exactly *values*."""
    values = np.asarray(values, dtype="f4")
    locs = np.zeros(len(values), dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = values
    locs["y_pos_nm"] = values
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _widget_with(make_napari_viewer, *datasets):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(list(datasets))
    filters = widget.data_filter_itf
    filters.list_of_filterable_parameters = ["x_pos_nm", "y_pos_nm"]
    filters.current_parameter_idx = 0
    filters.current_dataset_idx = 0
    return widget, filters


def _band(filters, low, high, mode="Bandpass"):
    filters.filter_slider_values_decimal = [low, high]
    filters.filter_mode_active_idx = filters.filter_modes.index(mode)


# ------------------------------------------------------------------- the band


def test_bandpass_records_the_rows_outside_the_band(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)

    # x spans 0..9, so 20%..60% of that span is 1.8..5.4.
    _band(filters, 0.2, 0.6)
    filters.apply_filtering(0, update_layers=False)

    assert filters.indices_for(dataset).tolist() == [0, 1, 6, 7, 8, 9]


def test_bandstop_records_the_rows_inside_the_band(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)

    _band(filters, 0.2, 0.6, mode="Bandstop")
    filters.apply_filtering(0, update_layers=False)

    assert filters.indices_for(dataset).tolist() == [2, 3, 4, 5]


def test_applying_a_second_band_accumulates(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)

    _band(filters, 0.0, 0.5)
    filters.apply_filtering(0, update_layers=False)
    first = filters.indices_for(dataset).tolist()

    _band(filters, 0.5, 1.0)
    filters.apply_filtering(0, update_layers=False)

    # Filters compose: nothing a previous band excluded comes back.
    assert set(first).issubset(filters.indices_for(dataset).tolist())


def test_no_band_at_all_excludes_nothing(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)

    _band(filters, 0.0, 1.0)
    filters.apply_filtering(0, update_layers=False)

    assert filters.indices_for(dataset).tolist() == []


def test_apply_filtering_defaults_to_the_selected_dataset(make_napari_viewer):
    """The button hands Qt's `checked` bool through; None must work too."""
    first, second = _dataset(range(10), "a"), _dataset(range(10), "b")
    widget, filters = _widget_with(make_napari_viewer, first, second)
    filters.current_dataset_idx = 1

    _band(filters, 0.2, 0.6)
    filters.apply_filtering(False, update_layers=False)
    assert filters.indices_for(second).size
    assert filters.indices_for(first).size == 0

    filters.filter_indices.clear()
    filters.apply_filtering(None, update_layers=False)
    assert filters.indices_for(second).size


# ------------------------------------------------------------------ apply-all


def test_apply_to_all_uses_the_visible_band_everywhere(make_napari_viewer):
    """The band comes from the histogram the user is looking at.

    The second dataset covers a different range, so recomputing a band per
    dataset would exclude a different fraction of each. Applying *this* band
    means the same absolute cut lands on both.
    """
    shown = _dataset(range(10), "shown")
    other = _dataset(np.arange(10) + 100, "other")
    widget, filters = _widget_with(make_napari_viewer, shown, other)
    filters.current_dataset_idx = 0

    _band(filters, 0.2, 0.6)
    filters.apply_filtering_to_all()

    assert filters.indices_for(shown).tolist() == [0, 1, 6, 7, 8, 9]
    # 1.8..5.4 excludes every row of a dataset that starts at 100.
    assert filters.indices_for(other).tolist() == list(range(10))


# ------------------------------------------------------------------- identity


def test_filter_records_are_keyed_by_dataset_id(make_napari_viewer):
    first, second = _dataset(range(10), "a"), _dataset(range(10), "b")
    widget, filters = _widget_with(make_napari_viewer, first, second)

    _band(filters, 0.2, 0.6)
    filters.apply_filtering(1, update_layers=False)

    assert sorted(filters.filter_indices) == [second.dataset_id]


def test_a_filter_survives_an_earlier_dataset_being_unloaded(make_napari_viewer):
    """The misalignment the id keying exists to prevent."""
    first, second = _dataset(range(10), "a"), _dataset(range(10), "b")
    widget, filters = _widget_with(make_napari_viewer, first, second)

    _band(filters, 0.2, 0.6)
    filters.apply_filtering(1, update_layers=False)
    recorded = filters.indices_for(second).tolist()

    widget.unload_dataset(0)

    assert filters.indices_for(second).tolist() == recorded
    assert first.dataset_id not in filters.filter_indices


def test_unloading_releases_the_filter_record(make_napari_viewer):
    first, second = _dataset(range(10), "a"), _dataset(range(10), "b")
    widget, filters = _widget_with(make_napari_viewer, first, second)

    _band(filters, 0.2, 0.6)
    filters.apply_filtering_to_all()
    assert len(filters.filter_indices) == 2

    widget.unload_dataset(0)
    assert sorted(filters.filter_indices) == [second.dataset_id]


def test_clearing_the_session_releases_every_record(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)
    _band(filters, 0.2, 0.6)
    filters.apply_filtering(0, update_layers=False)

    widget.clear_datasets()
    assert filters.filter_indices == {}


def test_reset_restores_every_dataset(make_napari_viewer):
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)
    _band(filters, 0.2, 0.6)
    filters.apply_filtering(0, update_layers=False)

    filters.reset_all_filtering()

    assert filters.filter_indices == {}
    assert dataset.number_of_active_entries() == 10


# ------------------------------------------------------------ reaching render


def test_recorded_indices_reach_the_rendered_selection(make_napari_viewer):
    """The whole point: a recorded filter must actually remove localizations."""
    dataset = _dataset(range(10))
    widget, filters = _widget_with(make_napari_viewer, dataset)

    _band(filters, 0.2, 0.6)
    filters.apply_filtering(0, update_layers=False)
    widget.data_to_layer_itf.update_data_range(dataset)

    assert dataset.x_pos_nm.tolist() == [2.0, 3.0, 4.0, 5.0]
