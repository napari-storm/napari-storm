"""The canonical table's contract, tested without a host application.

`test_active_mask.py` covers the same behaviour through the dataset classes,
which is where the application uses it. These tests pin down the core contract
itself: stable IDs, unit resolution, ownership, and cache invalidation.
"""

import numpy as np
import pytest

from napari_storm.core import (
    COORDINATE_DTYPE,
    DEFAULT_POSITION_COLUMNS,
    LocalizationTable,
)

NM_COLUMNS = DEFAULT_POSITION_COLUMNS
PIXEL_COLUMNS = {"x": "x_pos_pixels", "y": "y_pos_pixels", "z": "z_pos_pixels"}


def _nm_records(n=10, zdim=False):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    records = np.rec.array(np.zeros(n, dtype=fields))
    records.x_pos_nm = np.arange(n, dtype="f4")
    records.y_pos_nm = np.arange(n, dtype="f4") * 2
    if zdim:
        records.z_pos_nm = np.arange(n, dtype="f4") * 3
    return records


def _pixel_records(n=10):
    records = np.rec.array(
        np.zeros(n, dtype=[("x_pos_pixels", "f4"), ("y_pos_pixels", "f4")])
    )
    records.x_pos_pixels = np.arange(n, dtype="f4")
    records.y_pos_pixels = np.arange(n, dtype="f4") * 2
    return records


def _nm_table(n=10, zdim=False):
    return LocalizationTable(_nm_records(n, zdim=zdim))


def _pixel_table(n=10, scale=100.0):
    return LocalizationTable(
        _pixel_records(n), position_columns=PIXEL_COLUMNS, position_scale_nm=scale
    )


# ------------------------------------------------------------------- ownership


def test_the_table_copies_its_records_by_default():
    records = _nm_records()
    table = LocalizationTable(records)
    records.x_pos_nm[0] = 999.0
    assert table.records.x_pos_nm[0] == 0.0


def test_copy_false_hands_ownership_to_the_table():
    records = _nm_records()
    table = LocalizationTable(records, copy=False)
    # Handed out read-only, but backed by the caller's own array.
    assert np.shares_memory(table.records, records)


def test_active_records_are_a_read_only_view_when_nothing_is_filtered():
    table = _nm_table()
    assert np.shares_memory(table.active_records, table.records)
    with pytest.raises(ValueError):
        table.active_records.x_pos_nm[0] = 1.0


def test_active_records_are_read_only_when_filtered():
    table = _nm_table()
    table.apply_filters(np.arange(4))
    assert len(table.active_records) == 4
    with pytest.raises(ValueError):
        table.active_records.x_pos_nm[0] = 1.0


# ------------------------------------------------------------------ stable IDs


def test_active_ids_are_canonical_row_indices():
    table = _nm_table()
    table.apply_filters(np.array([1, 4, 7]))
    assert table.active_ids.tolist() == [1, 4, 7]
    # Narrowing further keeps the surviving IDs addressing the same rows.
    table.bandpass("x_pos_nm", 2, 10)
    assert table.active_ids.tolist() == [4, 7]
    assert table.records.x_pos_nm[table.active_ids].tolist() == [4.0, 7.0]


def test_ids_survive_a_reset():
    table = _nm_table()
    table.apply_filters(np.array([3]))
    table.reset()
    assert table.active_ids.tolist() == list(range(10))


# ----------------------------------------------------------------------- units


def test_pixel_columns_are_exposed_in_nanometres():
    table = _pixel_table()
    assert table.coordinate_nm("x").tolist() == [i * 100.0 for i in range(10)]
    assert table.coordinate_nm("x").dtype == COORDINATE_DTYPE


def test_nanometre_bounds_are_converted_to_the_stored_unit():
    table = _pixel_table()
    values, low, high = table.resolve_property("x_pos_nm", 200.0, 400.0)
    assert values.tolist() == list(range(10))
    assert (low, high) == (2.0, 4.0)


def test_a_column_that_exists_verbatim_is_used_as_is():
    table = _pixel_table()
    values, low, high = table.resolve_property("x_pos_pixels", 2.0, 4.0)
    assert (low, high) == (2.0, 4.0)
    assert values.tolist() == list(range(10))


def test_an_unknown_column_is_a_clear_error():
    table = _nm_table()
    with pytest.raises(KeyError):
        table.resolve_property("photon_count", 0, 1)


def test_changing_the_scale_invalidates_the_cached_coordinates():
    table = _pixel_table()
    assert table.coordinate_nm("x")[1] == 100.0
    table.position_scale_nm = 10.0
    assert table.coordinate_nm("x")[1] == 10.0


def test_setting_the_same_scale_keeps_the_cache():
    table = _pixel_table()
    first = table.coordinate_nm("x")
    table.position_scale_nm = 100.0
    assert table.coordinate_nm("x") is first


def test_missing_axes_are_reported_rather_than_guessed():
    table = _nm_table(zdim=False)
    assert table.has_axis("x")
    assert not table.has_axis("z")
    with pytest.raises(KeyError):
        table.coordinate_nm("z")


# ---------------------------------------------------------------- invalidation


def test_adjust_column_rewrites_the_table_and_drops_the_cache():
    table = _pixel_table()
    assert table.coordinate_nm("x")[1] == 100.0
    table.adjust_column("x_pos_pixels", offset=1.0)
    assert table.records.x_pos_pixels[1] == 2.0
    assert table.coordinate_nm("x")[1] == 200.0


def test_active_coordinates_track_the_mask():
    table = _nm_table()
    assert len(table.active_coordinate_nm("x")) == 10
    table.apply_filters(np.array([0, 9]))
    assert table.active_coordinate_nm("x").tolist() == [0.0, 9.0]


def test_unfiltered_active_coordinates_avoid_a_second_allocation():
    table = _nm_table()
    assert table.active_coordinate_nm("x") is table.coordinate_nm("x")


# ------------------------------------------------------------------- filtering


def test_apply_filters_takes_indices_or_a_mask():
    by_index = _nm_table()
    by_index.apply_filters(np.array([2, 5]))

    by_mask = _nm_table()
    mask = np.zeros(10, dtype=bool)
    mask[[2, 5]] = True
    by_mask.apply_filters(mask)

    assert by_index.active_ids.tolist() == by_mask.active_ids.tolist() == [2, 5]


def test_apply_filters_removes_after_keeping():
    table = _nm_table()
    table.apply_filters(np.arange(5), np.array([[0, 1]]))  # nested, as np.where gives
    assert table.active_ids.tolist() == [2, 3, 4]


def test_non_finite_rows_are_kept_by_a_bandpass():
    """The negated-exclusion form is deliberate; NaN must not be dropped."""
    records = _nm_records()
    records.x_pos_nm[3] = np.nan
    table = LocalizationTable(records)
    table.bandpass("x_pos_nm", 2, 5)
    assert 3 in table.active_ids.tolist()


def test_restrict_by_percent_spans_the_current_extent():
    table = _nm_table()
    table.restrict_by_percent({"x": (0, 50), "y": (0, 100)})
    assert table.active_coordinate_nm("x").tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_restrict_by_percent_skips_axes_the_table_does_not_have():
    table = _nm_table(zdim=False)
    table.restrict_by_percent({"x": (0, 100), "y": (0, 100), "z": (0, 10)})
    assert table.n_active == 10


def test_restrict_by_percent_derives_every_bound_before_applying_any():
    """Otherwise narrowing x would shrink the extent y is measured against."""
    table = _nm_table()
    table.restrict_by_percent({"x": (0, 50), "y": (0, 50)})
    # y spans 0..18, so its half-way bound is 9.0 -> ids 0..4 by y as well.
    assert table.active_ids.tolist() == [0, 1, 2, 3, 4]


def test_keep_values_selects_by_membership():
    table = _nm_table()
    table.keep_values("x_pos_nm", [1.0, 3.0, 8.0])
    assert table.active_ids.tolist() == [1, 3, 8]


def test_deactivate_positions_is_relative_to_the_selection():
    table = _nm_table()
    table.apply_filters(np.array([4, 5, 6]))
    table.deactivate_positions([0, 2])
    assert table.active_ids.tolist() == [5]


def test_bandpass_with_no_bounds_is_refused():
    table = _nm_table()
    with pytest.raises(ValueError):
        table.bandpass("x_pos_nm")


# --------------------------------------------------------------------- budget


def test_limit_active_to_strides_over_the_active_rows():
    table = _nm_table(n=100)
    assert table.limit_active_to(10) == 90
    assert table.active_ids.tolist() == list(range(0, 100, 10))


def test_limit_active_to_is_a_no_op_under_the_limit():
    table = _nm_table()
    assert table.limit_active_to(10) == 0
    assert table.limit_active_to(None) == 0
    assert table.n_active == 10


# ------------------------------------------------ selection vs display limit


def test_the_display_limit_does_not_touch_the_selection():
    """The budget is an accommodation to the GPU, not an edit of the data."""
    table = _nm_table(n=100)
    table.apply_filters(np.arange(50))
    hidden = table.limit_active_to(10)

    assert hidden == 40
    assert table.n_filtered == 50
    assert table.n_active == 10
    assert table.is_display_limited
    assert table.filtered_ids.tolist() == list(range(50))
    assert table.active_ids.tolist() == list(range(0, 50, 5))


def test_an_export_reads_the_selection_not_the_display_subsample():
    table = _nm_table(n=100)
    table.apply_filters(np.arange(50))
    table.limit_active_to(10)

    assert len(table.filtered_records) == 50
    assert len(table.active_records) == 10
    assert len(table.filtered_coordinate_nm("x")) == 50
    assert len(table.active_coordinate_nm("x")) == 10


def test_a_new_selection_clears_the_display_limit():
    table = _nm_table(n=100)
    table.limit_active_to(10)
    assert table.is_display_limited

    table.apply_filters(np.arange(20))
    assert not table.is_display_limited
    assert table.n_active == table.n_filtered == 20


def test_narrowing_the_selection_starts_from_the_selection():
    """Not from the thinned display set, which would compound the two."""
    table = _nm_table(n=100)
    table.limit_active_to(10)
    table.bandpass("x_pos_nm", 0, 19)
    assert table.n_filtered == 20
    assert not table.is_display_limited


def test_a_budget_that_stops_binding_restores_the_full_view():
    table = _nm_table(n=100)
    table.limit_active_to(10)
    assert table.limit_active_to(1000) == 0
    assert not table.is_display_limited
    assert table.n_active == 100


def test_percent_windows_are_measured_against_the_selection():
    """Otherwise the slider would move when the GPU budget changed."""
    unthinned = _nm_table(n=100)
    unthinned.restrict_by_percent({"x": (0, 50), "y": (0, 100)})

    thinned = _nm_table(n=100)
    thinned.limit_active_to(5)
    thinned.restrict_by_percent({"x": (0, 50), "y": (0, 100)})

    assert thinned.filtered_ids.tolist() == unthinned.filtered_ids.tolist()


def test_deselecting_by_position_ignores_the_display_limit():
    table = _nm_table(n=100)
    table.apply_filters(np.arange(10))
    table.limit_active_to(2)
    table.deactivate_positions([0])
    assert table.filtered_ids.tolist() == list(range(1, 10))
