"""The active set is a mask over the canonical table, not a second copy.

Covers P1-01 (no record-array copy on a filter change) and P1-02 (coordinate
columns are converted once and cached) from docs/modernization-review.md.
"""
import numpy as np
import pytest

from napari_storm.localization_dataset_types import (LocalizationDataBaseClass,
                                                     StormDataClass)
from napari_storm.localization_dataset_types.data_formats import \
    storm_data_dtype


def _base_dataset(n=10):
    locs = np.rec.array(
        np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    )
    locs.x_pos_nm = np.arange(n, dtype="f4")
    locs.y_pos_nm = np.arange(n, dtype="f4") * 2
    return LocalizationDataBaseClass(locs, name="base", zdim_present=False)


def _storm_dataset(n=10, pixelsize_nm=100.0):
    locs = np.rec.array(np.zeros(n, dtype=storm_data_dtype))
    locs.x_pos_pixels = np.arange(n, dtype="f4")
    locs.y_pos_pixels = np.arange(n, dtype="f4") * 2
    return StormDataClass(
        locs=locs, name="storm", pixelsize_nm=pixelsize_nm, zdim_present=False
    )


# --------------------------------------------------------------- mask identity


def test_unfiltered_active_rows_share_storage_with_the_canonical_table():
    dataset = _base_dataset()
    # No copy at all while nothing is filtered out: this is the allocation the
    # old np.delete-based reset_filters made on every gesture.
    assert np.shares_memory(dataset.locs_active, dataset.locs_all)
    assert dataset.number_of_active_entries() == dataset.number_of_entries()


def test_filtering_does_not_copy_or_modify_the_canonical_table():
    dataset = _base_dataset()
    canonical = dataset.locs_all
    dataset.apply_filters(np.arange(4), None)

    assert dataset.locs_all is canonical
    assert dataset.number_of_entries() == 10
    assert dataset.number_of_active_entries() == 4
    assert dataset.active_mask.dtype == np.bool_
    assert dataset.active_mask.tolist() == [True] * 4 + [False] * 6


def test_apply_filters_accepts_a_boolean_mask():
    dataset = _base_dataset()
    keep = np.zeros(10, dtype=bool)
    keep[[1, 3, 5]] = True
    dataset.apply_filters(keep, None)
    assert dataset.x_pos_nm.tolist() == [1.0, 3.0, 5.0]


def test_apply_filters_does_not_alias_the_mask_it_was_given():
    dataset = _base_dataset()
    keep = np.ones(10, dtype=bool)
    dataset.apply_filters(keep, np.array([0]))
    # Mutating the caller's array afterwards must not change the active set.
    keep[:] = False
    assert dataset.number_of_active_entries() == 9


def test_active_rows_are_read_only():
    dataset = _base_dataset()
    dataset.apply_filters(np.arange(4), None)
    with pytest.raises(ValueError):
        dataset.locs_active.x_pos_nm[0] = 99.0


def test_locs_active_cannot_be_assigned():
    dataset = _base_dataset()
    with pytest.raises(AttributeError):
        dataset.locs_active = dataset.locs_all.copy()


def test_set_filter_mask_rejects_a_wrong_length_mask():
    dataset = _base_dataset()
    with pytest.raises(ValueError):
        dataset.set_filter_mask(np.ones(3, dtype=bool))


# ------------------------------------------------------------------- caching


def test_coordinate_columns_are_converted_once_and_cached():
    dataset = _storm_dataset()
    first = dataset.x_pos_nm
    assert first is dataset.x_pos_nm
    assert first.dtype == np.float32
    assert first.tolist() == [i * 100.0 for i in range(10)]


def test_filtering_invalidates_the_active_coordinate_cache():
    dataset = _storm_dataset()
    assert len(dataset.x_pos_nm) == 10
    dataset.apply_filters(np.arange(3), None)
    assert dataset.x_pos_nm.tolist() == [0.0, 100.0, 200.0]


def test_changing_the_pixel_size_invalidates_the_cache():
    dataset = _storm_dataset()
    assert dataset.x_pos_nm[1] == 100.0
    dataset.pixelsize_nm = 10.0
    assert dataset.x_pos_nm[1] == 10.0


def test_adjust_column_updates_the_canonical_table_and_the_cache():
    dataset = _storm_dataset()
    assert dataset.x_pos_nm[1] == 100.0
    dataset.adjust_column("x_pos_pixels", offset=5.0)
    assert dataset.locs_all.x_pos_pixels[1] == 6.0
    assert dataset.x_pos_nm[1] == 600.0
    dataset.adjust_column("x_pos_pixels", scale=2.0)
    assert dataset.x_pos_nm[1] == 1200.0


# ------------------------------------------------------------------ filtering


def test_bandpass_narrows_the_existing_active_set():
    dataset = _base_dataset()
    dataset.bandpass_locs_filter_by_property("x_pos_nm", 2, 7)
    assert dataset.x_pos_nm.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    dataset.bandpass_locs_filter_by_property("y_pos_nm", 0, 9)
    assert dataset.x_pos_nm.tolist() == [2.0, 3.0, 4.0]


def test_storm_bandpass_converts_nanometre_bounds_to_pixels():
    dataset = _storm_dataset()
    dataset.bandpass_locs_filter_by_property("x_pos_nm", 200, 400)
    assert dataset.x_pos_nm.tolist() == [200.0, 300.0, 400.0]


def test_restrict_by_percent_matches_the_absolute_bounds_it_implies():
    dataset = _base_dataset()
    dataset.restrict_locs_by_percent([0, 50], [0, 100])
    # x spans 0..9, so the upper half-open bound is 4.5.
    assert dataset.x_pos_nm.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_reset_filters_restores_every_row_without_copying():
    dataset = _base_dataset()
    dataset.apply_filters(np.arange(2), None)
    dataset.reset_filters()
    assert dataset.number_of_active_entries() == 10
    assert np.shares_memory(dataset.locs_active, dataset.locs_all)


def test_remove_locs_by_index_addresses_the_active_set():
    dataset = _base_dataset()
    dataset.apply_filters(np.arange(5), None)
    dataset.remove_locs_by_index([0, 2])
    assert dataset.x_pos_nm.tolist() == [1.0, 3.0, 4.0]


def test_restrict_by_photon_count_keeps_rows_at_the_threshold():
    dataset = _storm_dataset()
    # set_column is the sanctioned writer; reaching into locs_all is refused
    # precisely because it would leave the derived caches stale.
    dataset.set_column("photon_count", np.arange(10, dtype="f4") * 100)
    dataset.restrict_locs_by_photon_count(500)
    assert dataset.number_of_active_entries() == 5
