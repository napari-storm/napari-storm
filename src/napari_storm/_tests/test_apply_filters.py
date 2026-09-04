import numpy as np


def test_keep_all(two_d_dataset):
    ds = two_d_dataset
    n = len(ds.locs_all)
    ds.apply_filters(np.arange(n), None)
    assert len(ds.locs_active) == n


def test_keep_all_empty_param_list(two_d_dataset):
    ds = two_d_dataset
    n = len(ds.locs_all)
    ds.apply_filters(np.arange(n), np.array([], dtype=int))
    assert len(ds.locs_active) == n


def test_spatial_filter_keeps_subset(two_d_dataset):
    ds = two_d_dataset
    n = len(ds.locs_all)
    keep = np.arange(n // 2)
    ds.apply_filters(keep, None)
    assert len(ds.locs_active) == n // 2


def test_parameter_filter_removes(two_d_dataset):
    ds = two_d_dataset
    n = len(ds.locs_all)
    ds.apply_filters(np.arange(n), np.array([0, 1]))
    assert len(ds.locs_active) == n - 2


def test_combined_filters(two_d_dataset):
    ds = two_d_dataset
    # Keep only first 5 spatially; then remove index 0 by parameter
    ds.apply_filters(np.arange(5), np.array([0]))
    assert len(ds.locs_active) == 4


def test_reset_restores_all(two_d_dataset):
    ds = two_d_dataset
    n = len(ds.locs_all)
    ds.apply_filters(np.arange(n // 2), None)
    assert len(ds.locs_active) == n // 2
    ds.reset_filters()
    assert len(ds.locs_active) == n


def test_locs_all_unchanged_after_filter(two_d_dataset):
    """apply_filters must not modify locs_all."""
    ds = two_d_dataset
    n = len(ds.locs_all)
    ds.apply_filters(np.arange(n // 2), np.array([0]))
    assert len(ds.locs_all) == n
