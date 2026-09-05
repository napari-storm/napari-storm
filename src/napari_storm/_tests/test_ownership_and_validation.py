"""Ownership invariants and rejection of unrenderable scientific input.

Every case here was reproduced against the previous code before being fixed:
external writes through the public `records` property left the derived
coordinate caches stale, a caller could desynchronise a mask from the counts
derived from it, the dataset list could be appended to behind the store's back,
adding the same dataset twice corrupted the id map, and zero-valued sigma or
photon columns reached divisions guarded only by `assert`.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (
    DatasetStore,
    InvalidLocalizationData,
    LocalizationTable,
    require_positive_maximum,
    sanitize_positive,
)
from napari_storm.localization_dataset_types import (
    LocalizationDataBaseClass,
    StormDataClass,
)
from napari_storm.localization_dataset_types.data_formats import storm_data_dtype

PIXEL_COLUMNS = {"x": "x_pos_pixels", "y": "y_pos_pixels", "z": "z_pos_pixels"}


def _pixel_table(n=5, scale=100.0):
    records = np.rec.array(
        np.zeros(n, dtype=[("x_pos_pixels", "f4"), ("y_pos_pixels", "f4")])
    )
    records.x_pos_pixels = np.arange(n, dtype="f4")
    return LocalizationTable(
        records, position_columns=PIXEL_COLUMNS, position_scale_nm=scale
    )


def _dataset(values, name="ds"):
    values = np.asarray(values, dtype="f4")
    locs = np.zeros(len(values), dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = values
    locs["y_pos_nm"] = values
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


# ------------------------------------------------------- table ownership


def test_the_canonical_table_cannot_be_written_through():
    """The write itself was harmless; the silently stale cache was not."""
    table = _pixel_table()
    assert table.coordinate_nm("x")[1] == 100.0
    with pytest.raises(ValueError):
        table.records.x_pos_pixels[1] = 42.0
    assert table.coordinate_nm("x")[1] == 100.0


def test_columns_and_coordinates_are_handed_out_read_only():
    table = _pixel_table()
    for array in (
        table.column("x_pos_pixels"),
        table.coordinate_nm("x"),
        table.filtered_coordinate_nm("x"),
        table.active_coordinate_nm("x"),
        table.filter_mask,
        table.active_mask,
    ):
        assert array.flags.writeable is False


def test_set_column_is_the_sanctioned_replacement_and_invalidates():
    table = _pixel_table()
    assert table.coordinate_nm("x")[1] == 100.0
    table.set_column("x_pos_pixels", np.arange(5, dtype="f4") * 2)
    assert table.coordinate_nm("x")[1] == 200.0


def test_set_column_rejects_the_wrong_length():
    table = _pixel_table()
    with pytest.raises(ValueError):
        table.set_column("x_pos_pixels", np.arange(3, dtype="f4"))


def test_a_mask_is_copied_on_the_way_in():
    """Otherwise the caller can desynchronise the mask from n_filtered."""
    table = _pixel_table()
    mask = np.ones(5, dtype=bool)
    table.set_filter_mask(mask)

    mask[:] = False

    assert table.n_filtered == 5
    assert int(table.filter_mask.sum()) == 5


# ------------------------------------------------------- store ownership


class _Dataset:
    name = "probe"


def test_the_dataset_list_cannot_be_appended_to():
    store = DatasetStore()
    store.add(_Dataset())
    with pytest.raises(AttributeError):
        store.datasets.append(_Dataset())
    assert len(store) == 1


def test_the_dataset_view_still_behaves_like_a_sequence():
    store = DatasetStore()
    first, second = _Dataset(), _Dataset()
    store.add(first)
    store.add(second)

    assert store.datasets == [first, second]
    assert list(store.datasets) == [first, second]
    assert store.datasets[1] is second
    assert len(store.datasets) == 2
    assert first in store.datasets
    assert [d for d in store.datasets] == [first, second]


def test_adding_the_same_dataset_twice_is_refused():
    """It used to give one object two ids and leave the first one dangling."""
    store = DatasetStore()
    dataset = _Dataset()
    first_id = store.add(dataset)

    with pytest.raises(ValueError):
        store.add(dataset)

    assert len(store) == 1
    assert dataset.dataset_id == first_id
    assert store.get(first_id) is dataset

    store.remove(dataset)
    assert len(store) == 0
    assert store.get(first_id) is None


# ---------------------------------------------------------- validation


def test_sanitize_repairs_and_counts():
    values, repaired = sanitize_positive(
        np.array([1.0, 0.0, -2.0, np.nan, 4.0]), "sigma", 0.5
    )
    assert repaired == 3
    assert values.tolist() == [1.0, 0.5, 0.5, 0.5, 4.0]


def test_sanitize_leaves_good_data_untouched():
    original = np.array([1.0, 2.0])
    values, repaired = sanitize_positive(original, "sigma", 0.5)
    assert repaired == 0
    assert values is original


def test_an_entirely_unusable_column_is_refused():
    """Nothing to normalize against; the channel would come out blank."""
    with pytest.raises(InvalidLocalizationData):
        sanitize_positive(np.zeros(4), "photon_count", 0.5)


def test_require_positive_maximum_reports_what_was_wrong():
    require_positive_maximum(np.array([0.0, 2.0]), "values")
    for bad in (np.array([]), np.zeros(3), np.array([np.nan])):
        with pytest.raises(InvalidLocalizationData):
            require_positive_maximum(bad, "values")


def test_non_finite_positions_are_excluded_from_the_selection():
    dataset = _dataset([0.0, np.nan, 2.0, np.inf, 4.0])

    assert dataset.exclude_non_finite_positions() == 2

    assert dataset.number_of_entries() == 5
    assert dataset.number_of_active_entries() == 3
    assert dataset.x_pos_nm.tolist() == [0.0, 2.0, 4.0]


def test_excluding_twice_reports_nothing_the_second_time():
    dataset = _dataset([0.0, np.nan, 2.0])
    assert dataset.exclude_non_finite_positions() == 1
    assert dataset.exclude_non_finite_positions() == 0


# -------------------------------------------------------- integration


def test_loading_data_with_nan_positions_warns_and_draws_the_rest(
    make_napari_viewer,
):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    messages = []
    widget.data_to_layer_itf.on_resource_limit_applied = messages.append

    widget.get_dataset_from_test_mode([_dataset([0.0, np.nan, 2.0, 3.0])])

    dataset = widget.localization_datasets[0]
    assert dataset.number_of_active_entries() == 3
    assert widget.data_to_layer_itf.layer_for(dataset).n_localizations == 3
    assert any("non-finite position" in message for message in messages)


def test_zero_sigmas_are_repaired_and_reported_once(make_napari_viewer):
    """One zero used to turn every render value into NaN via the normalization."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    messages = []
    widget.data_to_layer_itf.on_resource_limit_applied = messages.append

    locs = np.rec.array(np.zeros(4, dtype=storm_data_dtype))
    locs.x_pos_pixels = np.arange(4, dtype="f4")
    locs.y_pos_pixels = np.arange(4, dtype="f4")
    locs.sigma_x_pixels = np.array([1.0, 0.0, 1.0, 1.0], dtype="f4")
    locs.sigma_y_pixels = np.ones(4, dtype="f4")
    dataset = StormDataClass(
        locs=locs, name="zero-sigma", zdim_present=False, sigma_present=True
    )
    widget.get_dataset_from_test_mode([dataset])
    widget.render_config.gaussian_mode = 1
    widget.data_to_layer_itf.update_layer_appearance()

    values = widget.data_to_layer_itf.render_state[dataset.dataset_id].values
    assert np.all(np.isfinite(values))
    assert any("sigma_x_pixels" in message for message in messages)

    # Repeating the update must not repeat the warning.
    before = len(messages)
    widget.data_to_layer_itf.update_layer_appearance()
    assert len(messages) == before


# ------------------------------------------------- §7.2 numeric validation


def test_unparseable_numeric_input_is_refused_rather_than_raising(
    make_napari_viewer,
):
    """§7.2: no numeric control may accept input that cannot be parsed.

    These are live `textChanged` handlers, so half-typed input is normal and an
    exception from a keystroke would surface from the Qt event loop instead of
    reaching the user.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_dataset(np.linspace(0, 5000, 32))])
    good_scalebar = widget.render_config.scalebar_size_nm
    good_distance = widget.grid_plane_line_distance_um

    for rubbish in ("", "-", "abc", "1.2.3", "0", "-5"):
        widget.Esbsize.setText(rubbish)
        widget._sync_scalebar_config()
        widget.Egrid_line_distance.setText(rubbish)
        widget.update_grid_plane_line_distance()

    # The last good values survive; nothing raised.
    assert widget.render_config.scalebar_size_nm == good_scalebar
    assert widget.grid_plane_line_distance_um == good_distance


def test_valid_numeric_input_still_applies(make_napari_viewer):
    """The guard must not be a wall."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_dataset(np.linspace(0, 5000, 32))])

    widget.Esbsize.setText("750")
    widget._sync_scalebar_config()
    widget.Egrid_line_distance.setText("2.5")
    widget.update_grid_plane_line_distance()

    assert widget.render_config.scalebar_size_nm == 750
    assert widget.grid_plane_line_distance_um == 2.5


def test_both_numeric_fields_carry_validators(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    assert widget.Esbsize.validator() is not None
    assert widget.Egrid_line_distance.validator() is not None


# ------------------------------------------- §7.4 the host's UI is the host's


def test_the_plugin_does_not_hide_napari_core_ui(make_napari_viewer, monkeypatch):
    """§7.4 names this, down to the reader's line numbers.

    napari's layer list and layer controls operate on our layers and belong to
    the user, not to us. Opening the dock used to hide both, with no way to ask
    for them back.

    Asserted on the *call* rather than on dock visibility: headless, the main
    window is never shown, so every widget reports itself hidden and a
    visibility check passes whether or not the plugin behaves.
    """
    from napari_storm.napari_particles import _napari_compat

    hidden = []
    monkeypatch.setattr(
        _napari_compat,
        "set_builtin_layer_docks_visible",
        lambda viewer, visible: hidden.append(visible) or True,
    )

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_dataset(np.linspace(0, 5000, 16))])

    assert hidden == [], f"the plugin hid napari's own docks: {hidden}"
