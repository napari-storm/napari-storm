"""Manual alignment controls, and the invariant they must not break.

§7.4 asks for "manual alignment controls with reversible reset and numeric
entry". Until now `WorldTransform` existed as a contract with nothing that
could set a non-identity value, so the capability was untested in practice.

The invariant these must not break is the one §3.5 was about: a transform moves
what is *drawn*, never the measurements, and moving one dataset must not move
another.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


def _dataset(name="ds", n=200, offset=0.0):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(1_000, 3_000, n) + offset
    locs["y_pos_nm"] = np.linspace(1_000, 5_000, n) + offset
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _widget(make_napari_viewer, names=("a",)):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode(
        [_dataset(name, offset=100.0 * i) for i, name in enumerate(names)]
    )
    return widget, viewer


def _controls(widget, index=0):
    return widget.channel[index]


def _drawn_x(widget, dataset):
    # Coordinates are (z, y, x), napari's order, so x is the last column.
    return widget.data_to_layer_itf.layer_for(dataset).localization_coords[:, 2]


# ------------------------------------------------------------- numeric entry


def test_typing_a_shift_moves_what_is_drawn(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    before = _drawn_x(widget, dataset).min()

    _controls(widget)._shift_spins["x"].setValue(2.0)  # µm

    assert _drawn_x(widget, dataset).min() == pytest.approx(before + 2000.0, abs=1.0)


def test_a_shift_does_not_touch_the_measurements(make_napari_viewer):
    """The §3.5 rule: a transform is a view, not an edit."""
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    measured = np.array(dataset.table.coordinate_nm("x"), copy=True)

    _controls(widget)._shift_spins["x"].setValue(5.0)

    assert np.array_equal(dataset.table.coordinate_nm("x"), measured)


def test_each_axis_moves_only_its_own(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    # (z, y, x): y is the middle column, so shifting x must leave it alone.
    y_before = (
        widget.data_to_layer_itf.layer_for(dataset).localization_coords[:, 1].min()
    )

    _controls(widget)._shift_spins["x"].setValue(3.0)

    y_after = (
        widget.data_to_layer_itf.layer_for(dataset).localization_coords[:, 1].min()
    )
    assert y_after == pytest.approx(y_before, abs=1.0)


def test_the_transform_is_recorded_on_the_dataset_state(make_napari_viewer):
    """Not held in the widget: the store owns it, so it can be saved."""
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]

    _controls(widget)._shift_spins["y"].setValue(1.5)

    transform = widget.dataset_store.state(dataset.dataset_id).transform
    assert transform.translation_nm[1] == pytest.approx(1500.0)


# ------------------------------------------------------------------- reset


def test_reset_returns_the_dataset_to_where_it_was_measured(make_napari_viewer):
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    before = _drawn_x(widget, dataset).min()
    controls = _controls(widget)
    controls._shift_spins["x"].setValue(4.0)
    controls._shift_spins["y"].setValue(-2.0)

    controls.reset_shift()

    assert _drawn_x(widget, dataset).min() == pytest.approx(before, abs=1.0)
    assert all(spin.value() == 0.0 for spin in controls._shift_spins.values())


def test_reset_is_reversible_not_destructive(make_napari_viewer):
    """Reset then shift again must work, which a one-way reset would not."""
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    before = _drawn_x(widget, dataset).min()
    controls = _controls(widget)

    controls._shift_spins["x"].setValue(4.0)
    controls.reset_shift()
    controls._shift_spins["x"].setValue(1.0)

    assert _drawn_x(widget, dataset).min() == pytest.approx(before + 1000.0, abs=1.0)


# --------------------------------------------------------------- isolation


def test_moving_one_dataset_does_not_move_another(make_napari_viewer):
    """§7.4's gate, now that something can actually set a transform."""
    widget, _ = _widget(make_napari_viewer, names=("a", "b"))
    first, second = widget.localization_datasets
    other_before = _drawn_x(widget, second).min()

    _controls(widget, 0)._shift_spins["x"].setValue(6.0)

    assert _drawn_x(widget, second).min() == pytest.approx(other_before, abs=1.0)
    assert widget.dataset_store.state(second.dataset_id).transform.translation_nm == (
        0.0,
        0.0,
        0.0,
    )
