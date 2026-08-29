"""The Level 2 state model: bounds, transform, per-dataset state, planning.

All of it host-free except the last section, which checks that the application
is actually driven by it rather than merely carrying it alongside.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (
    EMPTY,
    AppearanceChanged,
    Bounds,
    DatasetState,
    DatasetStore,
    DatasetTraits,
    GaussianSettings,
    LayerAppearance,
    LocalizationTable,
    MaskChanged,
    RenderPlanner,
    TransformChanged,
    WorldTransform,
)


def _table(n=10, zdim=False):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    records = np.rec.array(np.zeros(n, dtype=fields))
    records.x_pos_nm = np.arange(n, dtype="f4")
    records.y_pos_nm = np.arange(n, dtype="f4") * 2
    if zdim:
        records.z_pos_nm = np.arange(n, dtype="f4") * 3
    return LocalizationTable(records)


# ----------------------------------------------------------------- bounds


def test_empty_bounds_are_the_identity_for_union():
    """Which is why accumulating extents needs no special case for the first."""
    assert EMPTY.is_empty
    assert EMPTY.union(Bounds(2.0, 5.0)) == Bounds(2.0, 5.0)
    assert Bounds(2.0, 5.0).union(EMPTY) == Bounds(2.0, 5.0)


def test_bounds_union_and_span():
    merged = Bounds(0.0, 4.0).union(Bounds(-2.0, 1.0))
    assert merged == Bounds(-2.0, 4.0)
    assert merged.span == 6.0
    assert merged.centre == 1.0


def test_bounds_of_an_empty_set_is_empty():
    assert Bounds.of(np.array([])) is EMPTY
    assert EMPTY.span == 0.0
    assert EMPTY.centre is None


def test_percent_maps_onto_the_interval_not_onto_zero():
    """The specialisation this replaced only held while data started at zero."""
    bounds = Bounds(10_000.0, 12_000.0)
    assert bounds.percent_to_absolute([0, 100]).tolist() == [10_000.0, 12_000.0]
    assert bounds.percent_to_absolute([50, 50]).tolist() == [11_000.0, 11_000.0]


def test_percent_on_an_empty_interval_returns_the_percentages():
    assert EMPTY.percent_to_absolute([0, 50]).tolist() == [0.0, 50.0]


# -------------------------------------------------------------- transform


def test_the_identity_transform_returns_its_input_untouched():
    values = np.arange(5, dtype=np.float32)
    transform = WorldTransform()
    assert transform.is_identity
    assert transform.apply_axis("x", values) is values


def test_a_transform_scales_then_translates():
    transform = WorldTransform(scale=(2.0, 1.0, 1.0), translation_nm=(5.0, 0.0, 0.0))
    assert transform.apply_axis("x", np.array([1.0, 2.0])).tolist() == [7.0, 9.0]
    assert transform.apply_axis("y", np.array([1.0])).tolist() == [1.0]


def test_a_transform_round_trips():
    """A world-space bound has to be comparable against untransformed data."""
    transform = WorldTransform(scale=(2.0, 3.0, 1.0), translation_nm=(5.0, -1.0, 0.0))
    values = np.array([1.0, 4.0, 9.0])
    there = transform.apply_axis("x", values)
    assert transform.inverse_axis("x", there).tolist() == pytest.approx(values.tolist())


def test_a_zero_scale_is_refused_rather_than_dividing_by_it():
    transform = WorldTransform(scale=(0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        transform.inverse_axis("x", np.array([1.0]))


def test_replacing_one_axis_leaves_the_others():
    transform = WorldTransform().with_translation(x=10.0).with_scale(z=2.0)
    assert transform.translation_nm == (10.0, 0.0, 0.0)
    assert transform.scale == (1.0, 1.0, 2.0)


def test_an_unknown_axis_is_an_error():
    with pytest.raises(KeyError):
        WorldTransform().apply_axis("w", np.array([1.0]))


# ---------------------------------------------------------- dataset state


def test_appearance_updates_leave_unspecified_fields_alone():
    state = DatasetState(dataset_id=1, name="a")
    state.with_appearance(opacity=0.5)
    state.with_appearance(colormap="red")
    assert state.appearance == LayerAppearance(colormap="red", opacity=0.5)


def test_bounds_are_derived_from_the_table_through_the_transform():
    state = DatasetState(dataset_id=1)
    state.transform = WorldTransform(translation_nm=(1000.0, 0.0, 0.0))
    state.update_bounds_from(_table(), zdim_present=False)

    assert state.bounds_for("x") == Bounds(1000.0, 1009.0)
    assert state.bounds_for("y") == Bounds(0.0, 18.0)
    assert state.bounds_for("z") is EMPTY


# ---------------------------------------------------------------- events


def test_the_store_holds_state_and_announces_changes():
    class _Dataset:
        name = "channel"

    store = DatasetStore()
    seen = []
    dataset = _Dataset()
    dataset_id = store.add(dataset)
    store.subscribe(seen.append)

    store.set_appearance(dataset_id, opacity=0.25)
    store.set_transform(dataset_id, WorldTransform(translation_nm=(5.0, 0, 0)))
    store.notify_mask_changed(dataset_id)

    assert [type(event) for event in seen] == [
        AppearanceChanged,
        TransformChanged,
        MaskChanged,
    ]
    assert seen[0].appearance.opacity == 0.25
    assert store.state(dataset_id).transform.translation_nm == (5.0, 0, 0)
    assert store.state(dataset_id).name == "channel"


def test_state_goes_when_its_dataset_does():
    class _Dataset:
        name = "x"

    store = DatasetStore()
    dataset = _Dataset()
    dataset_id = store.add(dataset)
    assert store.state(dataset_id) is not None
    store.remove(dataset)
    assert store.state(dataset_id) is None


def test_changing_an_absent_dataset_is_an_error():
    store = DatasetStore()
    with pytest.raises(KeyError):
        store.set_appearance(404, opacity=1.0)
    with pytest.raises(KeyError):
        store.set_transform(404, WorldTransform())
    # A mask notification for something absent is simply nothing to announce.
    store.notify_mask_changed(404)


# --------------------------------------------------------------- planner


def test_the_planner_runs_with_no_host():
    planner = RenderPlanner()
    request = planner.plan(
        _table(), GaussianSettings(), DatasetTraits(), name="headless"
    )
    assert request.coords.shape == (10, 3)
    assert request.sigmas.shape == (10, 3)
    assert request.values.shape == (10,)
    assert request.active_ids.tolist() == list(range(10))


def test_two_dimensional_data_is_pinned_to_the_z_plane():
    request = RenderPlanner().plan(
        _table(), GaussianSettings(), DatasetTraits(zdim_present=False), name="flat"
    )
    assert np.all(request.coords[:, 0] == 1.0)


def test_the_planner_writes_coordinates_in_renderer_order():
    """(z, x, y) -- the one place that ordering is applied."""
    request = RenderPlanner().plan(
        _table(zdim=True),
        GaussianSettings(),
        DatasetTraits(zdim_present=True),
        name="volume",
    )
    # (z, y, x): x is the last column, y the middle one.
    assert request.coords[:, 2].tolist() == [float(i) for i in range(10)]
    assert request.coords[:, 1].tolist() == [float(i * 2) for i in range(10)]
    assert request.coords[:, 0].tolist() == [float(i * 3) for i in range(10)]


def test_a_transform_moves_what_is_drawn_and_not_the_data():
    table = _table()
    transform = WorldTransform(translation_nm=(1000.0, 0.0, 0.0))
    request = RenderPlanner().plan(
        table,
        GaussianSettings(),
        DatasetTraits(),
        name="shifted",
        transform=transform,
    )
    assert request.coords[0, 2] == 1000.0
    # The measurements themselves are untouched.
    assert table.coordinate_nm("x")[0] == 0.0


def test_fixed_mode_gives_every_localization_the_same_gaussian():
    sigmas, size = RenderPlanner().sigmas(
        _table().selection(), GaussianSettings(mode=0), DatasetTraits()
    )
    assert np.allclose(sigmas, sigmas[0])
    assert size > 0


def test_the_size_limit_caps_the_billboard():
    request = RenderPlanner().plan(
        _table(), GaussianSettings(), DatasetTraits(), name="capped", size_limit=3.0
    )
    assert request.size == 3.0


def test_variable_mode_without_uncertainty_is_refused():
    from napari_storm.core import InvalidLocalizationData

    with pytest.raises(InvalidLocalizationData):
        RenderPlanner().plan(
            _table(),
            GaussianSettings(mode=1),
            DatasetTraits(sigma_present=False, photon_count_present=False),
            name="unsizeable",
        )


def test_z_colouring_without_z_is_refused():
    from napari_storm.core import InvalidLocalizationData

    with pytest.raises(InvalidLocalizationData):
        RenderPlanner().plan(
            _table(),
            GaussianSettings(z_color_encoding=True),
            DatasetTraits(zdim_present=False),
            name="flat",
        )


# ----------------------------------------------------- driven by the state


def test_a_channel_control_changes_state_and_the_layer_follows(make_napari_viewer):
    """Appearance now outlives the widget that set it."""
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_plain_dataset(range(10))])
    dataset = widget.localization_datasets[0]

    widget.channel[0].Slider_opacity.setValue(30)

    state = widget.dataset_store.state_of(dataset)
    assert state.appearance.opacity == pytest.approx(0.3)
    assert widget.data_to_layer_itf.layer_for(dataset).opacity == pytest.approx(0.3)


def test_announcing_a_mask_change_redraws_only_that_dataset(make_napari_viewer):
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(
        [_plain_dataset(range(20), "a"), _plain_dataset(range(20), "b")]
    )
    changed, untouched = widget.localization_datasets
    interface = widget.data_to_layer_itf
    other_layer = interface.layer_for(untouched)
    other_coords = other_layer.localization_coords.copy()

    # Record a real parameter filter for that dataset, then announce it.
    # MaskChanged means "the selection inputs changed, recompute and redraw",
    # so setting a mask by hand first would simply be discarded.
    filters = widget.data_filter_itf
    filters._record_indices(changed, np.arange(15))
    widget.dataset_store.notify_mask_changed(changed.dataset_id)

    assert interface.layer_for(changed).n_localizations < 20
    assert interface.layer_for(untouched) is other_layer
    assert np.array_equal(other_layer.localization_coords, other_coords)


def test_moving_a_dataset_redraws_it_at_the_new_place(make_napari_viewer):
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([_plain_dataset(range(10))])
    dataset = widget.localization_datasets[0]
    interface = widget.data_to_layer_itf
    before = interface.layer_for(dataset).localization_coords[:, 2].copy()

    widget.dataset_store.set_transform(
        dataset.dataset_id, WorldTransform(translation_nm=(500.0, 0.0, 0.0))
    )

    after = interface.layer_for(dataset).localization_coords[:, 2]
    assert np.allclose(after, before + 500.0)
