"""Connecting traces from the dock: the Decorators tab, the lifecycle, a scene.

The Decorators tab offers "Connect traces" wherever the loaded data says
which localizations belong to one molecule -- MINFLUX's tid -- and says why
not where it does not.  Ticked, every such dataset gets an overlay that moves
with its alignment, its filters and its render range, hides with its channel,
and goes with it; a saved scene brings the setting back.

The MINFLUX data here is written in the Imspector >= 24.10 layout and read by
the real reader, so the metres, the axial correction and the interleaving of
concurrent traces are all in the path.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import NO_TRACES_TOOLTIP, napari_storm
from napari_storm.core import (
    DatasetEntry,
    LayerAppearance,
    Scene,
    WorldTransform,
    load_scene,
    save_scene,
)
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.localization_dataset_types.minflux_v2 import MinfluxDataV2Class

from .test_traces import _minflux_tracking_array


def _minflux(tmp_path, name="tracking", zdim=True, n_traces=5, n_cycles=8):
    array = _minflux_tracking_array(n_traces=n_traces, n_cycles=n_cycles)
    # Something to colour by that changes along every trace.
    array["efo"] = np.arange(len(array), dtype="f4")
    if not zdim:
        array["loc"][:, 2] = 0.0
    path = tmp_path / f"{name}.npy"
    np.save(path, array)
    return MinfluxDataV2Class().load(str(path), name=name)


def _plain(name="plain", n=50):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 3000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _dock(make_napari_viewer, datasets=()):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    if datasets:
        widget.get_dataset_from_test_mode(list(datasets))
    return widget, viewer


def _trace_layer(widget, dataset):
    return widget.data_to_layer_itf.trace_layer_for(dataset)


def _choose_colour(widget, name):
    widget.Btrace_color_by.setCurrentIndex(widget.Btrace_color_by.findData(name))


# ---------------------------------------------------------- the Decorators tab


def test_without_trace_ids_the_box_is_off_and_says_why(make_napari_viewer):
    widget, _viewer = _dock(make_napari_viewer)
    assert not widget.Ctraces.isEnabled()
    assert widget.Ctraces.toolTip() == NO_TRACES_TOOLTIP

    widget.get_dataset_from_test_mode([_plain()])
    assert not widget.Ctraces.isEnabled()
    assert "trace id" in widget.Ctraces.toolTip()
    assert not widget.Btrace_color_by.isEnabled()
    assert not widget.Strace_width.isEnabled()


def test_minflux_data_offers_traces_and_its_columns(make_napari_viewer, tmp_path):
    widget, _viewer = _dock(make_napari_viewer, [_minflux(tmp_path)])
    assert widget.Ctraces.isEnabled()
    assert not widget.Ctraces.isChecked()  # opt-in
    offered = [
        widget.Btrace_color_by.itemData(i)
        for i in range(widget.Btrace_color_by.count())
    ]
    assert offered[:3] == ["trace", "time", "progress"]
    assert "efo" in offered
    # The trace id, the time and the positions have better places to be.
    assert not {"trace_id", "time_s", "x_pos_nm"} & set(offered)


def test_ticking_draws_traces_for_the_datasets_that_have_them(
    make_napari_viewer, tmp_path
):
    tracking, plain = _minflux(tmp_path), _plain()
    widget, viewer = _dock(make_napari_viewer, [tracking, plain])
    assert "plain" in widget.Ctraces.toolTip()  # named as left out

    widget.Ctraces.setChecked(True)

    layer = _trace_layer(widget, tracking)
    assert layer in viewer.layers
    assert _trace_layer(widget, plain) is None
    assert widget.data_to_layer_itf.draws_traces(tracking)
    assert widget.dataset_store.state_of(tracking).appearance.traces is True
    assert widget.dataset_store.state_of(plain).appearance.traces is None
    assert widget.Btrace_color_by.isEnabled() and widget.Strace_width.isEnabled()
    # Five traces of eight cycles, every cycle a localization.
    assert len(layer.data) == 40
    assert len(np.unique(layer.data[:, 0])) == 5


def test_colouring_and_width_reach_the_layer(make_napari_viewer, tmp_path):
    tracking = _minflux(tmp_path)
    widget, _viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)
    layer = _trace_layer(widget, tracking)
    first = layer.data[:, 0] == layer.data[0, 0]
    assert len(np.unique(np.asarray(layer.track_colors)[first], axis=0)) == 1

    _choose_colour(widget, "efo")  # a column: planned, then coloured by

    assert len(np.unique(np.asarray(layer.track_colors)[first], axis=0)) == 8
    assert widget.dataset_store.state_of(tracking).appearance.trace_color_by == "efo"

    widget.Strace_width.setValue(5)
    assert layer.tail_width == 5.0


def test_a_dataset_loaded_later_joins(make_napari_viewer, tmp_path):
    widget, _viewer = _dock(make_napari_viewer, [_minflux(tmp_path, "first")])
    widget.Ctraces.setChecked(True)
    second = _minflux(tmp_path, "second")
    widget.get_dataset_from_test_mode([second])
    assert widget.Ctraces.isChecked()
    assert _trace_layer(widget, second) is not None


def _custom_traces(name="custom", n_traces=4, n_points=6):
    """A table of our own with trace ids, a time -- and no `efo`."""
    rows = n_traces * n_points
    locs = np.zeros(
        rows,
        dtype=[
            ("x_pos_nm", "f4"),
            ("y_pos_nm", "f4"),
            ("z_pos_nm", "f4"),
            ("trace_id", "i4"),
            ("time_s", "f4"),
        ],
    )
    locs["trace_id"] = np.tile(np.arange(n_traces), n_points)
    locs["time_s"] = np.arange(rows)
    locs["x_pos_nm"] = np.arange(rows) * 20.0
    locs["y_pos_nm"] = locs["trace_id"] * 300.0
    locs["z_pos_nm"] = np.arange(rows) * 5.0
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=True)


def test_a_column_not_every_dataset_has_falls_back_to_one_hue_per_trace(
    make_napari_viewer, tmp_path, recwarn
):
    tracking = _minflux(tmp_path)
    widget, _viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)
    _choose_colour(widget, "efo")

    custom = _custom_traces()
    widget._apply_loaded_datasets([custom], merge=True)

    assert _trace_layer(widget, custom) is not None
    assert widget.Btrace_color_by.currentData() == "trace"
    assert widget.Btrace_color_by.findData("efo") < 0
    for dataset in (tracking, custom):
        appearance = widget.data_to_layer_itf.renderer.appearance(dataset.dataset_id)
        assert appearance.trace_color_by == "trace"
    assert not [w for w in recwarn if "coloured by" in str(w.message)]


# ----------------------------------------------------------- registration


def _vertex_set(rows):
    return {tuple(row) for row in np.asarray(rows, dtype=np.float64)}


@pytest.mark.parametrize("zdim", [True, False])
def test_trajectories_pass_through_the_drawn_localizations(
    make_napari_viewer, tmp_path, zdim
):
    """After an alignment and a render-range crop, in 2-D and 3-D: every
    vertex of every trajectory is a localization the splats are drawn at."""
    tracking = _minflux(tmp_path, zdim=zdim, n_cycles=20)
    assert tracking.zdim_present is zdim
    widget, viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)

    widget.dataset_store.set_transform(
        tracking.dataset_id,
        WorldTransform(scale=(1.0, 2.0, 1.0), translation_nm=(2_500.0, -700.0, 30.0)),
    )
    widget.render_config.range_x_percent = np.array([10, 75])
    widget.data_to_layer_itf.update_layers()

    splats = widget.data_to_layer_itf.layer_for(tracking).localization_coords
    traces = _trace_layer(widget, tracking).data[:, 2:]
    assert 0 < len(traces) < tracking.number_of_entries()
    assert _vertex_set(traces) <= _vertex_set(splats)
    # Every drawn localization of a trace with two or more drawn is connected.
    drawn_ids = tracking.table.column("trace_id")[tracking.table.active_ids]
    ids, counts = np.unique(drawn_ids, return_counts=True)
    assert len(traces) == counts[counts >= 2].sum()
    # And they really were moved: x by 2.5 um.
    raw_x = tracking.table.coordinate_nm("x")[tracking.table.active_ids]
    assert splats[:, 2].min() == pytest.approx(raw_x.min() + 2_500.0)
    assert viewer.dims.ndisplay == (3 if zdim else 2)


def test_the_traces_follow_the_channel_controls(make_napari_viewer, tmp_path):
    tracking = _minflux(tmp_path)
    widget, _viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)
    itf = widget.data_to_layer_itf

    widget.channel[0].Bshow_channel.setChecked(False)
    assert not itf.draws_traces(tracking)
    widget.channel[0].Bshow_channel.setChecked(True)
    assert itf.draws_traces(tracking)


def test_turning_traces_on_keeps_the_camera(make_napari_viewer, tmp_path):
    widget, viewer = _dock(make_napari_viewer, [_minflux(tmp_path)])
    viewer.camera.angles = (10.0, 20.0, 70.0)
    viewer.camera.zoom = 0.5
    before = (tuple(viewer.camera.center), viewer.camera.zoom, viewer.camera.angles)

    widget.Ctraces.setChecked(True)
    assert (
        tuple(viewer.camera.center),
        viewer.camera.zoom,
        viewer.camera.angles,
    ) == before
    widget.change_camera("XZ")  # the dock's own view buttons still work
    widget.Ctraces.setChecked(False)
    assert viewer.dims.ndim == 3


def test_the_arrow_keys_move_the_overlay_with_the_data(make_napari_viewer, tmp_path):
    """They nudge every layer, and an overlay has a time axis in front."""
    from napari.utils.key_bindings import KeyBinding

    tracking = _minflux(tmp_path)
    widget, viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)

    viewer.keymap[KeyBinding.from_str("Up")](viewer)

    splats = widget.data_to_layer_itf.layer_for(tracking)
    traces = _trace_layer(widget, tracking)
    np.testing.assert_array_equal(splats.translate, [0, -50, 0])
    np.testing.assert_array_equal(traces.translate, [0, 0, -50, 0])


# ---------------------------------------------------------------- lifecycle


def test_unloading_a_dataset_removes_its_traces(make_napari_viewer, tmp_path):
    first, second = _minflux(tmp_path, "first"), _minflux(tmp_path, "second")
    widget, viewer = _dock(make_napari_viewer, [first, second])
    widget.Ctraces.setChecked(True)
    gone = _trace_layer(widget, first)

    widget.unload_dataset(first)

    assert gone not in viewer.layers
    assert _trace_layer(widget, second) in viewer.layers


def test_clearing_and_closing_leave_no_trace_layers(make_napari_viewer, tmp_path):
    widget, viewer = _dock(make_napari_viewer, [_minflux(tmp_path)])
    widget.Ctraces.setChecked(True)
    widget.clear_datasets()
    assert len(viewer.layers) == 0
    assert not widget.Ctraces.isEnabled()

    widget.get_dataset_from_test_mode([_minflux(tmp_path, "again")])
    assert len(viewer.layers) == 2  # still ticked, so drawn again
    widget.close_session()
    assert len(viewer.layers) == 0


def test_toggling_does_not_accumulate_layers(make_napari_viewer, tmp_path):
    widget, viewer = _dock(make_napari_viewer, [_minflux(tmp_path)])
    for _ in range(10):
        widget.Ctraces.setChecked(True)
        widget.Ctraces.setChecked(False)
    assert len(viewer.layers) == 1
    widget.Ctraces.setChecked(True)
    assert len(viewer.layers) == 2


def test_deleting_the_trace_layer_unticks_the_box(make_napari_viewer, tmp_path):
    """Deleting the overlay means "no traces", not "unload the data"."""
    tracking = _minflux(tmp_path)
    widget, viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)

    viewer.layers.remove(_trace_layer(widget, tracking))

    assert widget.localization_datasets[0] is tracking
    assert not widget.Ctraces.isChecked()
    assert widget.dataset_store.state_of(tracking).appearance.traces is False
    widget.data_to_layer_itf.update_layers()
    assert _trace_layer(widget, tracking) is None


def test_deleting_one_of_two_keeps_the_box_ticked(make_napari_viewer, tmp_path):
    """Even while the other's channel is hidden: its traces are still on."""
    first, second = _minflux(tmp_path, "first"), _minflux(tmp_path, "second")
    widget, viewer = _dock(make_napari_viewer, [first, second])
    widget.Ctraces.setChecked(True)
    widget.channel[1].Bshow_channel.setChecked(False)

    viewer.layers.remove(_trace_layer(widget, first))

    assert widget.Ctraces.isChecked()
    widget.channel[1].Bshow_channel.setChecked(True)
    assert widget.data_to_layer_itf.draws_traces(second)
    assert not widget.data_to_layer_itf.draws_traces(first)


# -------------------------------------------------------------------- scene


def test_the_scene_format_round_trips_trace_settings(tmp_path):
    appearance = LayerAppearance(traces=True, trace_color_by="time", trace_width_px=3)
    path = tmp_path / "scene.json"
    save_scene(path, Scene(datasets=(DatasetEntry("a", appearance=appearance),)))
    restored = load_scene(path).datasets[0].appearance
    assert restored.traces is True
    assert restored.trace_color_by == "time"
    assert restored.trace_width_px == 3.0


def test_a_scene_from_before_traces_says_nothing_about_them(tmp_path):
    path = tmp_path / "old.json"
    save_scene(path, Scene(datasets=(DatasetEntry("a"),)))
    assert "trace" not in path.read_text()
    restored = load_scene(path).datasets[0].appearance
    assert restored.traces is None and restored.trace_width_px is None


def test_a_saved_scene_brings_the_traces_back(make_napari_viewer, tmp_path):
    tracking = _minflux(tmp_path)
    widget, _viewer = _dock(make_napari_viewer, [tracking])
    widget.Ctraces.setChecked(True)
    _choose_colour(widget, "progress")
    widget.Strace_width.setValue(4)
    path = tmp_path / "scene.json"
    widget.save_scene_to(path)

    widget.Ctraces.setChecked(False)
    _choose_colour(widget, "trace")
    widget.Strace_width.setValue(1)
    widget.load_scene_from(path)

    assert widget.Ctraces.isChecked()
    assert widget.Btrace_color_by.currentData() == "progress"
    assert widget.Strace_width.value() == 4
    appearance = widget.data_to_layer_itf.renderer.appearance(tracking.dataset_id)
    assert (appearance.traces, appearance.trace_color_by) == (True, "progress")
    assert _trace_layer(widget, tracking).tail_width == 4.0
