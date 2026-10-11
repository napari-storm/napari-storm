"""The complete applied/preview/undone cycle through the dock's planner."""

from threading import Event

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import NullRenderer
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.postprocessing.drift import DriftModel

from .test_postprocessing import fixture


@pytest.fixture
def loaded(make_napari_viewer):
    viewer = make_napari_viewer()
    renderer = NullRenderer()
    dock = napari_storm(viewer, renderer=renderer)
    dataset = LocalizationDataBaseClass(
        fixture(), name="postprocessing", zdim_present=True
    )
    dock._apply_loaded_datasets([dataset])
    yield dock, dataset, renderer
    dock.close_session()


def test_applied_preview_filter_replan_undo_discard(loaded):
    dock, d, renderer = loaded
    post = dock.postprocessing
    before = d.table.records.copy()
    original_ids = d.table.filtered_ids.copy()
    frames = d.table.column("frame_number")
    model = DriftModel([0, 99], [[0.0, 0, 0], [500.0, 50, 10]])
    post.apply(d, model, frames)
    np.testing.assert_array_equal(d.table.filtered_ids, original_ids)
    assert not np.array_equal(d.table.records.x_pos_nm, before.x_pos_nm)
    post.widget.fraction.setValue(6)
    post._apply_preview()
    expected = before.x_pos_nm - 0.6 * model.evaluate(frames)[:, 0]
    np.testing.assert_allclose(
        renderer.requests[d.dataset_id].coords[:, 2], expected, atol=5e-5
    )
    dock.data_to_layer_itf.refresh_dataset(d)
    np.testing.assert_allclose(
        renderer.requests[d.dataset_id].coords[:, 2], expected, atol=5e-5
    )
    post.transition("undo")
    assert d.table.records.tobytes() == before.tobytes()
    with pytest.raises(ValueError):
        d.table.adjust_column("x_pos_nm", offset=1)
    post.transition("reapply")
    post.transition("discard")
    assert d.table.records.tobytes() == before.tobytes()
    assert not d.table.has_position_stash


def test_intentional_crop_stays_absolute(loaded):
    dock, d, _ = loaded
    dtl = dock.data_to_layer_itf
    dock.render_config.range_x_percent = np.array([20.0, 80.0])
    absolute = dtl.percent_to_absolute(
        dtl.render_range_x, dock.render_config.range_x_percent
    )
    dock.postprocessing.apply(
        d, DriftModel([0, 99], [[0, 0, 0], [100, 0, 0]]), d.table.column("frame_number")
    )
    np.testing.assert_allclose(
        dtl.percent_to_absolute(dtl.render_range_x, dock.render_config.range_x_percent),
        absolute,
    )


def test_cancelled_job_unlocks_without_changing_data(loaded, qtbot):
    dock, d, _ = loaded
    post = dock.postprocessing
    before = d.table.records.tobytes()
    gate = Event()

    def work(progress):
        gate.wait(2)
        progress("evaluation", {})
        return None

    post.start(work, lambda _: pytest.fail("Cancelled result was applied"))
    with pytest.raises(ValueError):
        d.table.adjust_column("frame_number", offset=1)
    post.cancel()
    gate.set()
    qtbot.waitUntil(lambda: not post.runner.busy, timeout=5000)
    assert d.table.records.tobytes() == before
    assert not d.table.has_position_stash
    d.table.adjust_column("frame_number", offset=1)


def test_unload_rejects_late_job_result(loaded, qtbot):
    dock, d, _ = loaded
    post = dock.postprocessing
    gate = Event()
    applied = []
    post.start(lambda progress: (gate.wait(2), 1)[1], applied.append)
    dock.clear_datasets()
    gate.set()
    qtbot.waitUntil(lambda: not post.runner.busy, timeout=5000)
    assert applied == []


def test_all_invalid_load_keeps_existing_session(loaded):
    dock, d, _ = loaded
    rows = fixture(3)
    rows.x_pos_nm[:] = np.nan
    invalid = LocalizationDataBaseClass(rows, name="bad", zdim_present=True)
    with pytest.raises(ValueError, match="finite"):
        dock._apply_loaded_datasets([invalid])
    assert list(dock.dataset_store) == [d]


def test_pair_overlay_follows_preview_and_deletion_preserves_traces(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    dock = napari_storm(viewer)
    d = LocalizationDataBaseClass(fixture(20), name="pairs", zdim_present=True)
    dock._apply_loaded_datasets([d])
    post = dock.postprocessing
    post.apply(
        d, DriftModel([0, 19], [[0, 0, 0], [10, 0, 0]]), d.table.column("frame_number")
    )
    dock.Ctraces.setChecked(True)
    post.widget.inputs["radius"].setValue(100)
    post.widget.pairs.setChecked(True)
    renderer = dock.data_to_layer_itf.renderer
    qtbot.waitUntil(lambda: renderer.draws_pairs(d.dataset_id), timeout=10000)
    layer = renderer.pair_layer(d.dataset_id)
    before = layer.data.copy()
    post.widget.fraction.setValue(10)
    post._apply_preview()
    assert not np.array_equal(layer.data, before)
    viewer.layers.remove(layer)
    assert not post.widget.pairs.isChecked()
    assert renderer.draws_traces(d.dataset_id)
    dock.close_session()


def test_time_view_changes_display_only_and_restores_camera(make_napari_viewer):
    viewer = make_napari_viewer()
    renderer = NullRenderer()
    dock = napari_storm(viewer, renderer=renderer)
    d = LocalizationDataBaseClass(fixture(20), name="time", zdim_present=False)
    dock._apply_loaded_datasets([d])
    post = dock.postprocessing
    before = d.table.records.tobytes()
    camera = (viewer.dims.ndisplay, tuple(viewer.camera.center), viewer.camera.zoom)
    post.time_view(True)
    assert viewer.dims.ndisplay == 3
    assert np.ptp(renderer.requests[d.dataset_id].coords[:, 0]) > 0
    assert d.table.records.tobytes() == before
    post.time_view(False)
    assert (
        viewer.dims.ndisplay,
        tuple(viewer.camera.center),
        viewer.camera.zoom,
    ) == camera
    assert np.all(renderer.requests[d.dataset_id].coords[:, 0] == 1)
    dock.close_session()


def test_drift_controls_keep_small_windows_and_fixed_sigma(loaded):
    dock, _, _ = loaded
    post = dock.postprocessing
    assert not {"sigma", "smooth", "cap"} & post.widget.inputs.keys()
    post.widget.inputs["window"].setValue(10)
    assert post.parameters().window == 10
    assert post.parameters().target_sigma_nm == 10
