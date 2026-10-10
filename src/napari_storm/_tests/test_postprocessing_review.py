"""Regression tests for the review of the post-processing integration.

Each test pins a defect the review reproduced, so it cannot come back.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (
    DatasetTraits,
    GaussianSettings,
    LocalizationTable,
    NullRenderer,
    RenderPlanner,
)
from napari_storm.core.renderer import LayerAppearance, RenderRequest
from napari_storm.core.world_transform import WorldTransform
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.postprocessing.drift import DriftModel

from .test_postprocessing import fixture


@pytest.fixture
def dock(make_napari_viewer):
    widget = napari_storm(make_napari_viewer(), renderer=NullRenderer())
    yield widget
    widget.close_session()


def _load(dock, name="a", merge=False):
    dataset = LocalizationDataBaseClass(fixture(), name=name, zdim_present=True)
    dock._apply_loaded_datasets([dataset], merge=merge)
    return dataset


# ---- core ------------------------------------------------------------------


def test_a_shifted_channel_survives_reset_render_range(dock):
    d = _load(dock)
    n = d.table.n_filtered
    dock.dataset_store.set_transform(
        d.dataset_id, WorldTransform(translation_nm=(5000.0, 0.0, 0.0))
    )
    dock.data_to_layer_itf.set_render_range_and_offset()
    dock.data_to_layer_itf.update_layers()
    assert d.table.n_filtered == n


def test_appending_records_with_a_text_field_keeps_selection_and_side_columns():
    records = np.zeros(
        4, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("name", "S4")]
    ).view(np.recarray)
    records["x_pos_nm"] = [0, 1, 2, 3]
    table = LocalizationTable(records)
    table.set_filter_mask(np.array([True, False, True, True]))
    table.set_side_column("group_id", np.array([0, 0, 1, 1]))
    grown = np.concatenate([records, records[:2]]).view(np.recarray)
    table.set_records(grown)
    assert table.filter_mask.tolist() == [True, False, True, True, True, True]
    assert table.column("group_id").tolist() == [0, 0, 1, 1, -1, -1]


def test_writing_a_side_column_writes_it():
    table = LocalizationTable(fixture(5))
    table.set_side_column("group_id", np.zeros(len(table), dtype=np.int64))
    table.adjust_column("group_id", offset=3)
    assert table.column("group_id").tolist() == [3] * len(table)


def test_a_correction_cannot_add_an_axis_to_an_existing_stash():
    table = LocalizationTable(fixture(5))
    table.apply_position_delta(np.ones((len(table), 3)), zdim_present=False)
    with pytest.raises(ValueError, match="axis"):
        table.apply_position_delta(np.ones((len(table), 3)), zdim_present=True)


def test_z_colour_does_not_flip_on_a_mirrored_dataset_with_a_drift():
    table = LocalizationTable(fixture(10))
    planner = RenderPlanner()
    settings = GaussianSettings(z_color_encoding=True)
    traits = DatasetTraits(zdim_present=True)
    mirrored = WorldTransform(scale=(1.0, 1.0, -1.0))
    plain = planner.plan(table, settings, traits, name="d", transform=mirrored)
    offset = planner.plan(
        table,
        settings,
        traits,
        name="d",
        transform=mirrored,
        position_offset=lambda ids: np.zeros((len(ids), 3)),
    )
    np.testing.assert_allclose(offset.values, plain.values)


def test_new_fields_come_last_for_hosts_that_construct_positionally():
    names = [f.name for f in RenderRequest.__dataclass_fields__.values()]
    assert names[-2:] == ["traces", "pairs"]
    names = [f.name for f in LayerAppearance.__dataclass_fields__.values()]
    assert names[-1] == "pairs"


def test_frame_column_is_guarded_whatever_its_name(dock):
    d = _load(dock)
    post = dock.postprocessing
    post.apply(d, DriftModel([0, 99], np.zeros((2, 3))), d.table.column("frame_number"))
    with pytest.raises(ValueError):
        d.table.adjust_column("frame_number", offset=1)


# ---- dock ------------------------------------------------------------------


def test_post_processing_tab_stays_last_after_another_load(dock):
    _load(dock, "a")
    _load(dock, "b", merge=True)
    labels = [dock.tabs.tabText(i) for i in range(dock.tabs.count())]
    assert labels[-1] == "Post-proc."


def test_undone_drift_disables_play_and_transfer(dock):
    d = _load(dock)
    post = dock.postprocessing
    post.apply(
        d,
        DriftModel([0, 99], [[0.0, 0, 0], [10.0, 0, 0]]),
        d.table.column("frame_number"),
    )
    post.transition("undo")
    assert not post.widget.actions["play"].isEnabled()
    assert not post.widget.actions["transfer"].isEnabled()
    assert post.widget.actions["reapply"].isEnabled()


def test_missing_comet_note_does_not_hide_results(dock, monkeypatch):
    import napari_storm.PostProcessing as module

    monkeypatch.setattr(module, "find_spec", lambda name: None)
    d = _load(dock)
    post = dock.postprocessing
    post.apply(
        d,
        DriftModel([0, 99], [[0.0, 0, 0], [10.0, 0, 0]]),
        d.table.column("frame_number"),
    )
    assert post.widget.status.text().startswith("Drift applied")
    assert "napari-storm[comet]" in post.widget.comet_note.text()


def test_a_cancelled_pair_search_is_asked_for_again(dock):
    d = _load(dock)
    post = dock.postprocessing
    post._pair_job = (d.dataset_id, d, "key")
    post.pair_keys[d.dataset_id] = "key"

    class Done:
        busy = False

        def poll(self):
            return None, (0, None, "Cancelled")

        def cancel(self):
            pass

        def close(self):
            pass

    post.pair_runner = Done()
    post.poll()
    assert d.dataset_id not in post.pair_keys


def test_time_view_with_a_nan_row_and_leaving_it_on_close(make_napari_viewer):
    viewer = make_napari_viewer()
    dock = napari_storm(viewer, renderer=NullRenderer())
    try:
        flat = [
            LocalizationDataBaseClass(fixture(nan=True), name="a", zdim_present=False),
            LocalizationDataBaseClass(fixture(), name="b", zdim_present=False),
        ]
        dock._apply_loaded_datasets(flat)
        post = dock.postprocessing
        post.time_view(True)
        assert np.isfinite(post._time_scale[0])
        assert viewer.dims.ndisplay == 3
        dock.unload_dataset(1)
        assert post._time_scale is None
        assert viewer.dims.ndisplay == 2
    finally:
        dock.close_session()


def test_the_embedding_recipe_in_the_docs_runs():
    """docs/post-processing.md's embedding example, run as written."""
    pytest.importorskip("comet")
    from napari_storm.postprocessing.comet_runner import CometParameters, run_comet
    from napari_storm.postprocessing.fixtures import drifting_emitters

    records, truth, _ = drifting_emitters()
    table = LocalizationTable(records)
    renderer, dataset_id = NullRenderer(), 1
    before = records.copy()

    coords = np.column_stack([table.coordinate_nm(a) for a in ("x", "y", "z")])
    frames = table.column("frame_number")
    model = run_comet(coords, frames, CometParameters(window=60))
    table.apply_position_delta(model.evaluate(frames))
    request = RenderPlanner().plan(
        table, GaussianSettings(), DatasetTraits(zdim_present=True), name="corrected"
    )
    renderer.open(dataset_id, request)
    renderer.update(dataset_id, request)
    table.restore_positions()

    assert table.records.tobytes() == before.tobytes()
    residual = model.evaluate(frames) - truth[frames]
    assert np.abs(residual - residual.mean(axis=0)).max() < 1.0


def test_run_progress_follows_comets_schedule():
    from napari_storm.PostProcessing import RunProgress, run_schedule
    from napari_storm.postprocessing.comet_runner import CometParameters

    # measured in phase 0: max drift 120 nm, target 5 nm -> 7 steps; at most 11
    assert run_schedule(120, 5)[1:] == (7, 11)
    assert run_schedule(300, 10)[1:] == (7, 13)
    progress = RunProgress(CometParameters(max_drift_nm=120))
    fractions = []
    for stage, info in [
        ("pairs_done", {"n_pairs": 10}),
        ("run_start", {"run": 1, "sigma_nm": 40}),
        ("run_end", {"run": 1, "sigma_nm": 40}),
        ("run_start", {"run": 2, "sigma_nm": 26.7}),
        ("evaluation", {"run": 2, "sigma_nm": 26.7}),
        ("interpolation", {}),
        ("apply", {}),
    ]:
        fraction, text = progress.update(stage, info)
        fractions.append(fraction)
    assert fractions == sorted(fractions)
    progress.update("run_start", {"run": 3, "sigma_nm": 17.8})
    assert progress.text.startswith("Step 3 of ~5 (≤ 11) · σ 17.8 nm · ETA")


def test_keep_percent_subsamples_reproducibly():
    from napari_storm.postprocessing.comet_runner import CometParameters, subsample
    data = np.column_stack([np.arange(1000.0)] * 4)
    half = subsample(data, CometParameters(keep_percent=50))
    assert len(half) == 500 and np.all(np.diff(half[:, 0]) > 0)
    np.testing.assert_array_equal(half, subsample(data, CometParameters(keep_percent=50)))
    assert subsample(data, CometParameters()) is data
    with pytest.raises(ValueError):
        CometParameters(keep_percent=0)


def test_keep_slider_reaches_the_parameters(dock):
    _load(dock)
    post = dock.postprocessing
    post.widget.keep.setValue(40)
    assert post.parameters().keep_percent == 40
