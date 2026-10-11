"""Data integrity and mathematical contracts of post-processing."""

import time
from threading import Event

import numpy as np
import pytest

from napari_storm.core import (
    DatasetTraits,
    GaussianSettings,
    LocalizationTable,
    RenderPlanner,
)
from napari_storm.core.localization_table import ExclusionReason
from napari_storm.core.pairs import PairSet
from napari_storm.core.world_transform import WorldTransform
from napari_storm.postprocessing.comet_runner import (
    CometParameters,
    prepare_input,
    run_comet,
)
from napari_storm.postprocessing.drift import DriftModel
from napari_storm.postprocessing.grouping import group_means, link_localizations
from napari_storm.postprocessing.jobs import JobRunner
from napari_storm.postprocessing.pair_network import find_pairs


def fixture(n=100, nan=False):
    records = np.zeros(
        n,
        dtype=[
            ("x_pos_nm", "f4"),
            ("y_pos_nm", "f4"),
            ("z_pos_nm", "f4"),
            ("frame_number", "i4"),
            ("trace_id", "i4"),
        ],
    ).view(np.recarray)
    rng = np.random.default_rng(8)
    for name in ("x_pos_nm", "y_pos_nm", "z_pos_nm"):
        records[name] = rng.normal(100, 10, n)
    records.frame_number = np.arange(n)
    records.trace_id = np.arange(n) // 2
    if nan:
        records.x_pos_nm[-1] = np.nan
    return records


def test_exclusion_restore_preserves_user_filters_and_nonfinite():
    table = LocalizationTable(fixture(8, nan=True))
    mask = np.arange(8) >= 5
    table.set_excluded(mask)
    table.keep_values("frame_number", [1, 5, 6, 7])
    assert table.filtered_ids.tolist() == [1]
    table.set_excluded(mask, exclude=False)
    assert table.filtered_ids.tolist() == [1, 5, 6]
    assert table.excluded[-1] == ExclusionReason.NON_FINITE
    table.reset()
    assert table.filtered_ids.tolist() == list(range(7))


def test_stash_undo_reapply_guards_and_discard_are_exact():
    table = LocalizationTable(fixture())
    before = table.records.copy()
    delta = np.tile([10.123, -5.0, 2.0], (len(table), 1))
    table.apply_position_delta(delta)
    corrected = table.records.copy()
    for name in ("x_pos_nm", "frame_number", "trace_id"):
        with pytest.raises(ValueError, match="locked"):
            table.adjust_column(name, offset=1)
    table.restore_positions()
    assert table.records.tobytes() == before.tobytes()
    with pytest.raises(ValueError):
        table.set_records(before)
    with pytest.raises(ValueError):
        table.position_scale_nm = 2
    table.apply_position_delta(delta)
    assert table.records.tobytes() == corrected.tobytes()
    table.discard_position_stash()
    table.adjust_column("x_pos_nm", offset=1)
    np.testing.assert_array_equal(table.column("x_pos_nm"), before.x_pos_nm + 1)


def test_2d_stash_leaves_sentinel_and_bad_apply_is_atomic():
    table = LocalizationTable(fixture())
    z = table.column("z_pos_nm").copy()
    table.apply_position_delta(np.ones((len(table), 3)), zdim_present=False)
    np.testing.assert_array_equal(table.column("z_pos_nm"), z)
    before = table.records.tobytes()
    with pytest.raises(ValueError):
        table.apply_position_delta(np.full((len(table), 3), np.nan))
    assert table.records.tobytes() == before


def test_clamped_anchored_drift_and_save_roundtrip(tmp_path):
    model = DriftModel(
        [10, 20, 30],
        [[1, 2, 3], [3, 4, 5], [5, 6, 7]],
        clock="trace",
        rank_times=np.linspace(0, 100, 31),
    )
    np.testing.assert_allclose(model.evaluate([-100, 10]), 0)
    np.testing.assert_allclose(model.evaluate([30, 100]), [[4, 4, 4]] * 2)
    np.testing.assert_allclose(
        model.evaluate_time(model.rank_times), model.evaluate(np.arange(31))
    )
    path = tmp_path / "drift.h5"
    model.save(path)
    np.testing.assert_array_equal(
        DriftModel.load(path).evaluate([0, 15, 50]), model.evaluate([0, 15, 50])
    )
    with pytest.raises(ValueError, match="strictly"):
        DriftModel([0, 2], np.zeros((2, 3)), clock="trace", rank_times=[1.0, 1.0])


def test_offsets_keep_pairs_traces_and_exports_registered():
    table = LocalizationTable(fixture(10))
    planner = RenderPlanner()
    settings, traits = GaussianSettings(), DatasetTraits(zdim_present=True)
    pairs = PairSet(np.array([[0, 5], [2, 7]]), np.array([5.0, 5.0]))
    kwargs = dict(
        name="test",
        transform=WorldTransform(scale=(2.0, 3.0, 4.0)),
        trace_column="trace_id",
        pairs=pairs,
    )
    baseline = planner.plan(table, settings, traits, **kwargs)
    preview = planner.plan(
        table,
        settings,
        traits,
        position_offset=lambda ids: np.tile([1.0, 2.0, 3.0], (len(ids), 1)),
        **kwargs,
    )
    np.testing.assert_allclose(
        preview.coords - baseline.coords, np.tile([12.0, 6.0, 2.0], (10, 1)), atol=1e-4
    )
    np.testing.assert_allclose(
        preview.pairs.coords, preview.coords[preview.pairs.localization_ids]
    )
    np.testing.assert_allclose(
        preview.traces.coords, preview.coords[preview.traces.localization_ids]
    )
    np.testing.assert_array_equal(
        planner.plan(table, settings, traits, **kwargs).coords, baseline.coords
    )


def test_input_keeps_all_rows_and_original_frames():
    times = np.array([1, 2, 3, 4, 100, 101, 102, 103])
    coords = np.arange(24).reshape(8, 3)
    data = prepare_input(coords, times, CometParameters(window=2))
    np.testing.assert_array_equal(data[:, :3], coords)
    np.testing.assert_array_equal(data[:, 3], times)


def test_grouping_dark_frames_and_split_at_window_boundary():
    coords = np.zeros((4, 3))
    frames = np.array([0, 2, 9, 10])
    groups = link_localizations(coords, frames, max_dark_frames=1)
    assert groups.tolist() == [0, 0, 1, 1]
    means, times = group_means(coords, frames, groups, window=10)
    assert times.tolist() == [1, 9, 10]


def test_pair_network_complete_small_and_bounded_dense():
    coords = np.zeros((20, 3))
    coords[:, 0] = np.arange(20)
    times = np.arange(20)
    windows = times // 5
    pairs = find_pairs(coords, times, windows, np.arange(20), 3.0, cap=100)
    expected = [
        (i, j)
        for i in range(20)
        for j in range(i + 1, 20)
        if j - i <= 3 and windows[i] != windows[j]
    ]
    assert set(map(tuple, pairs.row_ids)) == set(expected)
    bounded = find_pairs(coords, times, windows, np.arange(20), 30.0, cap=10)
    assert len(bounded.row_ids) <= 10


def test_worker_discards_superseded_result_and_keeps_only_latest():
    runner = JobRunner()
    gate = Event()
    started = Event()

    def first(progress):
        started.set()
        gate.wait(2)
        return 1

    runner.submit(first)
    assert started.wait(2)
    runner.submit(lambda p: 2)
    generation = runner.submit(lambda p: 3)
    gate.set()
    result = None
    deadline = time.monotonic() + 5
    while runner.busy and time.monotonic() < deadline:
        _, value = runner.poll()
        if value:
            result = value
        time.sleep(0.005)
    runner.close()
    assert result == (generation, 3, None)


def test_real_comet_recovers_known_drift_without_mutating_input():
    pytest.importorskip("comet")
    rng = np.random.default_rng(5)
    n = 4000
    emitters = rng.normal(0, 100, (50, 3))
    frames = rng.integers(0, 200, n)
    truth = np.column_stack([frames * 0.1, frames * -0.04, frames * 0.03])
    coords = emitters[rng.integers(0, 50, n)] + truth + rng.normal(0, 1, (n, 3))
    before = coords.copy()
    model = run_comet(coords, frames, CometParameters(window=400, max_drift_nm=30))
    # Every localization, not only the knots: the ends (first and last half
    # window) are where clamping at the knots, or knots at geometric window
    # centres, would show. Drift is defined up to a constant, so compare
    # offset-free; the anchor is the start of the acquisition.
    estimate = model.evaluate(frames)
    residual = estimate - (truth - truth.mean(axis=0))
    residual -= residual.mean(axis=0)
    assert np.abs(residual).max() < 1.0
    assert model.frame_range == (float(frames.min()), float(frames.max()))
    np.testing.assert_allclose(model.evaluate([frames.min()]), 0, atol=1e-12)
    np.testing.assert_array_equal(coords, before)
    # Compare the adapter to the tested COMET entry point, before anchoring.
    # This catches changes to segmentation, smoothing and optimization defaults.
    import comet

    _, native = comet.comet_run_kd(
        np.column_stack((coords, frames)),
        segmentation_mode=1,
        segmentation_var=400,
        max_drift_nm=30,
        target_sigma_nm=10,
        mode="cpu",
        display=False,
        interactive=False,
        return_details=True,
    )
    np.testing.assert_array_equal(model.knots, native.knot_frames)
    np.testing.assert_allclose(model.displacements_nm, native.knot_drift_nm, atol=1e-12)


def test_ns_export_drops_exclusions_and_keeps_applied_state(tmp_path):
    import h5py

    from napari_storm.core.dataset_state import DriftView
    from napari_storm.localization_dataset_types import LocalizationDataBaseClass
    from napari_storm.postprocessing.io import save_localizations

    d = LocalizationDataBaseClass(fixture(10), name="export", zdim_present=True)
    model = DriftModel([0, 9], [[0, 0, 0], [10, 0, 0]])
    times = d.table.column("frame_number")
    d.table.apply_position_delta(model.evaluate(times))
    d.table.set_excluded(np.arange(10) == 3)
    state = DriftView(model, times, alpha=0.6)
    path = save_localizations(tmp_path / "sample.with.dots.ns", d, state)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(
            f["dataset"][:], d.table.records[d.table.excluded == 0]
        )
        assert f["dataset"].attrs["drift_applied"]
        assert f["postprocessing"].attrs["clock"] == "frame"
        np.testing.assert_array_equal(
            f["postprocessing/frame_range"][:], model.frame_range
        )


def test_live_append_preserves_selections_and_exclusions():
    records = fixture(10)
    table = LocalizationTable(records[:5])
    table.set_excluded(np.arange(5) == 1)
    table.keep_values("frame_number", [1, 2])
    table.set_records(records)
    assert table.filtered_ids.tolist() == [2, 5, 6, 7, 8, 9]
    table.set_excluded(np.arange(10) == 1, exclude=False)
    assert table.filtered_ids.tolist() == [1, 2, 5, 6, 7, 8, 9]


def test_ten_localizations_per_window_uses_native_comet(monkeypatch):
    comet = pytest.importorskip("comet")
    from types import SimpleNamespace

    from comet.core.segmenter import segment_by_num_locs_per_window

    times = np.repeat(np.arange(20), 10)
    coords = np.zeros((len(times), 3))
    native = segment_by_num_locs_per_window(times, 10)
    assert native.n_segments == 20

    def run(data, **kwargs):
        assert kwargs["segmentation_mode"] == 1
        assert kwargs["segmentation_var"] == 10
        assert kwargs["target_sigma_nm"] == 10
        assert kwargs["mode"] == "cpu"
        assert (
            not {"initial_sigma_nm", "boxcar_width", "max_locs_per_segment"}
            & kwargs.keys()
        )
        np.testing.assert_array_equal(data[:, 3], times)
        return None, SimpleNamespace(
            knot_frames=native.center_frames,
            knot_drift_nm=np.zeros((20, 3)),
            n_pairs=19900,
            backend="cpu",
            n_evaluations=1,
            timings_s={},
            sigma_accepted_nm=10,
            n_runs=1,
            auto_downsampled=False,
        )

    monkeypatch.setattr(comet, "comet_run_kd", run)
    model = run_comet(coords, times, CometParameters(window=10))
    np.testing.assert_array_equal(model.knots, np.arange(20))


def test_drift_is_held_flat_outside_the_data_not_outside_the_knots():
    model = DriftModel(
        [10, 20, 30, 40],
        [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]],
        frame_range=(5, 45),
    )
    # between the data's first frame and the first knot the spline runs on, as COMET's does
    assert model.evaluate([5])[0, 0] < model.evaluate([10])[0, 0]
    assert model.evaluate([0])[0, 0] == model.evaluate([5])[0, 0]
    assert model.evaluate([50])[0, 0] == model.evaluate([45])[0, 0]
    with pytest.raises(ValueError, match="enclose"):
        DriftModel([10, 20], np.zeros((2, 3)), frame_range=(12, 30))


def test_pair_network_keeps_the_extreme_points():
    rng = np.random.default_rng(4)
    coords = np.column_stack(
        [rng.random(3000) * 1000, rng.random(3000) * 1000, np.zeros(3000)]
    )
    times = np.arange(3000)
    windows = times // 100
    result = find_pairs(coords, times, windows, np.arange(3000), 40.0)
    from scipy.spatial import cKDTree

    expected = {
        (i, j) for i, j in cKDTree(coords).query_pairs(40.0) if windows[i] != windows[j]
    }
    assert {tuple(sorted(e)) for e in result.row_ids.tolist()} == expected


def test_memory_budget_includes_dense_frame_interpolation():
    from napari_storm.postprocessing.pair_budget import estimate_memory

    coords = np.zeros((2, 3))
    short = estimate_memory(coords, 10, frame_count=10)
    long = estimate_memory(coords, 10, frame_count=10_000_000)
    assert long.pairs == short.pairs
    assert long.additional_bytes - short.additional_bytes == (10_000_000 - 10) * 128


@pytest.mark.parametrize("window", [0, -1, 1.5, np.inf, np.nan])
def test_window_count_must_be_a_positive_integer(window):
    with pytest.raises(ValueError, match="positive integer"):
        CometParameters(window=window)
