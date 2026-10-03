"""Traces arranged as trajectories: the host-free half.

A trace is one molecule localized repeatedly.  Connected in time order through
its own localizations it is a path, and the thing that can silently go wrong
is the order: MINFLUX writes the iterations of concurrent traces interleaved,
and a path drawn through them in file order zig-zags between molecules, while
one drawn through a trace's rows in the wrong order doubles back on itself.
Neither fails a shape check, so the synthetic traces here each walk a straight
line at a fixed step: once correctly ordered every step is that length, and a
mis-sorted trace gives itself away.

The other property is registration: a trajectory has to pass exactly through
the localizations the splat renderer draws, after a world transform, a filter
and the render budget's thinning.  That is checked here against the request's
own coordinates; `test_trace_overlay` checks it again on screen.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.lib.recfunctions import repack_fields

from napari_storm.core import (
    ACTIVE,
    FILTERED,
    DatasetTraits,
    GaussianSettings,
    InvalidLocalizationData,
    LayerAppearance,
    LocalizationTable,
    NullRenderer,
    RenderPlanner,
    WorldTransform,
    find_trace_column,
    is_trace_column,
    plan_traces,
    trace_spread_is_meaningful,
)
from napari_storm.core.render_planner import FLAT_DATA_Z_NM
from napari_storm.core.traces import COLOR_BY_TRACE, arrange_traces

#: Each synthetic trace steps this far, in nanometres, from one localization to
#: the next.
STEP_NM = 10.0

SAMPLE_DATA = Path(__file__).resolve().parents[3] / "sample_data"

DTYPE = [
    ("x_pos_nm", "f4"),
    ("y_pos_nm", "f4"),
    ("z_pos_nm", "f4"),
    ("trace_id", "i4"),
    ("time_s", "f4"),
    ("efo", "f4"),
]


def _interleaved(n_traces=4, n_points=6, z=True):
    """Traces in acquisition order: round-robin, as concurrent ones arrive.

    Round-robin is how concurrent MINFLUX traces are written, and it is the
    case a naive grouping of an unsorted array gets wrong.  Each trace walks
    along its own straight line, STEP_NM a step.
    """
    rows = n_traces * n_points
    records = np.rec.array(np.zeros(rows, dtype=DTYPE))
    step = np.repeat(np.arange(n_points), n_traces)
    trace = np.tile(np.arange(n_traces), n_points)
    records.trace_id = trace
    records.time_s = step * 0.01 + trace * 0.001
    records.x_pos_nm = trace * 1000 + step * STEP_NM * 0.6
    records.y_pos_nm = trace * 1000.0
    records.z_pos_nm = step * STEP_NM * 0.8 if z else 0.0
    records.efo = np.arange(rows, dtype="f4")
    return records


def _plan(records, traits=None, **kwargs):
    traits = traits or DatasetTraits(zdim_present=True)
    return plan_traces(
        LocalizationTable(records), traits, trace_column="trace_id", **kwargs
    )


def _steps_of(vertices, trace):
    coords = vertices.coords[vertices.trace_index == trace].astype(float)
    return np.linalg.norm(np.diff(coords, axis=0), axis=1)


# ------------------------------------------------------------- the ordering


def test_interleaved_traces_come_back_grouped_and_time_ordered():
    vertices = _plan(_interleaved())
    assert np.all(np.diff(vertices.trace_index) >= 0)
    within_one_trace = np.diff(vertices.trace_index) == 0
    assert np.all(np.diff(vertices.time)[within_one_trace] > 0)
    assert vertices.n_traces == 4
    assert vertices.n_segments == 4 * 5


def test_a_sorted_trace_does_not_double_back():
    """The failure a shape check would miss: right rows, wrong order."""
    vertices = _plan(_interleaved())
    for trace in range(vertices.n_traces):
        assert np.allclose(_steps_of(vertices, trace), STEP_NM, rtol=1e-5)


def test_the_time_column_decides_the_order_not_the_rows():
    """Shuffled rows still give straight walks: time is the sort key."""
    records = _interleaved()
    shuffled = records[np.random.default_rng(1).permutation(len(records))]
    vertices = _plan(np.rec.array(shuffled))
    assert vertices.time_column == "time_s"
    for trace in range(vertices.n_traces):
        assert np.allclose(_steps_of(vertices, trace), STEP_NM, rtol=1e-5)


def test_without_a_time_column_each_trace_keeps_its_row_order():
    """Rows are written in acquisition order, so they are the fallback."""
    records = _interleaved()
    without_time = np.rec.array(
        repack_fields(records[["x_pos_nm", "y_pos_nm", "z_pos_nm", "trace_id"]])
    )
    vertices = _plan(without_time)
    assert vertices.time_column is None
    for trace in range(vertices.n_traces):
        assert np.allclose(_steps_of(vertices, trace), STEP_NM, rtol=1e-5)
    # And what it was ordered by is each vertex's place in its trace.
    assert vertices.time[: vertices.lengths[0]].tolist() == [0, 1, 2, 3, 4, 5]


def test_row_order_can_be_asked_for_even_with_a_time_column():
    records = _interleaved()
    records.time_s = records.time_s[::-1]  # time that runs backwards
    vertices = _plan(records, time_column=None)
    assert np.allclose(_steps_of(vertices, 0), STEP_NM, rtol=1e-5)
    assert vertices.localization_ids[0] == 0


def test_ties_in_time_keep_the_row_order():
    """float32 cannot always tell neighbouring MINFLUX iterations apart."""
    records = _interleaved()
    records.time_s = 1234.5  # one time for every row
    vertices = _plan(records)
    for trace in range(vertices.n_traces):
        assert np.allclose(_steps_of(vertices, trace), STEP_NM, rtol=1e-5)


def test_negative_ids_belong_to_no_trace():
    """-1 is "unlinked" to tracking and clustering tools, not one long trace."""
    records = _interleaved(n_traces=3)
    records.trace_id[records.trace_id == 2] = -1
    vertices = _plan(records)
    assert vertices.trace_ids.tolist() == [0, 1]
    assert np.all(records.trace_id[vertices.localization_ids] >= 0)


def test_a_trace_of_one_localization_has_nothing_to_connect():
    records = _interleaved(n_traces=2)
    lone = np.rec.array(np.zeros(1, dtype=DTYPE))
    lone.trace_id = 7
    vertices = _plan(np.rec.array(np.concatenate([records, lone])))
    assert 7 not in vertices.trace_ids
    assert vertices.n_traces == 2


def test_arrange_traces_reports_where_each_trace_starts():
    order, starts = arrange_traces(np.array([5, 3, 5, 3, 3, 9]))
    assert order.tolist() == [1, 3, 4, 0, 2]  # trace 9 has one row: dropped
    assert starts.tolist() == [0, 3]


def test_nothing_selected_is_no_traces_rather_than_an_error():
    records = _interleaved()
    table = LocalizationTable(records)
    table.set_filter_mask(np.zeros(len(table), dtype=bool))
    vertices = plan_traces(table, DatasetTraits(zdim_present=True))
    assert vertices.n_vertices == 0 and vertices.n_traces == 0


# ------------------------------------------------------------ registration


def _same_rows(request):
    """The request's own coordinate for every trace vertex."""
    position = {int(i): k for k, i in enumerate(request.active_ids)}
    rows = [position[int(i)] for i in request.traces.localization_ids]
    return request.coords[rows]


def test_vertices_are_the_coordinates_the_splats_get():
    """Through a world transform and a filter, exactly -- not approximately."""
    table = LocalizationTable(_interleaved(n_traces=5, n_points=8))
    keep = np.ones(len(table), dtype=bool)
    keep[::3] = False
    table.set_filter_mask(keep)
    transform = WorldTransform(
        scale=(1.5, 0.75, 2.0), translation_nm=(5_000.0, -250.0, 40.0)
    )
    request = RenderPlanner().plan(
        table,
        GaussianSettings(),
        DatasetTraits(zdim_present=True),
        name="aligned",
        transform=transform,
        trace_column="trace_id",
    )

    assert request.traces.n_vertices > 0
    assert set(request.traces.localization_ids) <= set(request.active_ids)
    np.testing.assert_array_equal(request.traces.coords, _same_rows(request))
    assert request.traces.coords.dtype == request.coords.dtype


def test_flat_data_sits_on_the_plane_its_splats_are_drawn_on():
    request = RenderPlanner().plan(
        LocalizationTable(_interleaved(z=False)),
        GaussianSettings(),
        DatasetTraits(zdim_present=False),
        name="flat",
        trace_column="trace_id",
    )
    assert np.all(request.traces.coords[:, 0] == FLAT_DATA_Z_NM)
    np.testing.assert_array_equal(request.traces.coords, _same_rows(request))


def test_the_display_limit_thins_traces_with_the_splats():
    """Under the render budget a trace runs through what is drawn."""
    table = LocalizationTable(_interleaved(n_traces=4, n_points=40))
    table.limit_active_to(len(table) // 2)
    vertices = plan_traces(table, DatasetTraits(zdim_present=True), selection=ACTIVE)
    assert set(vertices.localization_ids) <= set(table.active_ids)
    assert vertices.n_vertices <= table.n_active


def test_the_filtered_selection_ignores_the_display_limit():
    table = LocalizationTable(_interleaved(n_traces=4, n_points=40))
    table.limit_active_to(len(table) // 2)
    vertices = plan_traces(table, DatasetTraits(zdim_present=True), selection=FILTERED)
    assert vertices.n_vertices == table.n_filtered


def test_planning_traces_changes_nothing_else_in_the_request():
    """Traces are a visualisation: no width or weight is derived from them."""
    table = LocalizationTable(_interleaved())
    planner = RenderPlanner()
    settings, traits = GaussianSettings(), DatasetTraits(zdim_present=True)
    plain = planner.plan(table, settings, traits, name="a")
    traced = planner.plan(table, settings, traits, name="a", trace_column="trace_id")
    assert plain.traces is None
    for field in ("coords", "sigmas", "values", "active_ids"):
        np.testing.assert_array_equal(getattr(plain, field), getattr(traced, field))
    assert plain.size == traced.size


# ---------------------------------------------------------------- colouring


def test_a_trace_keeps_its_hue_when_a_filter_removes_its_neighbours():
    table = LocalizationTable(_interleaved(n_traces=6))
    traits = DatasetTraits(zdim_present=True)
    before = plan_traces(table, traits)
    hue_of_5 = before.color_values(COLOR_BY_TRACE)[before.vertex_trace_ids == 5]

    table.set_filter_mask(table.column("trace_id") >= 3)
    after = plan_traces(table, traits)

    assert after.trace_ids.tolist() == [3, 4, 5]
    np.testing.assert_array_equal(
        after.color_values(COLOR_BY_TRACE)[after.vertex_trace_ids == 5], hue_of_5
    )
    hues = after.color_values(COLOR_BY_TRACE)
    assert np.all((hues >= 0) & (hues < 1))


def test_neighbouring_traces_get_distant_hues():
    hues = _plan(_interleaved(n_traces=8)).color_values(COLOR_BY_TRACE)
    per_trace = np.unique(np.round(hues, 9))
    gaps = np.abs(np.diff(per_trace))
    assert len(per_trace) == 8 and gaps.min() > 0.05


def test_progress_runs_from_the_first_localization_to_the_last():
    vertices = _plan(_interleaved(n_points=5))
    progress = vertices.color_values("progress")[: vertices.lengths[0]]
    assert progress.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_time_colours_by_the_time_column():
    vertices = _plan(_interleaved())
    np.testing.assert_array_equal(
        vertices.color_values("time"),
        _interleaved().time_s[vertices.localization_ids].astype(float),
    )


def test_a_planned_column_is_reordered_with_the_vertices():
    """A column left in acquisition order would colour the wrong vertex."""
    records = _interleaved()
    vertices = _plan(records, properties=("efo",))
    np.testing.assert_array_equal(
        vertices.color_values("efo"), records.efo[vertices.localization_ids]
    )


def test_an_unplanned_column_says_how_to_plan_it():
    with pytest.raises(KeyError, match="trace_properties"):
        _plan(_interleaved()).color_values("efo")


# --------------------------------------------------------- which columns


def test_any_integer_column_can_identify_traces():
    """Picasso's `group`, a tracker's id: a declared column, any name."""
    records = _interleaved()
    renamed = np.rec.array(
        records.astype(
            [
                ("x_nm", "f4"),
                ("y_nm", "f4"),
                ("z_nm", "f4"),
                ("group", "i4"),
                ("t", "f4"),
                ("efo", "f4"),
            ]
        )
    )
    table = LocalizationTable(
        renamed, position_columns={"x": "x_nm", "y": "y_nm", "z": "z_nm"}
    )
    assert find_trace_column(table) is None  # not offered by default ...
    vertices = plan_traces(
        table,
        DatasetTraits(zdim_present=True),
        trace_column="group",
        time_column="t",
    )  # ... but taken when named
    for trace in range(vertices.n_traces):
        assert np.allclose(_steps_of(vertices, trace), STEP_NM, rtol=1e-5)


def test_whole_numbers_in_a_float_column_identify_traces():
    records = _interleaved()
    as_float = np.rec.array(records.astype([(n, "f8") for n, _ in DTYPE]))
    table = LocalizationTable(as_float)
    assert is_trace_column(table, "trace_id")
    assert plan_traces(table, DatasetTraits(zdim_present=True)).n_traces == 4


@pytest.mark.parametrize("bad", [0.5, np.nan])
def test_a_fraction_or_a_nan_does_not(bad):
    records = np.rec.array(_interleaved().astype([(n, "f8") for n, _ in DTYPE]))
    records.trace_id[3] = bad
    table = LocalizationTable(records)
    assert not is_trace_column(table, "trace_id")
    with pytest.raises(InvalidLocalizationData, match="whole numbers"):
        plan_traces(table, DatasetTraits(), trace_column="trace_id")


def test_a_table_without_traces_is_told_so():
    records = np.rec.array(np.zeros(4, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")]))
    table = LocalizationTable(records)
    assert find_trace_column(table) is None
    with pytest.raises(InvalidLocalizationData, match="no trace column"):
        plan_traces(table, DatasetTraits())
    with pytest.raises(InvalidLocalizationData, match="no trace column 'tid'"):
        plan_traces(table, DatasetTraits(), trace_column="tid")


def test_a_missing_time_column_is_named():
    with pytest.raises(InvalidLocalizationData, match="no time column 'tim'"):
        _plan(_interleaved(), time_column="tim")


# --------------------------------------------------------------- real files


def _minflux_tracking_array(n_traces=5, n_cycles=6, n_itr=4):
    """A MINFLUX v2 export of a tracking run: several cycles per trace.

    The reader keeps one localization per cycle (the final iteration); cycles
    of concurrent traces are interleaved, and each trace walks a straight line
    in metres, as Imspector writes it.
    """
    from .test_minflux_v2 import V2_DTYPE

    rows = n_traces * n_cycles * n_itr
    a = np.zeros(rows, dtype=V2_DTYPE)
    cycle = np.repeat(np.arange(n_cycles), n_traces * n_itr)
    trace = np.tile(np.repeat(np.arange(n_traces), n_itr), n_cycles)
    a["vld"] = True
    a["itr"] = np.tile(np.arange(n_itr), n_traces * n_cycles)
    a["fnl"] = a["itr"] == n_itr - 1
    a["tid"] = trace + 100
    a["tim"] = np.arange(rows) * 1e-4
    a["loc"][:, 0] = (trace * 1000 + cycle * STEP_NM) * 1e-9
    a["loc"][:, 1] = trace * 500e-9
    a["loc"][:, 2] = cycle * 4e-9
    return a


def test_a_minflux_tracking_run_is_read_and_arranged(tmp_path):
    """Through the real reader: metres, the axial correction, interleaving."""
    from napari_storm.localization_dataset_types.minflux_v2 import (
        MinfluxDataV2Class,
    )

    path = tmp_path / "tracking.npy"
    np.save(path, _minflux_tracking_array())
    dataset = MinfluxDataV2Class().load(str(path))
    assert find_trace_column(dataset.table) == "trace_id"

    vertices = plan_traces(dataset.table, DatasetTraits(zdim_present=True))

    assert vertices.trace_ids.tolist() == [100, 101, 102, 103, 104]
    assert vertices.time_column == "time_s"
    assert vertices.lengths.tolist() == [6] * 5
    for trace in range(vertices.n_traces):
        steps = _steps_of(vertices, trace)
        assert np.allclose(steps, steps[0], rtol=1e-4)
        assert steps[0] > STEP_NM  # x and the (corrected) z both advance


@pytest.mark.skipif(not SAMPLE_DATA.is_dir(), reason="sample_data/ not present")
@pytest.mark.parametrize("name", ["imspector_v2.npy", "pyminflux_v2.pmx"])
def test_the_bundled_minflux_samples_have_nothing_to_connect(name):
    """One localization per trace, as a fixed-sample export can be: no lines,
    and no error either."""
    from napari_storm.localization_dataset_types.minflux_v2 import (
        MinfluxDataV2Class,
    )

    dataset = MinfluxDataV2Class().load(str(SAMPLE_DATA / name))
    assert find_trace_column(dataset.table) == "trace_id"
    vertices = plan_traces(dataset.table, DatasetTraits(zdim_present=True))
    assert vertices.n_segments == 0


# -------------------------------------------------------- the renderer side


def test_appearance_validates_the_trace_fields():
    assert LayerAppearance(traces=True, trace_width_px=3).trace_width_px == 3
    with pytest.raises(ValueError, match="trace_width_px"):
        LayerAppearance(trace_width_px=0)
    with pytest.raises(ValueError, match="trace_color_by"):
        LayerAppearance(trace_color_by="")


def test_a_host_can_check_for_traces_without_a_version():
    """The capability checks a host makes, as pyflux does for the palette."""
    from dataclasses import fields

    from napari_storm.core import LocalizationRenderer

    assert {"traces", "trace_color_by", "trace_width_px"} <= {
        f.name for f in fields(LayerAppearance)
    }
    assert hasattr(LocalizationRenderer, "draws_traces")
    assert LocalizationRenderer().draws_traces(1) is False


def test_the_null_renderer_draws_traces_only_when_asked():
    renderer = NullRenderer()
    table = LocalizationTable(_interleaved())
    request = RenderPlanner().plan(
        table,
        GaussianSettings(),
        DatasetTraits(zdim_present=True),
        name="t",
        trace_column="trace_id",
    )
    renderer.open(1, request)
    assert renderer.appearance(1).traces is False
    assert not renderer.draws_traces(1)

    renderer.set_appearance(1, LayerAppearance(traces=True))
    assert renderer.draws_traces(1)
    renderer.set_appearance(1, LayerAppearance(opacity=0.0))
    assert not renderer.draws_traces(1)


def test_spread_estimates_precision_only_when_nothing_is_moving():
    """Not a detail: on real files it is 4 nm fixed against 25 nm tracking,
    and the difference is the molecule, not the instrument."""
    assert trace_spread_is_meaningful(is_tracking=False)
    assert not trace_spread_is_meaningful(is_tracking=True)
