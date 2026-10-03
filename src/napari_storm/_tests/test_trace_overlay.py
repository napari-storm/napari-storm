"""Traces drawn over their localizations: the napari half, on every backend.

`test_traces` checks which localizations are connected and in which order.
This checks what reaches the viewer: a Tracks layer per dataset that asks for
one, with every vertex shown -- no time slider to move first, which is how the
pyMINFLUX prototype of this looked empty when it opened -- lying exactly on the
splats it connects, following the dataset's visibility and opacity, and gone
when the dataset is.  A Tracks layer carries a time axis, and adding one makes
napari refit the camera; the view a user set up must survive both.

Rendering reads pixels from `_scene_canvas.render()`.  A viewer that is never
shown renders its frame off-centre, so what was drawn is located in the image
rather than assumed at its middle.
"""

import numpy as np
import pytest
from napari.layers import Tracks
from scipy import ndimage

from napari_storm.core import (
    DatasetTraits,
    GaussianSettings,
    LayerAppearance,
    LocalizationTable,
    RenderPlanner,
    WorldTransform,
)
from napari_storm.core.renderer import Changed
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.points_renderer import NapariPointsRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer
from napari_storm.napari_particles.trace_overlay import (
    MAX_LAYER_TRACE_ID,
    TRACE_LAYER_SUFFIX,
    tracks_data,
)

BACKENDS = [InstancedRenderer, NapariParticlesRenderer, NapariPointsRenderer]

DTYPE = [
    ("x_pos_nm", "f4"),
    ("y_pos_nm", "f4"),
    ("z_pos_nm", "f4"),
    ("trace_id", "i4"),
    ("time_s", "f4"),
]


def _walks(n_traces=6, n_points=12, seed=0, zdim=True):
    """Interleaved random walks, as concurrent MINFLUX traces are written."""
    rng = np.random.default_rng(seed)
    rows = n_traces * n_points
    records = np.rec.array(np.zeros(rows, dtype=DTYPE))
    start = rng.uniform(1_000, 9_000, (n_traces, 3))
    steps = rng.normal(0, 80, (n_points, n_traces, 3)).cumsum(axis=0)
    positions = (start[None] + steps).reshape(-1, 3)
    records.x_pos_nm = positions[:, 0]
    records.y_pos_nm = positions[:, 1]
    records.z_pos_nm = positions[:, 2] / 10 if zdim else 0.0
    records.trace_id = np.tile(np.arange(n_traces), n_points)
    records.time_s = np.arange(rows) * 1e-3
    return records


def _plan(table, zdim=True, transform=None, changed=Changed.EVERYTHING, **kw):
    return RenderPlanner().plan(
        table,
        GaussianSettings(fixed_sigma_xy_nm=20.0, fixed_sigma_z_nm=20.0),
        DatasetTraits(zdim_present=zdim),
        name="walks",
        colormap="gray",
        transform=transform or WorldTransform(),
        changed=changed,
        trace_column="trace_id",
        **kw,
    )


def _open(make_napari_viewer, backend_class, table=None, zdim=True, **kw):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    table = table or LocalizationTable(_walks(zdim=zdim))
    request = _plan(table, zdim=zdim, **kw)
    renderer.open(1, request)
    viewer.dims.ndisplay = 3 if zdim else 2
    return viewer, renderer, table, request


def _centres(renderer, dataset_id=1):
    """The localization centres a backend is drawing, one row each."""
    layer = renderer.layer(dataset_id)
    centres = getattr(layer, "localization_coords", None)
    return np.asarray(layer.data if centres is None else centres)


def _image(viewer):
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    return np.asarray(canvas.render())[..., :3].astype(np.int16)


# ---------------------------------------------------------- the contract


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_dataset_opens_without_traces_and_draws_them_when_asked(
    make_napari_viewer, backend_class
):
    viewer, renderer, _table, request = _open(make_napari_viewer, backend_class)
    assert renderer.appearance(1).traces is False
    assert renderer.trace_layer(1) is None
    assert not renderer.draws_traces(1)

    renderer.set_appearance(1, LayerAppearance(traces=True))

    layer = renderer.trace_layer(1)
    assert isinstance(layer, Tracks)
    assert layer in viewer.layers
    assert layer.name == "walks" + TRACE_LAYER_SUFFIX
    assert renderer.draws_traces(1)
    assert renderer.appearance(1).traces is True
    assert len(layer.data) == request.traces.n_vertices


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_trace_vertices_are_the_splat_centres(make_napari_viewer, backend_class):
    """After a world transform and a filter: the same numbers, exactly."""
    table = LocalizationTable(_walks())
    keep = np.ones(len(table), dtype=bool)
    keep[::4] = False
    table.set_filter_mask(keep)
    transform = WorldTransform(
        scale=(1.2, 0.8, 2.0), translation_nm=(3_000.0, -500.0, 25.0)
    )
    _viewer, renderer, _table, request = _open(
        make_napari_viewer, backend_class, table=table, transform=transform
    )
    renderer.set_appearance(1, LayerAppearance(traces=True))

    by_id = dict(zip(request.active_ids.tolist(), _centres(renderer)))
    vertices = renderer.trace_layer(1).data
    expected = np.array([by_id[i] for i in request.traces.localization_ids])
    # napari sorts a Tracks layer's rows by (id, time); every vertex has the
    # same time, so its order within a trace has to be the one we gave it.
    np.testing.assert_array_equal(vertices[:, 2:], expected.astype(np.float64))


def test_tracks_data_is_napari_layout_with_one_time_point():
    vertices = _plan(LocalizationTable(_walks())).traces
    data = tracks_data(vertices)
    assert data.shape == (vertices.n_vertices, 5)  # ID, T, Z, Y, X
    assert np.all(data[:, 1] == 0)
    assert data[:, 0].tolist() == vertices.vertex_trace_ids.tolist()


def test_ids_too_large_for_napari_are_renumbered():
    """napari allocates a lookup row for every id up to the largest."""
    records = _walks()
    records.trace_id = records.trace_id + MAX_LAYER_TRACE_ID
    vertices = _plan(LocalizationTable(records)).traces
    assert tracks_data(vertices)[:, 0].max() == vertices.n_traces - 1


# ------------------------------------------------- everything, at once


def test_every_vertex_is_shown_without_moving_a_slider(make_napari_viewer):
    """The prototype's Tracks layer opened empty until the time slider moved."""
    viewer, renderer, _table, _request = _open(make_napari_viewer, InstancedRenderer)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)

    # One time point, so napari has nothing to slide through ...
    assert viewer.dims.ndim == 4
    assert viewer.dims.nsteps[0] == 1
    # ... and every vertex at it, so none is hidden or faded.
    assert layer.current_time == 0
    assert np.all(layer.track_times <= layer.current_time + layer.head_length)
    assert layer.track_connex.sum() == len(layer.data) - len(
        np.unique(layer.data[:, 0])
    )


def _blobs(image, threshold=60):
    """Centroids of the bright blobs in *image*, brightest first."""
    labels, n = ndimage.label(image.max(axis=2) > threshold)
    centres = ndimage.center_of_mass(np.ones_like(labels), labels, range(1, n + 1))
    return np.array(centres)


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_a_drawn_trace_runs_from_its_first_splat_to_its_last(
    make_napari_viewer, ndisplay
):
    """On screen: the line starts and ends on the localizations it connects."""
    records = np.rec.array(np.zeros(2, dtype=DTYPE))
    records.x_pos_nm = [2_000.0, 6_000.0]
    records.y_pos_nm = [3_000.0, 5_000.0]
    records.z_pos_nm = [0.0, 400.0]
    records.time_s = [0.0, 1.0]
    zdim = ndisplay == 3
    viewer, renderer, _table, _request = _open(
        make_napari_viewer,
        InstancedRenderer,
        table=LocalizationTable(records),
        zdim=zdim,
    )
    # Green, so the first trace's hue -- red -- stands out even on a splat.
    renderer.set_appearance(1, LayerAppearance(colormap="green"))
    if zdim:
        viewer.camera.angles = (5.0, 25.0, 80.0)
    viewer.camera.center = (200.0, 4_000.0, 4_000.0)
    viewer.camera.zoom = 0.08
    splats = _image(viewer)
    renderer.set_appearance(
        1, LayerAppearance(traces=True, trace_width_px=2.0, opacity=1.0)
    )
    both = _image(viewer)

    ends = _blobs(splats)
    assert len(ends) == 2, "expected the two splats as two blobs"
    line = np.argwhere(np.abs(both - splats)[..., 0] > 40)
    assert len(line) > 50

    a, b = ends
    along = b - a
    length = np.linalg.norm(along)
    unit = along / length
    offsets = line - a
    position = offsets @ unit
    distance = np.abs(offsets @ np.array([-unit[1], unit[0]]))
    scale = viewer.window._qt_viewer.canvas._scene_canvas.pixel_scale
    # Every line pixel lies on the segment between the two splat centres ...
    assert distance.max() <= 3 * scale + 2
    # ... and the line covers all of it, from one centre to the other.
    assert position.min() <= 2 * scale + 2
    assert position.max() >= length - 2 * scale - 2


# ------------------------------------------ the view the user set up


def test_turning_traces_on_and_off_leaves_the_view_alone(make_napari_viewer):
    """A new axis makes napari refit the camera; the overlay undoes that."""
    viewer, renderer, _table, _request = _open(make_napari_viewer, InstancedRenderer)
    viewer.camera.angles = (12.0, -30.0, 75.0)
    viewer.camera.zoom = 0.37
    viewer.camera.center = (10.0, 4_321.0, 5_678.0)
    viewer.layers.selection.active = renderer.layer(1)
    before = (
        tuple(viewer.camera.center),
        viewer.camera.zoom,
        tuple(viewer.camera.angles),
        viewer.dims.ndisplay,
    )
    splats = _image(viewer)

    renderer.set_appearance(1, LayerAppearance(traces=True, opacity=1.0))
    renderer.trace_layer(1).opacity = 0.0  # drawn, but invisible
    assert (
        tuple(viewer.camera.center),
        viewer.camera.zoom,
        tuple(viewer.camera.angles),
        viewer.dims.ndisplay,
    ) == before
    assert viewer.layers.selection.active is renderer.layer(1)
    # The localizations draw as they did, a fourth axis or not.
    _assert_same_picture(_image(viewer), splats)

    renderer.set_appearance(1, LayerAppearance(traces=False))
    assert viewer.dims.ndim == 3
    assert (
        tuple(viewer.camera.center),
        viewer.camera.zoom,
        tuple(viewer.camera.angles),
        viewer.dims.ndisplay,
    ) == before
    _assert_same_picture(_image(viewer), splats)


def _assert_same_picture(image, reference):
    """The same picture, up to the GPU's rounding of a float sum to 8 bits.

    The camera checks above are exact; this asks whether the splats moved or
    changed.  Bit-for-bit equality asked more than that: on the macOS CI
    runner the summed-contrast resolve rounded six of 1.44 million channel
    values one step differently between two identical frames.  A view that
    moved, or a layer that drew, changes thousands of pixels by far more.
    """
    difference = np.abs(image.astype(np.int16) - reference.astype(np.int16))
    assert difference.max() <= 2, f"largest change {difference.max()}"
    assert (
        np.count_nonzero(difference) <= 1e-4 * difference.size
    ), f"{np.count_nonzero(difference)} of {difference.size} values changed"


def test_flat_data_keeps_its_plane_in_2d(make_napari_viewer):
    """Five columns for flat data too, so time never lands on z."""
    viewer, renderer, _table, request = _open(
        make_napari_viewer, InstancedRenderer, zdim=False
    )
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)
    assert layer.ndim == 4
    assert np.all(layer.data[:, 2] == request.coords[0, 0])
    assert viewer.dims.ndisplay == 2
    assert viewer.dims.nsteps[:2] == (1, 1)


# ------------------------------------------------- following the dataset


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_overlay_follows_filters(make_napari_viewer, backend_class):
    viewer, renderer, table, _request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)

    table.set_filter_mask(table.column("trace_id") < 2)
    renderer.update(1, _plan(table))

    assert renderer.trace_layer(1) is layer
    assert set(np.unique(layer.data[:, 0])) == {0, 1}


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_overlay_follows_visibility_and_opacity(make_napari_viewer, backend_class):
    _viewer, renderer, _table, _request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)

    renderer.set_appearance(1, LayerAppearance(opacity=0.4))
    assert layer.opacity == pytest.approx(0.4)
    renderer.set_appearance(1, LayerAppearance(opacity=0.0))  # a hidden channel
    assert not renderer.draws_traces(1)
    renderer.set_appearance(1, LayerAppearance(opacity=1.0))
    renderer.set_visible(1, False)  # a filter that left nothing
    assert not layer.visible and not renderer.draws_traces(1)
    renderer.update(1, _plan(LocalizationTable(_walks())))
    assert layer.visible and renderer.draws_traces(1)


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_width_and_colour_reach_the_layer(make_napari_viewer, backend_class):
    _viewer, renderer, table, _request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True, trace_width_px=5.0))
    layer = renderer.trace_layer(1)
    assert layer.tail_width == 5.0

    by_trace = np.array(layer.track_colors)
    ids = layer.data[:, 0]
    for trace in np.unique(ids):
        assert len(np.unique(by_trace[ids == trace], axis=0)) == 1

    renderer.set_appearance(1, LayerAppearance(trace_color_by="progress"))
    by_progress = np.array(layer.track_colors)
    first = ids == ids[0]
    assert len(np.unique(by_progress[first], axis=0)) > 1
    assert renderer.appearance(1).trace_color_by == "progress"


def test_an_unplanned_column_falls_back_to_one_hue_per_trace(make_napari_viewer):
    _viewer, renderer, _table, _request = _open(make_napari_viewer, InstancedRenderer)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    per_trace = np.array(renderer.trace_layer(1).track_colors)
    with pytest.warns(UserWarning, match="trace_properties"):
        renderer.set_appearance(1, LayerAppearance(trace_color_by="efo"))
    np.testing.assert_array_equal(renderer.trace_layer(1).track_colors, per_trace)


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_traces_only_update_leaves_the_splats_alone(
    make_napari_viewer, backend_class
):
    _viewer, renderer, table, request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    splats = _centres(renderer).copy()

    table.set_filter_mask(table.column("trace_id") == 3)
    renderer.update(1, _plan(table, changed=Changed.TRACES))

    np.testing.assert_array_equal(_centres(renderer), splats)
    assert set(np.unique(renderer.trace_layer(1).data[:, 0])) == {3}


def test_a_width_change_does_not_rebuild_the_layer(make_napari_viewer):
    _viewer, renderer, _table, request = _open(make_napari_viewer, InstancedRenderer)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)
    data = layer.data
    renderer.set_appearance(1, LayerAppearance(trace_width_px=4.0))
    renderer.update(1, request.with_changes(Changed.SIGMAS | Changed.VALUES))
    assert renderer.trace_layer(1) is layer
    assert layer.data is data


def test_nothing_to_connect_means_no_layer(make_napari_viewer):
    """And so no time axis in the viewer either."""
    viewer, renderer, table, _request = _open(make_napari_viewer, InstancedRenderer)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    table.set_filter_mask(np.arange(len(table)) < 3)  # one row from each of 3
    renderer.update(1, _plan(table))
    assert renderer.trace_layer(1) is None
    assert viewer.dims.ndim == 3
    assert renderer.appearance(1).traces is True  # still asked for

    table.reset()
    renderer.update(1, _plan(table))
    assert renderer.trace_layer(1) is not None


def test_a_request_without_trace_vertices_draws_none(make_napari_viewer):
    viewer = make_napari_viewer()
    renderer = InstancedRenderer(viewer)
    renderer.open(
        1,
        RenderPlanner().plan(
            LocalizationTable(_walks()),
            GaussianSettings(),
            DatasetTraits(zdim_present=True),
            name="plain",
        ),
    )
    renderer.set_appearance(1, LayerAppearance(traces=True))
    assert renderer.trace_layer(1) is None
    assert not renderer.draws_traces(1)


# --------------------------------------------------------- the lifecycle


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_closing_the_dataset_removes_its_traces(make_napari_viewer, backend_class):
    viewer, renderer, _table, _request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    layer = renderer.trace_layer(1)

    renderer.close(1)

    assert layer not in viewer.layers
    assert renderer.trace_layer(1) is None
    assert renderer._traces.host_bytes(1) == 0


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_close_all_and_detach_leave_nothing_behind(make_napari_viewer, backend_class):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    for dataset_id in (1, 2):
        renderer.open(dataset_id, _plan(LocalizationTable(_walks(seed=dataset_id))))
        renderer.set_appearance(dataset_id, LayerAppearance(traces=True))
    assert len(viewer.layers) == 4

    renderer.close_all()
    renderer.detach()

    assert len(viewer.layers) == 0
    # Detached: a layer removed later is none of the overlay's business.
    viewer.add_points(np.zeros((1, 3)))
    viewer.layers.clear()


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_reopening_starts_without_traces(make_napari_viewer, backend_class):
    viewer, renderer, table, _request = _open(make_napari_viewer, backend_class)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    renderer.open(1, _plan(table))
    assert renderer.trace_layer(1) is None
    assert len(viewer.layers) == 1


def test_a_trace_layer_the_user_deletes_stays_deleted(make_napari_viewer):
    """Deleting it means "not these traces": the next update must not bring
    it back, and the host hears about it."""
    viewer, renderer, table, _request = _open(make_napari_viewer, InstancedRenderer)
    heard = []
    renderer.on_traces_removed_by_host = heard.append
    renderer.set_appearance(1, LayerAppearance(traces=True))

    viewer.layers.remove(renderer.trace_layer(1))

    assert heard == [1]
    assert renderer.is_open(1)  # the dataset itself stays
    assert renderer.appearance(1).traces is False
    renderer.update(1, _plan(table))
    assert renderer.trace_layer(1) is None


def test_deleting_the_localizations_takes_their_traces(make_napari_viewer):
    viewer, renderer, _table, _request = _open(make_napari_viewer, InstancedRenderer)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    traces = renderer.trace_layer(1)

    viewer.layers.remove(renderer.layer(1))

    assert traces not in viewer.layers
    assert renderer.trace_layer(1) is None


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_the_overlay_reports_its_memory(make_napari_viewer, backend_class):
    _viewer, renderer, _table, request = _open(make_napari_viewer, backend_class)
    before = renderer.host_bytes(1)
    renderer.set_appearance(1, LayerAppearance(traces=True))
    added = renderer.host_bytes(1) - before
    assert added >= renderer.trace_layer(1).data.nbytes
