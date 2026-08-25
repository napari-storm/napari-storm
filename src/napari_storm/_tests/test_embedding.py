"""Using napari-storm from another application, with no dock widget.

`docs/embedding.md` is written from this file. It is a test rather than a
snippet in a markdown fence so that it cannot quietly stop working: the example
a host copies is the one CI runs.

The question behind it is Level 5's — can a host like ImSwitch render
localizations without adopting our UI? — and the answer has to be demonstrated
rather than asserted, which is what §7.5 asks for.
"""
import numpy as np

from napari_storm.core import (DatasetTraits, GaussianSettings,
                               LayerAppearance, LocalizationTable,
                               RenderPlanner, WorldTransform)
from napari_storm.napari_particles.selection import select_renderer


def _records(n=20_000, seed=0):
    """Whatever the host already has, as a numpy record array."""
    rng = np.random.default_rng(seed)
    records = np.rec.array(
        np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")])
    )
    records.x_pos_nm = rng.uniform(0, 10_000, n)
    records.y_pos_nm = rng.uniform(0, 10_000, n)
    records.z_pos_nm = rng.uniform(-500, 500, n)
    return records


def _brightness(viewer):
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    return float(np.asarray(canvas.render())[..., :3].mean())


# --------------------------------------------------------------- the example


def test_a_host_can_render_localizations_without_the_dock_widget(
    make_napari_viewer,
):
    """The whole of it. Everything else in this file elaborates one step."""
    viewer = make_napari_viewer()

    # 1. Your data. The table is the canonical copy; nothing downstream
    #    reorders or filters it in place.
    table = LocalizationTable(_records())

    # 2. Decide what to draw. Host-free: no napari, no Qt.
    request = RenderPlanner().plan(
        table,
        GaussianSettings(fixed_sigma_xy_nm=30.0),
        DatasetTraits(zdim_present=True),
        name="from-the-host",
    )

    # 3. Draw it. `select_renderer` picks the instanced backend when the GL
    #    session can instance and falls back to billboards when it cannot.
    renderer = select_renderer(viewer)
    renderer.open(1, request)

    viewer.dims.ndisplay = 3
    assert _brightness(viewer) > 0.5


def test_the_default_colormap_is_visible(make_napari_viewer):
    """Omitting a colormap must not produce a black canvas.

    It used to: napari assigns an arbitrary unnamed colormap, and the instanced
    backend -- which samples the colormap in its own shader -- resolved that to
    black. Every piece of state reported itself healthy, which made it the
    hardest possible thing for a new embedder to debug.
    """
    viewer = make_napari_viewer()
    table = LocalizationTable(_records())
    request = RenderPlanner().plan(
        table, GaussianSettings(), DatasetTraits(zdim_present=True), name="ch"
    )

    assert request.colormap is not None
    select_renderer(viewer).open(1, request)
    viewer.dims.ndisplay = 3

    assert _brightness(viewer) > 0.5


# ------------------------------------------------------------ the lifecycle


def test_a_host_can_filter_by_replacing_the_mask(make_napari_viewer):
    """Filtering costs one boolean array, not a copy of the data."""
    viewer = make_napari_viewer()
    table = LocalizationTable(_records())
    settings, traits = GaussianSettings(), DatasetTraits(zdim_present=True)
    renderer = select_renderer(viewer)
    renderer.open(1, RenderPlanner().plan(table, settings, traits, name="ch"))

    mask = np.zeros(len(table), dtype=bool)
    mask[: len(table) // 4] = True
    table.set_filter_mask(mask)
    renderer.update(1, RenderPlanner().plan(table, settings, traits, name="ch"))

    assert renderer.layer(1).n_localizations == len(table) // 4


def test_a_host_can_move_a_dataset_in_world_space(make_napari_viewer):
    """Placement is a value the host supplies, not a hidden global."""
    viewer = make_napari_viewer()
    table = LocalizationTable(_records())
    settings, traits = GaussianSettings(), DatasetTraits(zdim_present=True)
    renderer = select_renderer(viewer)
    renderer.open(1, RenderPlanner().plan(table, settings, traits, name="ch"))
    # Column 2 is x: coordinates are (z, y, x), napari's order.
    before = renderer.layer(1).localization_coords[:, 2].min()

    renderer.update(
        1,
        RenderPlanner().plan(
            table,
            settings,
            traits,
            name="ch",
            transform=WorldTransform(translation_nm=(5_000.0, 0.0, 0.0)),
        ),
    )

    assert renderer.layer(1).localization_coords[:, 2].min() > before + 4_000


def test_a_host_can_recolour_without_replanning(make_napari_viewer):
    """Appearance is separate from what is drawn, so it costs nothing."""
    viewer = make_napari_viewer()
    renderer = select_renderer(viewer)
    renderer.open(
        1,
        RenderPlanner().plan(
            LocalizationTable(_records()),
            GaussianSettings(),
            DatasetTraits(zdim_present=True),
            name="ch",
        ),
    )

    renderer.set_appearance(1, LayerAppearance(colormap="green", opacity=0.5))

    assert renderer.appearance(1).opacity == 0.5


def test_closing_releases_everything(make_napari_viewer):
    viewer = make_napari_viewer()
    renderer = select_renderer(viewer)
    layer = renderer.open(
        1,
        RenderPlanner().plan(
            LocalizationTable(_records()),
            GaussianSettings(),
            DatasetTraits(zdim_present=True),
            name="ch",
        ),
    )

    renderer.close(1)

    assert not renderer.is_open(1)
    assert layer not in viewer.layers


def test_two_channels_coexist(make_napari_viewer):
    """One renderer, many datasets, keyed by whatever ids the host uses."""
    viewer = make_napari_viewer()
    renderer = select_renderer(viewer)
    settings, traits = GaussianSettings(), DatasetTraits(zdim_present=True)

    for dataset_id, colour in ((7, "red"), (9, "green")):
        renderer.open(
            dataset_id,
            RenderPlanner().plan(
                LocalizationTable(_records(seed=dataset_id)),
                settings,
                traits,
                name=f"channel-{dataset_id}",
                colormap=colour,
            ),
        )

    assert renderer.is_open(7) and renderer.is_open(9)
    assert len(renderer.layers) == 2
