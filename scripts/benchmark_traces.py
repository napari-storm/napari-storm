#!/usr/bin/env python3
"""What connecting traces costs, at the size of a long MINFLUX run.

Usage:
    python scripts/benchmark_traces.py [n_traces] [vertices_per_trace]

Defaults to 10,000 traces of 100 localizations -- a million vertices -- each a
3-D random walk, written interleaved as concurrent MINFLUX traces arrive, with
a float32 time column.  Measured, best of a few repeats:

* planning: `plan_traces`, the host-free grouping and time ordering, and the
  same through `RenderPlanner.plan` against a plan without traces;
* the overlay: turning traces on (the Tracks layer is built), a filter change
  (its data is replaced), a width or colouring change, and turning it off;
* drawing: frame time rotating in 3-D, with and without the overlay, rendered
  with `SceneCanvas.render(size=...)`, whose pixel readback waits for the GPU.

The numbers in the commit that added traces came from this script.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from napari_storm.napari_particles._napari_compat import (  # noqa: E402
    enable_instanced_backend,
)

enable_instanced_backend()

import napari  # noqa: E402

from napari_storm.core import (  # noqa: E402
    DatasetTraits,
    GaussianSettings,
    LayerAppearance,
    LocalizationTable,
    RenderPlanner,
    plan_traces,
)
from napari_storm.core.renderer import Changed  # noqa: E402
from napari_storm.napari_particles.selection import select_renderer  # noqa: E402

N_TRACES = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
PER_TRACE = int(sys.argv[2]) if len(sys.argv) > 2 else 100
REPEATS = 3
SIZE = (2560, 1600)
FRAMES = 12


def walks(n_traces, per_trace, seed=0):
    """Interleaved random walks, as concurrent MINFLUX traces are written."""
    rng = np.random.default_rng(seed)
    n = n_traces * per_trace
    records = np.zeros(
        n,
        dtype=[
            ("x_pos_nm", "f4"),
            ("y_pos_nm", "f4"),
            ("z_pos_nm", "f4"),
            ("trace_id", "i4"),
            ("time_s", "f4"),
            ("efo", "f4"),
        ],
    )
    starts = rng.uniform(0, 50_000, (n_traces, 3))
    steps = rng.normal(0, 20, (per_trace, n_traces, 3)).cumsum(axis=0)
    positions = (starts[None] + steps).reshape(-1, 3)
    records["x_pos_nm"] = positions[:, 0]
    records["y_pos_nm"] = positions[:, 1]
    records["z_pos_nm"] = positions[:, 2] / 50
    records["trace_id"] = np.tile(np.arange(n_traces), per_trace)
    records["time_s"] = np.arange(n) * 1e-4
    records["efo"] = rng.uniform(1e4, 1e5, n)
    return np.rec.array(records)


def best(function, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def frame_ms(viewer, canvas):
    times = []
    for i in range(FRAMES):
        viewer.camera.angles = (0.0, 15.0 * np.sin(i / 3.0), 90.0 + 4.0 * i)
        start = time.perf_counter()
        canvas.render(size=SIZE)
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times[2:]))


def main():
    table = LocalizationTable(walks(N_TRACES, PER_TRACE), copy=False)
    traits = DatasetTraits(zdim_present=True)
    settings = GaussianSettings(fixed_sigma_xy_nm=10.0, fixed_sigma_z_nm=10.0)
    planner = RenderPlanner()
    print(f"{N_TRACES:,} traces x {PER_TRACE} = {len(table):,} vertices")

    seconds = best(lambda: plan_traces(table, traits, trace_column="trace_id"))
    print(f"plan_traces:                    {seconds:7.3f} s")
    without = best(lambda: planner.plan(table, settings, traits, name="b"))
    with_traces = best(
        lambda: planner.plan(table, settings, traits, name="b", trace_column="trace_id")
    )
    print(f"RenderPlanner.plan, no traces:  {without:7.3f} s")
    print(f"RenderPlanner.plan, traces:     {with_traces:7.3f} s")

    viewer = napari.Viewer(show=False)
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    renderer = select_renderer(viewer)
    request = planner.plan(table, settings, traits, name="b", trace_column="trace_id")
    renderer.open(1, request)
    viewer.dims.ndisplay = 3
    viewer.reset_view()
    print(f"frame, splats only:             {frame_ms(viewer, canvas):7.1f} ms")

    def on():
        renderer.set_appearance(1, LayerAppearance(traces=True))

    def off():
        renderer.set_appearance(1, LayerAppearance(traces=False))

    times = []
    for _ in range(REPEATS):
        off()
        start = time.perf_counter()
        on()
        times.append(time.perf_counter() - start)
    print(f"traces on (layer built):        {min(times):7.3f} s")
    print(f"frame, splats and traces:       {frame_ms(viewer, canvas):7.1f} ms")

    half = np.zeros(len(table), dtype=bool)
    half[: len(table) // 2] = True
    full = np.ones(len(table), dtype=bool)
    requests = []
    for mask in (half, full):
        table.set_filter_mask(mask)
        requests.append(
            planner.plan(table, settings, traits, name="b", trace_column="trace_id")
        )
    table.reset()
    # What a width or intensity change sends: the vertices have not moved.
    splats_only = [r.with_changes(Changed.SIGMAS | Changed.VALUES) for r in requests]

    seconds = best(lambda: [renderer.update(1, r) for r in requests]) / 2
    print(f"filter change, traces on:       {seconds:7.3f} s")
    off()
    seconds = best(lambda: [renderer.update(1, r) for r in requests]) / 2
    print(f"filter change, traces off:      {seconds:7.3f} s")
    on()
    seconds = best(lambda: [renderer.update(1, r) for r in splats_only]) / 2
    print(f"sigma-only change, traces on:   {seconds:7.3f} s")

    seconds = best(
        lambda: [
            renderer.set_appearance(1, LayerAppearance(trace_width_px=w))
            for w in (3.0, 2.0)
        ]
    )
    print(f"width change:                   {seconds / 2:7.3f} s")
    seconds = best(
        lambda: [
            renderer.set_appearance(1, LayerAppearance(trace_color_by=c))
            for c in ("time", "trace")
        ]
    )
    print(f"colouring change:               {seconds / 2:7.3f} s")
    times = []
    for _ in range(REPEATS):
        on()
        start = time.perf_counter()
        off()
        times.append(time.perf_counter() - start)
    print(f"traces off (layer removed):     {min(times):7.3f} s")
    splat_bytes = renderer.host_bytes(1) - renderer._traces.host_bytes(1)
    on()
    overlay_bytes = renderer.host_bytes(1) - splat_bytes
    print(f"overlay host bytes / vertex:    {overlay_bytes / len(table):7.1f}")
    renderer.close_all()
    viewer.close()


if __name__ == "__main__":
    main()
