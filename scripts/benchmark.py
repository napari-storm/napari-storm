#!/usr/bin/env python3
"""Baseline measurement harness for napari-storm.

Produces the Level 0 exit artifact described in docs/modernization-review.md:
load time, geometry cost, peak memory, and — critically — how long each
operation blocks the Qt main thread.

That last column matters because everything in this plugin runs on the UI
thread. A user cannot tell a 40-second freeze from a crash, and will report
both as "it crashed", so a crash reproducer that does not separate the two
will send Level 1 chasing a leak in scenarios that were only ever slow.

Usage:
    python scripts/benchmark.py                      # 100k only, quick
    python scripts/benchmark.py --sizes 100k,1M
    python scripts/benchmark.py --sizes 100k,1M,5M --out docs/baseline-report.md

Requires a working Qt/OpenGL environment; on a headless Linux box run it under
xvfb-run.
"""
from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# --------------------------------------------------------------------- timing


@contextmanager
def measure(store, key):
    """Record wall-clock seconds for a block, and absolute RSS afterwards.

    RSS is recorded as an absolute figure, not a delta.  A delta is meaningless
    across successive dataset sizes in one process: the allocator retains freed
    pages, so a later, larger dataset can report a *smaller* growth than an
    earlier one.  Each size is measured in its own subprocess (see main) so the
    absolute number is attributable.
    """
    gc.collect()
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - start
        store[key + "_rss_mb"] = _rss_bytes() / 1e6


def _rss_bytes():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        return raw if sys.platform == "darwin" else raw * 1024


# ------------------------------------------------------------------ scenarios


def render_array_bytes(layer):
    """Host-side bytes held by one Particles layer, per localization."""
    names = (
        "_coords",
        "_centercoords",
        "_sigmas",
        "_size",
        "_texcoords",
    )
    total = 0
    for name in names:
        arr = getattr(layer, name, None)
        if isinstance(arr, np.ndarray):
            total += arr.nbytes
    for name in ("_view_faces", "_view_vertices"):
        arr = getattr(layer, name, None)
        if isinstance(arr, np.ndarray):
            total += arr.nbytes
    data = getattr(layer, "data", None)
    if isinstance(data, tuple):
        for arr in data:
            if isinstance(arr, np.ndarray):
                total += arr.nbytes
    return total


BACKENDS = {
    "billboards": "napari_storm.napari_particles.renderer:NapariParticlesRenderer",
    "points": "napari_storm.napari_particles.points_renderer:NapariPointsRenderer",
    # Needs VisPy's gl+ backend, selected before any GL context exists -- which
    # is why each size already runs in its own subprocess.
    "instanced": "napari_storm.napari_particles.instanced_renderer:InstancedRenderer",
}


def _backend_class(name):
    """Resolve a backend by name, so a new candidate needs no code here."""
    import importlib

    if name == "instanced":
        from napari_storm.napari_particles._napari_compat import \
            enable_instanced_backend

        if not enable_instanced_backend():
            raise RuntimeError(
                "instanced rendering needs VisPy's gl+ backend; it could not "
                "be selected in this process"
            )

    module_name, class_name = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def run_size(label, n, zdim=False, backend="billboards"):
    """Measure the full open -> update -> close cycle for one dataset size."""
    import napari

    from napari_storm._dock_widget import napari_storm
    from napari_storm._tests.fixtures import make_dataset

    result = {"label": label, "n": n, "backend": backend}

    with measure(result, "fixture_s"):
        dataset = make_dataset(n, zdim=zdim)

    viewer = napari.Viewer(show=False)
    try:
        widget = napari_storm(
            napari_viewer=viewer, renderer=_backend_class(backend)(viewer)
        )

        with measure(result, "open_s"):
            widget.get_dataset_from_test_mode([dataset])

        dataset = widget.localization_datasets[0]
        # Asked of the backend rather than guessed from whichever arrays this
        # harness happens to know about, so two backends are measured alike.
        renderer = widget.data_to_layer_itf.renderer
        result["bytes_per_loc"] = renderer.host_bytes(dataset.dataset_id) / n

        # The dominant interactive path today: any setting change rebuilds.
        with measure(result, "update_s"):
            widget.data_to_layer_itf.update_layers()

        # Ten repeats is the leak-shaped scenario from the plan.
        with measure(result, "update_x10_s"):
            for _ in range(10):
                widget.data_to_layer_itf.update_layers()

        with measure(result, "close_s"):
            widget.clear_datasets()
    finally:
        viewer.close()
    return result


def run_extreme_gaussian():
    """Fragment-cost case: few localizations, enormous screen footprint.

    This harness runs off-screen and measures CPU-side work, so it cannot time
    GPU fill rate. What it *can* do is report the geometric quantity that drives
    that cost, and which the P0-04 screen-space cap would bound: how much of the
    field of view a single splat covers, and the resulting overdraw factor.
    """
    import napari

    from napari_storm._dock_widget import napari_storm
    from napari_storm._tests.fixtures import extreme_gaussian_dataset
    from napari_storm.ns_constants import FWHM_TO_SIGMA

    result = {"label": "extreme-gaussian", "n": 1_000}
    dataset = extreme_gaussian_dataset()

    viewer = napari.Viewer(show=False)
    try:
        widget = napari_storm(napari_viewer=viewer)
        widget.get_dataset_from_test_mode([dataset])
        # A 50 nm field of view drawn with a 20 um FWHM: every splat covers the
        # whole scene many times over.
        widget.render_config.fixed_sigma_xy_nm = 20_000.0 / FWHM_TO_SIGMA
        with measure(result, "update_s"):
            widget.data_to_layer_itf.update_layers()

        itf = widget.data_to_layer_itf
        layer = itf.layer_for(widget.localization_datasets[0])
        result["bytes_per_loc"] = render_array_bytes(layer) / result["n"]

        # Quad edge length in world units, versus the field of view.
        splat_nm = float(np.max(layer._size))
        fov_nm = max(
            itf.render_range_x[1] - itf.render_range_x[0],
            itf.render_range_y[1] - itf.render_range_y[0],
            1e-9,
        )
        result["splat_over_fov"] = splat_nm / fov_nm
        # Total shaded area divided by the visible area.
        result["overdraw"] = result["n"] * (splat_nm**2) / (fov_nm**2)
        widget.clear_datasets()
    finally:
        viewer.close()
    return result


# --------------------------------------------------------------------- report


def environment():
    import napari
    import vispy

    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "napari": napari.__version__,
        "vispy": vispy.__version__,
    }


def render_report(rows, env):
    lines = [
        "# napari-storm baseline measurements",
        "",
        "Generated by `python scripts/benchmark.py`. This is the Level 0 exit",
        "artifact described in `modernization-review.md`; regenerate it whenever a",
        "performance claim needs to be re-checked.",
        "",
        "## Environment",
        "",
        "| Component | Version |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Measurements",
        "",
        "All timings are wall-clock seconds spent blocking the Qt main thread.",
        "The scenarios build their datasets in memory rather than reading a file,",
        "so this is the work that has *not* been moved off the UI thread: geometry",
        "expansion, buffer construction and the handover to napari.",
        "",
        "| Dataset | Localizations | Open (s) | 1 update (s) | 10 updates (s) | Close (s) | Layer arrays (MB) | RSS (MB) | Render bytes/loc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def cell(row, key, spec):
        """Format a measurement, or an em dash when the scenario omits it."""
        value = row.get(key)
        return "—" if value is None else format(value, spec)

    for r in rows:
        lines.append(
            "| {label} | {n} | {open} | {update} | {update10} | {close} "
            "| {arrays} | {rss} | {bpl} |".format(
                label=r["label"],
                n=f"{r['n']:,}",
                open=cell(r, "open_s", ".2f"),
                update=cell(r, "update_s", ".2f"),
                update10=cell(r, "update_x10_s", ".2f"),
                close=cell(r, "close_s", ".2f"),
                arrays=(
                    "—"
                    if r.get("bytes_per_loc") is None
                    else format(r["bytes_per_loc"] * r["n"] / 1e6, ".0f")
                ),
                rss=cell(r, "open_s_rss_mb", ".0f"),
                bpl=cell(r, "bytes_per_loc", ".0f"),
            )
        )
    extreme = next((r for r in rows if "splat_over_fov" in r), None)
    if extreme is not None:
        lines += [
            "",
            "## Fragment cost (extreme-Gaussian case)",
            "",
            "This harness runs off-screen and measures CPU work, so it does not time",
            "GPU fill rate. It reports the geometry that drives it instead — the",
            "quantity the screen-space cap (P0-04) bounds.",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| Localizations | {extreme['n']:,} |",
            f"| Splat edge / field-of-view width | {extreme['splat_over_fov']:.2f}× |",
            f"| Overdraw (total shaded area / visible area) | {extreme['overdraw']:,.0f}× |",
            "",
            "The fixture asks for a 20 µm FWHM on a 50 nm field of view. Uncapped that",
            "is a splat edge ~12 365× the viewport and an overdraw factor of 1.5e11:",
            "cost set by pixels shaded, which no memory budget can bound. The cap",
            "clamps the quad to half the field of view, leaving the numbers above.",
        ]

    lines += [
        "",
        "## Notes",
        "",
        "- Reading a file now runs on a worker with progress and cancel (P1-09), but",
        "  building billboard geometry and handing it to napari does not — that is a",
        "  main-thread requirement of the renderer. The scenarios above construct",
        "  their datasets in memory, so every second in this table is main-thread",
        "  time: this measures the part that is still blocking, not the part that",
        "  was moved off.",
        "- `10 updates` is the repeated-setting-change scenario. It should scale",
        "  linearly with `1 update`; superlinear growth indicates the layer/callback",
        "  accumulation described in P0-01 and P0-02.",
        "- `Layer arrays` is exact and deterministic: bytes/loc x count. Prefer it",
        "  over `RSS`, which on macOS excludes compressed pages and so can report a",
        "  *smaller* figure for a larger dataset once the machine is under memory",
        "  pressure. Both exclude VisPy buffers and GPU copies.",
        "- `Render bytes/loc` counts host-side arrays owned by the layer only. It",
        "  excludes VisPy buffers and GPU copies, so true footprint is higher.",
        "- The default render budget is 2 GB, i.e. ~5.8M localizations, so none of",
        "  the fixtures above is thinned. Set `NAPARI_STORM_RENDER_BUDGET_MB` to",
        "  measure the degraded path, or to 0 to disable the budget.",
        "- The extreme-Gaussian row has a trivial localization count; its cost is",
        "  overdraw, not geometry.",
        "",
    ]
    return "\n".join(lines)


def _jsonable(result):
    """numpy scalars (float32 from the render arrays) are not JSON-serializable."""
    out = {}
    for key, value in result.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def _run_in_subprocess(label, zdim, backend="billboards"):
    """Measure one scenario in a fresh interpreter.

    Isolation matters for two reasons: RSS is only attributable in a process
    that has not already loaded another dataset, and a size large enough to be
    killed by the OS takes only its own subprocess down rather than the run.
    """
    import json
    import subprocess

    cmd = [sys.executable, str(Path(__file__).resolve()), "--single", label]
    if zdim:
        cmd.append("--zdim")
    if backend:
        cmd += ["--backend", backend]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  {label}: FAILED (exit {proc.returncode})")
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")
        return {"label": label, "n": 0, "failed": True}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    print(f"  {label}: produced no result")
    return {"label": label, "n": 0, "failed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="100k",
        help="comma-separated sizes from 100k,1M,5M (default: 100k)",
    )
    parser.add_argument("--zdim", action="store_true", help="use 3-D fixtures")
    parser.add_argument(
        "--backend",
        # Follows the plugin's own default, so an unqualified run measures what
        # a user actually gets.  The earlier tables in baseline-report.md and
        # backend-comparison.md name their backend per row and stay valid.
        default="instanced",
        choices=sorted(BACKENDS),
        help="renderer backend to measure (default: instanced)",
    )
    parser.add_argument(
        "--skip-extreme", action="store_true", help="skip the extreme-Gaussian case"
    )
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    parser.add_argument(
        "--single",
        help="internal: measure one scenario and emit JSON (used per-subprocess)",
    )
    args = parser.parse_args(argv)

    from napari_storm._tests.fixtures import FIXTURE_SIZES

    # Child mode: one scenario, JSON on stdout.
    if args.single:
        import json

        if args.single == "extreme-gaussian":
            result = run_extreme_gaussian()
        else:
            result = run_size(
                args.single,
                FIXTURE_SIZES[args.single],
                zdim=args.zdim,
                backend=args.backend,
            )
        print("RESULT_JSON:" + json.dumps(_jsonable(result)))
        return 0

    rows = []
    for label in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        if label not in FIXTURE_SIZES:
            parser.error(f"unknown size {label!r}; choose from {list(FIXTURE_SIZES)}")
        print(f"measuring {label} ...", flush=True)
        rows.append(_run_in_subprocess(label, args.zdim, args.backend))

    if not args.skip_extreme:
        print("measuring extreme-gaussian ...", flush=True)
        rows.append(_run_in_subprocess("extreme-gaussian", args.zdim, args.backend))

    rows = [r for r in rows if not r.get("failed")]
    if not rows:
        print("no scenario completed")
        return 1
    report = render_report(rows, environment())
    if args.out:
        args.out.write_text(report)
        print(f"\nwrote {args.out}")
    else:
        print()
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
