#!/usr/bin/env python3
"""Frame time in napari: per-localization window vs summed contrast.

Usage:
    python scripts/benchmark_summed_contrast.py [daxview .h5]

Defaults to the 4Pi STORM sample in sample_data/, drawn as it is and tiled to
about 1M and 5M localizations, rotating in 3-D at 2560x1600.

Same layer, same view, same rotation; only `summed_contrast` changes.  Rounds
are interleaved so drift (thermals, other load) hits both alike.  Rendered with
`SceneCanvas.render(size=...)`, which ends in a pixel readback and therefore
waits for the GPU; the readback is the same in both modes.
"""

import sys
import time
from pathlib import Path

import h5py
import napari
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from napari_storm.core import LayerAppearance, RenderRequest  # noqa: E402
from napari_storm.napari_particles.instanced_renderer import (  # noqa: E402
    InstancedRenderer,
)
from napari_storm.napari_particles.selection import (  # noqa: E402
    renderer_class_for_session,
)

SAMPLE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else REPO_ROOT / "sample_data" / "beta_ii_spectrin.storm_4pi.h5"
)
SIZE = (2560, 1600)
SIGMA_NM = 20 / 2.354  # the default fixed FWHM
FRAMES, ROUNDS = 12, 4


def spectrin_nm():
    with h5py.File(SAMPLE) as f:
        g = f["molecule_set_data"]
        t = g["datatable"][()]
        px = 1000.0 * float(g["xy_pixel_size_um"][()])
        pz = 1000.0 * float(g["z_pixel_size_um"][()])
    # (z, y, x), as the renderer takes it
    return np.stack(
        [t["Z_POS_PIXELS"] * pz, t["Y_POS_PIXELS"] * px, t["X_POS_PIXELS"] * px], 1
    ).astype(np.float32)


def tiled(coords, copies, rng):
    """*copies* jittered copies of the sample: denser, same structure."""
    out = np.concatenate([coords] * copies)
    out += rng.normal(0.0, 15.0, out.shape).astype(np.float32)
    return out


def request(coords):
    n = len(coords)
    return RenderRequest(
        coords=coords,
        sigmas=np.ones((n, 3), np.float32),
        size=5.0 * SIGMA_NM,
        values=np.ones(n, np.float32),
        name="bench",
        colormap="gray",
    )


def frame_ms(viewer, canvas, frames):
    times = []
    for i in range(frames):
        viewer.camera.angles = (0.0, 15.0 * np.sin(i / 3.0), 90.0 + 4.0 * i)
        start = time.perf_counter()
        canvas.render(size=SIZE)
        times.append(1e3 * (time.perf_counter() - start))
    return times


def main():
    viewer = napari.Viewer(show=False)
    cls, reason = renderer_class_for_session()
    assert cls is InstancedRenderer, reason
    rng = np.random.default_rng(0)
    base = spectrin_nm()
    datasets = [
        ("spectrin 351k", base),
        ("spectrin x3 1.05M", tiled(base, 3, rng)),
        ("spectrin x14 4.9M", tiled(base, 14, rng)),
    ]

    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    renderer = InstancedRenderer(viewer)
    viewer.dims.ndisplay = 3

    canvas.render(size=SIZE)
    empty = np.median(frame_ms(viewer, canvas, FRAMES))
    print(f"empty canvas (readback only): {empty:.1f} ms at {SIZE[0]}x{SIZE[1]}")
    print(f"{'dataset':<20} {'per-loc ms':>11} {'summed ms':>10} {'delta':>8}")

    for label, coords in datasets:
        renderer.open(1, request(coords))
        renderer.set_appearance(1, LayerAppearance(contrast_limits=(0.0, 5.0)))
        viewer.reset_view()
        results = {False: [], True: []}
        for summed in (False, True):  # warm-up: compile both shaders
            renderer.set_appearance(1, LayerAppearance(summed_contrast=summed))
            frame_ms(viewer, canvas, 3)
        for _round in range(ROUNDS):
            for summed in (False, True):
                renderer.set_appearance(1, LayerAppearance(summed_contrast=summed))
                results[summed] += frame_ms(viewer, canvas, FRAMES)
        old, new = np.median(results[False]), np.median(results[True])
        print(f"{label:<20} {old:>11.1f} {new:>10.1f} {new - old:>+8.1f}")
        renderer.close(1)
    viewer.close()


if __name__ == "__main__":
    main()
