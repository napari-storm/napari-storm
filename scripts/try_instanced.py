#!/usr/bin/env python3
"""Open napari-storm with the instanced backend, for looking at.

The Level 3 checkpoint. `backend-comparison.md` establishes what the instanced
backend costs and measures; this is for finding out what it *feels* like, and
what breaks when the plugin is restricted to it.

    python scripts/try_instanced.py                 # 200k synthetic points
    python scripts/try_instanced.py --n 2000000
    python scripts/try_instanced.py --file data.hdf5
    python scripts/try_instanced.py --backend billboards   # the current one

Things worth trying, because they are where a backend usually breaks: drag the
render-range sliders, switch between fixed and variable Gaussian mode, change
the Gaussian size, rotate into 3-D, toggle Z colour encoding, change a
colormap, hide and show a channel, load a second dataset, and unload one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000, help="synthetic count")
    parser.add_argument("--file", help="a localization file to open instead")
    parser.add_argument(
        "--backend",
        default="instanced",
        choices=("instanced", "billboards", "points"),
        help="which renderer to run (default: instanced)",
    )
    parser.add_argument("--zdim", action="store_true", help="3-D fixture")
    args = parser.parse_args(argv)

    # Before napari, before any GL context: selecting VisPy's instancing
    # backend afterwards has no effect, and the failure is a blank canvas
    # rather than an error, so it is done first and reported.
    if args.backend == "instanced":
        from napari_storm.napari_particles._napari_compat import (
            enable_instanced_backend,
        )

        if not enable_instanced_backend():
            print(
                "instanced rendering is unavailable: VisPy's 'gl+' backend "
                "could not be selected. Is PyOpenGL installed?"
            )
            return 2
        print("GL backend: gl+ (instancing available)")

    import napari

    from napari_storm._dock_widget import napari_storm

    backends = {
        "instanced": "napari_storm.napari_particles.instanced_renderer:"
        "InstancedRenderer",
        "billboards": "napari_storm.napari_particles.renderer:"
        "NapariParticlesRenderer",
        "points": "napari_storm.napari_particles.points_renderer:"
        "NapariPointsRenderer",
    }
    import importlib

    module_name, class_name = backends[args.backend].split(":")
    backend_class = getattr(importlib.import_module(module_name), class_name)

    viewer = napari.Viewer()
    widget = napari_storm(napari_viewer=viewer, renderer=backend_class(viewer))
    viewer.window.add_dock_widget(widget, name="napari-storm", area="right")

    if args.file:
        widget.open_localization_data_file_and_get_dataset(file_path=args.file)
    else:
        from napari_storm._tests.fixtures import make_dataset

        print(f"generating {args.n:,} localizations ...")
        widget.get_dataset_from_test_mode(
            [make_dataset(args.n, zdim=args.zdim, name="synthetic")]
        )

    dataset = widget.localization_datasets[0]
    renderer = widget.data_to_layer_itf.renderer
    drawn = dataset.number_of_active_entries()
    print(
        f"backend={args.backend}  drawn={drawn:,}  "
        f"host bytes/localization={renderer.host_bytes(dataset.dataset_id) / max(drawn, 1):.0f}"
    )
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
