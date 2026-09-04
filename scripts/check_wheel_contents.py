#!/usr/bin/env python3
"""Assert a built wheel actually contains the import package.

The wheel built from the previous setup.cfg contained six files, all of them
distribution metadata: `packages = find:` was used with a src layout but without
`[options.packages.find] where = src`, so package discovery resolved to an empty
list and the build succeeded anyway.  A green build is therefore not evidence
that the artifact is usable -- it has to be inspected.

Usage:
    python scripts/check_wheel_contents.py dist/napari_storm-*.whl
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# Files that must be present for the plugin to work once installed.
REQUIRED_FILES = [
    "napari_storm/__init__.py",
    "napari_storm/_dock_widget.py",
    "napari_storm/_reader.py",
    "napari_storm/napari.yaml",
]

REQUIRED_SUBPACKAGES = [
    "napari_storm/core/",
    "napari_storm/localization_dataset_types/",
    "napari_storm/napari_particles/",
    "napari_storm/pyqt/",
]

# Tests must not ship inside the distribution.
FORBIDDEN_PREFIXES = ["napari_storm/_tests/"]

MIN_MODULE_COUNT = 40


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    wheel_path = Path(argv[1])
    if not wheel_path.is_file():
        print(f"ERROR: no such wheel: {wheel_path}")
        return 2

    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    problems: list[str] = []

    for required in REQUIRED_FILES:
        if required not in names:
            problems.append(f"missing required file: {required}")

    for package in REQUIRED_SUBPACKAGES:
        init = package + "__init__.py"
        if init not in names:
            problems.append(f"missing subpackage (no {init})")

    for prefix in FORBIDDEN_PREFIXES:
        leaked = [n for n in names if n.startswith(prefix)]
        if leaked:
            problems.append(
                f"{len(leaked)} file(s) under {prefix} should not be shipped"
            )

    modules = [
        n for n in names if n.endswith(".py") and not n.startswith("napari_storm-")
    ]
    if len(modules) < MIN_MODULE_COUNT:
        problems.append(
            f"only {len(modules)} python modules in the wheel; "
            f"expected at least {MIN_MODULE_COUNT} -- package discovery is "
            f"probably misconfigured"
        )

    print(f"{wheel_path.name}: {len(names)} entries, {len(modules)} python modules")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  FAIL {problem}")
        return 1
    print("wheel contents OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
