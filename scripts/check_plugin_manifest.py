#!/usr/bin/env python3
"""Verify that every npe2 contribution resolves to something importable.

`npe2 validate` checks the manifest against its schema, not against itself: a
widget may reference a command id that is never defined and validation still
passes.  That is exactly what happened here -- the "SMLM Localization" widget
pointed at `napari-storm.show_localization_controls`, which did not exist, so
selecting it from the Plugins menu could not work.

This script closes that gap.  For every contribution that names a command it
checks the id is defined, then imports the target and confirms the attribute is
present and callable.

Usage:
    python scripts/check_plugin_manifest.py [path/to/napari.yaml]

Exits non-zero on the first problem, printing every problem found.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "napari_storm"
    / "napari.yaml"
)

# Contribution kinds that carry a `command` field.
COMMAND_BEARING = ("widgets", "readers", "writers", "sample_data")


def main(argv: list[str]) -> int:
    manifest_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_MANIFEST
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    try:
        from npe2 import PluginManifest
    except ImportError:
        print("ERROR: npe2 is required (pip install npe2)")
        return 2

    manifest = PluginManifest.from_file(manifest_path)
    contributions = manifest.contributions
    commands = {c.id: c for c in (contributions.commands or [])}

    problems: list[str] = []
    checked = 0

    for kind in COMMAND_BEARING:
        for contribution in getattr(contributions, kind, None) or []:
            command_id = getattr(contribution, "command", None)
            if command_id is None:
                continue
            checked += 1
            if command_id not in commands:
                problems.append(
                    f"{kind}: command id {command_id!r} is not defined in "
                    f"contributions.commands"
                )
                continue
            python_name = commands[command_id].python_name
            if not python_name:
                problems.append(f"{kind}: {command_id} has no python_name")
                continue
            module_name, _, attribute = python_name.partition(":")
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - report, do not mask
                problems.append(
                    f"{kind}: {command_id} -> cannot import {module_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            target = getattr(module, attribute, None)
            if target is None:
                problems.append(
                    f"{kind}: {command_id} -> {module_name} has no "
                    f"attribute {attribute!r}"
                )
            elif not callable(target):
                problems.append(
                    f"{kind}: {command_id} -> {python_name} is not callable"
                )
            else:
                print(f"  OK  {kind:<12} {command_id} -> {python_name}")

    # Commands that nothing references are dead weight, not errors.
    referenced = {
        getattr(c, "command", None)
        for kind in COMMAND_BEARING
        for c in (getattr(contributions, kind, None) or [])
    }
    for unused in sorted(set(commands) - referenced):
        print(f"  note: command {unused} is defined but not referenced")

    if problems:
        print(f"\n{len(problems)} problem(s) in {manifest_path}:")
        for problem in problems:
            print(f"  FAIL {problem}")
        return 1

    print(f"\nAll {checked} contribution(s) resolve in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
