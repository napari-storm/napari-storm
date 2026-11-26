#!/usr/bin/env python3
"""Keep smlm-view3d's version in step with its code and with napari-storm.

smlm-view3d, napari-storm's drawing core in smlm-view3d/, has its own version
in smlm-view3d/pyproject.toml.  A napari-storm release publishes it with
skip-existing, so two mistakes would pass without a word:

  pin        napari-storm's requirement must read ``smlm-view3d>=<that
             version>``.  Without a minimum, ``pip install -U napari-storm``
             keeps an older smlm-view3d already installed; with an older
             minimum, it can keep one napari-storm was never tested with.

  published  If PyPI already has that version, the smlm-view3d wheel built
             from this tree must match PyPI's, file for file and in its
             requirements.  Otherwise the code changed without a version
             bump, a napari-storm release would skip it, and napari-storm on
             PyPI would run on the older drawing core.  The fix is to bump
             smlm-view3d/pyproject.toml and the minimum in setup.cfg together.

Usage:
    python scripts/check_smlm_view3d_version.py pin
    python scripts/check_smlm_view3d_version.py published dist-smlm-view3d/*.whl

``--root`` names the repository (default: this script's parent folder).
Needs Python 3.11+ (tomllib) and ``packaging``; ``published`` asks pypi.org.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    sys.exit("check_smlm_view3d_version.py needs Python 3.11 or newer (tomllib)")

NAME = "smlm-view3d"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"

#: The METADATA fields that change what an install gets, beside the files.
METADATA_FIELDS = ("Requires-Dist", "Requires-Python", "Provides-Extra")


def repo_version(root: Path) -> Version:
    """smlm-view3d's version in smlm-view3d/pyproject.toml."""
    with open(root / "smlm-view3d" / "pyproject.toml", "rb") as f:
        return Version(tomllib.load(f)["project"]["version"])


def napari_storm_requirement(root: Path) -> Requirement | None:
    """napari-storm's requirement on smlm-view3d in setup.cfg, or None."""
    config = configparser.ConfigParser()
    config.read(root / "setup.cfg", encoding="utf-8")
    lines = config.get("options", "install_requires", fallback="").splitlines()
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == NAME:
            return requirement
    return None


def next_patch(version: Version) -> str:
    return f"{version.major}.{version.minor}.{version.micro + 1}"


def check_pin(root: Path) -> int:
    version = repo_version(root)
    requirement = napari_storm_requirement(root)
    wanted = f"{NAME}>={version}"
    if requirement is None:
        print(f"ERROR: setup.cfg's install_requires does not name {NAME}; add {wanted}")
        return 1
    minimums = [spec.version for spec in requirement.specifier if spec.operator == ">="]
    if not minimums:
        print(
            f"ERROR: setup.cfg requires '{requirement}' with no minimum, so "
            f"`pip install -U napari-storm` keeps whatever {NAME} is installed. "
            f"Write {wanted}: smlm-view3d/pyproject.toml is at {version}."
        )
        return 1
    if any(Version(minimum) != version for minimum in minimums):
        print(
            f"ERROR: setup.cfg requires '{requirement}', but "
            f"smlm-view3d/pyproject.toml is at {version}, the only version "
            f"napari-storm is tested with. Write {wanted}."
        )
        return 1
    if not requirement.specifier.contains(version, prereleases=True):
        print(
            f"ERROR: setup.cfg's '{requirement}' excludes {version}, "
            "smlm-view3d's own version."
        )
        return 1
    print(f"OK: setup.cfg requires '{requirement}'; smlm-view3d is at {version}.")
    return 0


def _pypi_release(version: Version) -> dict | None:
    """PyPI's JSON for this version, or None when PyPI does not have it."""
    url = PYPI_JSON.format(name=NAME, version=version)
    request = urllib.request.Request(
        url, headers={"User-Agent": "napari-storm-release-check"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _download_wheel(release: dict, folder: Path) -> Path:
    wheels = [f for f in release["urls"] if f["packagetype"] == "bdist_wheel"]
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected one wheel on PyPI, found {[f['filename'] for f in wheels]}"
        )
    entry = wheels[0]
    request = urllib.request.Request(
        entry["url"], headers={"User-Agent": "napari-storm-release-check"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["digests"]["sha256"]:
        raise RuntimeError(f"{entry['filename']}: sha256 does not match PyPI's")
    path = folder / entry["filename"]
    path.write_bytes(data)
    return path


def _contents(wheel: Path) -> tuple[dict[str, str], list[str]]:
    """File name -> sha256 outside .dist-info, and the install-relevant
    METADATA lines."""
    files, metadata = {}, []
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if ".dist-info/" in name:
                if name.endswith(".dist-info/METADATA"):
                    text = zf.read(name).decode("utf-8")
                    header = text.split("\n\n", 1)[0]
                    metadata = sorted(
                        line.strip()
                        for line in header.splitlines()
                        if line.split(":", 1)[0] in METADATA_FIELDS
                    )
                continue
            files[name] = hashlib.sha256(zf.read(name)).hexdigest()
    return files, metadata


def check_published(root: Path, wheel: Path) -> int:
    version = repo_version(root)
    name, wheel_version, _, _ = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != NAME or wheel_version != version:
        print(
            f"ERROR: {wheel.name} is not {NAME} {version}, the version in "
            "smlm-view3d/pyproject.toml"
        )
        return 2
    try:
        release = _pypi_release(version)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"ERROR: could not ask PyPI about {NAME} {version}: {exc}")
        return 2
    if release is None:
        print(
            f"OK: {NAME} {version} is not on PyPI yet; the next napari-storm "
            "release publishes it."
        )
        return 0

    with tempfile.TemporaryDirectory() as folder:
        published = _download_wheel(release, Path(folder))
        theirs, their_metadata = _contents(published)
    ours, our_metadata = _contents(wheel)

    changed = sorted(n for n in ours.keys() & theirs.keys() if ours[n] != theirs[n])
    added = sorted(ours.keys() - theirs.keys())
    removed = sorted(theirs.keys() - ours.keys())
    metadata_changed = our_metadata != their_metadata
    if not (changed or added or removed or metadata_changed):
        print(
            f"OK: {wheel.name} matches {NAME} {version} on PyPI; a napari-storm "
            "release leaves it as it is."
        )
        return 0

    print(
        f"ERROR: {NAME} {version} is already on PyPI, and the wheel built from "
        "this tree differs from it:"
    )
    for label, names in (("changed", changed), ("added", added), ("removed", removed)):
        for n in names:
            print(f"  {label}: {n}")
    if metadata_changed:
        for line in sorted(set(our_metadata) - set(their_metadata)):
            print(f"  requirement added: {line}")
        for line in sorted(set(their_metadata) - set(our_metadata)):
            print(f"  requirement removed: {line}")
    print(
        f"A napari-storm release would skip {NAME} and ship against the old "
        f"one. Bump smlm-view3d/pyproject.toml to {next_patch(version)} (or "
        f"higher) and setup.cfg's requirement to {NAME}>={next_patch(version)}."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the repository root (default: this script's parent folder)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pin", help="setup.cfg requires smlm-view3d>=its version")
    published = commands.add_parser(
        "published", help="a version PyPI has must match the built wheel"
    )
    published.add_argument("wheel", type=Path, help="the built smlm-view3d wheel")
    args = parser.parse_args(argv)
    if args.command == "pin":
        return check_pin(args.root)
    return check_published(args.root, args.wheel)


if __name__ == "__main__":
    sys.exit(main())
