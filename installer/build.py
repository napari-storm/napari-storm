"""Build the napari-storm desktop installer for this platform.

    python installer/build.py

Run from anywhere, with the Python of an environment that holds napari-storm,
smlm-view3d, PyQt6 and PyInstaller and nothing else the app should ship (see
installer/README.md).  Three steps, each of which can be skipped:

1. freeze -- PyInstaller, with installer/napari-storm.spec, into dist/;
2. smoke  -- start the frozen app on the sample data with the smoke test
   switched on (installer/launcher.py), and fail unless it printed the gl+
   line and drew the data with the instanced renderer;
3. package -- macOS: a .dmg holding the .app and a link to /Applications,
   made with hdiutil.  Windows: an Inno Setup installer, made with iscc.

The installer lands in dist/ as napari-storm-<version>-<platform>.<ext>, and
the sizes are printed at the end.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = HERE / "napari-storm.spec"
ISS = HERE / "napari-storm.iss"
SAMPLE = ROOT / "sample_data" / "beta_ii_spectrin.storm_4pi.h5"

# Kept in step with installer/launcher.py, which writes them.
GL_LINE = "GL backend: gl+ (instancing available)"
PASSED = "SMOKE TEST PASSED"

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"


def _app_folder():
    return DIST / ("napari-storm.app" if IS_MAC else "napari-storm")


def _executable():
    if IS_MAC:
        return _app_folder() / "Contents" / "MacOS" / "napari-storm"
    if IS_WIN:
        return _app_folder() / "napari-storm.exe"
    return _app_folder() / "napari-storm"


def _arch():
    machine = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)


def _installer_path(app_version):
    if IS_MAC:
        return DIST / f"napari-storm-{app_version}-macos-{_arch()}.dmg"
    return DIST / f"napari-storm-{app_version}-windows-{_arch()}-setup.exe"


def _size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()
    )


def _mb(n_bytes):
    return f"{n_bytes / 1e6:.0f} MB"


def freeze():
    print(f"== freeze: PyInstaller {version('pyinstaller')}, {SPEC.name}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(SPEC),
            "--noconfirm",
            "--clean",
            "--distpath",
            str(DIST),
            "--workpath",
            str(BUILD / "pyinstaller"),
        ],
        cwd=ROOT,
        check=True,
    )


def smoke(timeout):
    """Start the frozen app on the sample file; raise unless it passed."""
    executable = _executable()
    report = BUILD / "smoke-report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    print(f"== smoke: {executable.relative_to(ROOT)} {SAMPLE.name}", flush=True)
    env = dict(os.environ, NAPARI_STORM_SMOKE_TEST=str(report))
    started = time.monotonic()
    try:
        result = subprocess.run(
            [str(executable), str(SAMPLE)], env=env, timeout=timeout
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        returncode = "timeout"
    elapsed = time.monotonic() - started
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    print(text or "(the app wrote no report)")
    print(f"exit status {returncode} after {elapsed:.0f} s")
    problems = []
    if GL_LINE not in text:
        problems.append(f"the report has no {GL_LINE!r} line")
    if PASSED not in text:
        problems.append(f"the report has no {PASSED!r} line")
    if returncode != 0:
        problems.append(f"the app exited with {returncode}")
    if problems:
        raise SystemExit("smoke test failed: " + "; ".join(problems))
    print("smoke test passed")


def package_mac(app_version):
    dmg = _installer_path(app_version)
    staging = BUILD / "dmg"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    # ditto keeps the bundle's symlinks, extended attributes and signatures.
    subprocess.run(
        ["ditto", str(_app_folder()), str(staging / "napari-storm.app")], check=True
    )
    (staging / "Applications").symlink_to("/Applications")
    dmg.unlink(missing_ok=True)
    print(f"== package: {dmg.name}", flush=True)
    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            f"napari-storm {app_version}",
            "-srcfolder",
            str(staging),
            "-fs",
            "HFS+",
            "-format",
            "UDZO",
            "-imagekey",
            "zlib-level=9",
            "-ov",
            str(dmg),
        ],
        check=True,
    )
    return dmg


def _iscc():
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            candidate = Path(base) / "Inno Setup 6" / "ISCC.exe"
            if candidate.exists():
                return str(candidate)
    raise SystemExit(
        "Inno Setup's compiler (ISCC.exe) was not found: install Inno Setup 6 "
        "(https://jrsoftware.org/isinfo.php, or `choco install innosetup`)"
    )


def package_windows(app_version):
    setup = _installer_path(app_version)
    print(f"== package: {setup.name}", flush=True)
    subprocess.run(
        [
            _iscc(),
            f"/DAppVersion={app_version}",
            f"/DSourceDir={_app_folder()}",
            f"/DOutputDir={DIST}",
            f"/DOutputBaseFilename={setup.stem}",
            f"/DIconFile={BUILD / 'pyinstaller' / 'napari-storm' / 'icon' / 'napari-storm.ico'}",
            str(ISS),
        ],
        check=True,
    )
    return setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-freeze", action="store_true", help="reuse dist/")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=600,
        help="seconds before the smoke test is called hung (default 600)",
    )
    args = parser.parse_args(argv)

    if not (IS_MAC or IS_WIN):
        raise SystemExit("installers are built on macOS and Windows only")
    app_version = version("napari-storm")
    print(f"napari-storm {app_version}, Python {platform.python_version()}, {_arch()}")

    if not args.skip_freeze:
        freeze()
    if not args.skip_smoke:
        smoke(args.smoke_timeout)
    installer = None
    if not args.skip_package:
        installer = package_mac(app_version) if IS_MAC else package_windows(app_version)

    print("== sizes")
    print(f"{_app_folder().relative_to(ROOT)}: {_mb(_size(_app_folder()))}")
    if installer is not None:
        print(f"{installer.relative_to(ROOT)}: {_mb(_size(installer))}")
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
                out.write(f"installer={installer}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
