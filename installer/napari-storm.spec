# PyInstaller spec for the napari-storm desktop app.
#
#     python -m PyInstaller installer/napari-storm.spec
#
# run from the repository root, in an environment that has napari-storm,
# smlm-view3d, PyQt6 and PyInstaller installed and nothing else it should ship
# -- installer/build.py makes one and runs this.  The result is a one-folder
# app in dist/: dist/napari-storm.app on macOS, dist/napari-storm/ (with
# napari-storm.exe) on Windows.
#
# Most of what is below exists because napari and its plugins are found at
# run time rather than imported, which PyInstaller's import analysis cannot
# see: napari's lazy top-level module, npe2's entry-point plugin discovery
# (napari's built-in readers, napari-svg, napari-storm's own reader), and
# VisPy's GL backend modules, which it imports by name.  The gl+ backend in
# particular -- vispy.gloo.gl.glplus, over PyOpenGL -- is what instanced
# rendering needs; without it the app still starts, on the slower billboard
# renderer, which is the failure the smoke test in build.py exists to catch.

import os
import sys
from importlib.metadata import version

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

ROOT = os.path.dirname(SPECPATH)  # noqa: F821 - defined by PyInstaller
APP_NAME = "napari-storm"
APP_VERSION = version("napari-storm")


def _not_tests(name):
    # napari_storm.core is an alias: at import it enters smlm_view3d.core's
    # modules in sys.modules under its own name, so its submodules exist only
    # as smlm_view3d's, which are collected below.
    if name.startswith("napari_storm.core."):
        return False
    return not any(
        part in ("_tests", "tests", "benchmarks", "conftest")
        for part in name.split(".")
    )


# --- what is imported by name rather than by an import statement -------------

hiddenimports = []
for package in (
    "napari",  # lazy_loader.attach: napari.Viewer etc. resolve on first use
    "napari_builtins",  # npe2 commands, named as "module:function" strings
    "napari_svg",
    "napari_storm",
    "smlm_view3d",
    "magicgui.backends._qtpy",
    "app_model.backends.qt",
):
    hiddenimports += collect_submodules(package, filter=_not_tests)
hiddenimports += [
    # VisPy's GL backends (vispy.gloo.gl.use_gl) and app backends
    # (vispy.app.application), both imported by name.
    "vispy.gloo.gl.glplus",
    "vispy.gloo.gl.gl2",
    "vispy.app.backends._qt",
    "vispy.app.backends._pyqt6",
    # PyOpenGL's platform module; hook-OpenGL adds it, named here as well
    # because gl+ is useless without it.
    "OpenGL.platform.darwin" if sys.platform == "darwin" else "OpenGL.platform.win32",
]

# --- data files and the metadata entry-point discovery reads -----------------

datas = []
for package in ("napari", "napari_builtins", "napari_svg", "vispy", "npe2"):
    datas += collect_data_files(package, excludes=["**/_tests/**", "**/tests/**"])
datas += collect_data_files("napari_storm", excludes=["**/_tests/**"])
datas += collect_data_files("smlm_view3d")
for distribution in (
    # npe2 finds plugins by their `napari.manifest` entry points, which live
    # in the distribution metadata; napari-storm's reader is one of them.
    "napari",
    "napari-svg",
    "napari-storm",
    "smlm-view3d",
    "npe2",
    # Read with importlib.metadata.version() at import time.
    "magicgui",
    "app-model",
    "superqt",
    "psygnal",
    "in-n-out",
    "vispy",
):
    datas += copy_metadata(distribution)

# --- left out -----------------------------------------------------------------

excludes = [
    # napari's IPython console.  Without it napari says the console is not
    # installed when its button is pressed; with it the bundle carries an
    # IPython kernel, ZeroMQ and Jupyter's client stack.
    "napari_console",
    "qtconsole",
    "IPython",
    "ipykernel",
    "jupyter_client",
    "jupyter_core",
    "zmq",
    "debugpy",
    "jedi",
    "parso",
    "tornado",
    "matplotlib_inline",
    # Test frameworks and a GUI toolkit nobody calls.
    "pytest",
    "_pytest",
    "pytestqt",
    "tkinter",
    "_tkinter",
    # Other Qt bindings: PyInstaller refuses to collect two.
    "PyQt5",
    "PySide2",
    "PySide6",
]

# --- the app icon, made from the logo ----------------------------------------


def _make_icon():
    """Square the logo onto a transparent canvas and save it as .icns/.ico."""
    from PIL import Image

    logo = Image.open(os.path.join(ROOT, "resources", "napari_storm_logo_small.png"))
    logo = logo.convert("RGBA")
    side = max(logo.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
    square = square.resize((1024, 1024), Image.LANCZOS)
    folder = os.path.join(workpath, "icon")  # noqa: F821 - defined by PyInstaller
    os.makedirs(folder, exist_ok=True)
    if sys.platform == "darwin":
        path = os.path.join(folder, "napari-storm.icns")
        square.save(path)
    else:
        path = os.path.join(folder, "napari-storm.ico")
        square.save(path, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    return path


ICON = _make_icon()

a = Analysis(  # noqa: F821
    [os.path.join(SPECPATH, "launcher.py")],  # noqa: F821
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    # The Qt canvases the Data Filter and Post-processing tabs draw plots
    # on.  Without this the hook also chases imageio's TkAgg, and tkinter is
    # excluded.
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Qt5Agg", "Agg"]}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821


def _macos_minimum(binaries):
    """The newest macOS any collected binary was built for, as "14.4".

    The wheels decide this, not the build machine: PyQt6 6.12's Qt needs 14.4,
    for one.  Info.plist says it, so that an older Mac refuses the app with a
    message rather than a crash.
    """
    from PyInstaller.utils.osx import macosx_version_min

    newest = (11, 0, 0)
    for _dest, source, _kind in binaries:
        try:
            newest = max(newest, macosx_version_min(source))
        except Exception:  # noqa: BLE001 - not a Mach-O file, or unreadable
            continue
    major, minor, patch = newest
    return f"{major}.{minor}" + (f".{patch}" if patch else "")

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Windowed: no console window behind the viewer.  launcher.py sends output
    # to a log file instead.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier="io.github.napari-storm",
        version=APP_VERSION,
        info_plist={
            "CFBundleDisplayName": "napari-storm",
            "CFBundleName": "napari-storm",
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            # Follow the system's light/dark setting.
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": _macos_minimum(a.binaries),
            "NSHumanReadableCopyright": "BSD-3-Clause",
        },
    )
