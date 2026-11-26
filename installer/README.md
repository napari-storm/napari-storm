# Desktop installers

napari-storm for people who do not use Python: a `.dmg` for macOS on Apple
Silicon and a setup `.exe` for 64-bit Windows.  Each holds its own Python,
napari, PyQt6 and the napari-storm dock, frozen with
[PyInstaller](https://pyinstaller.org) as a one-folder app, and starts the way
the `napari-storm` command does.

**Status: prototype.**  The installers are not signed, so macOS and Windows
warn before the first start (see [Signing](#signing)).

| File | What it is |
| --- | --- |
| `napari-storm.spec` | What PyInstaller collects, and why |
| `launcher.py` | The script that is frozen: starts the app, or runs the smoke test |
| `build.py` | Freeze, smoke-test and package, on macOS or Windows |
| `napari-storm.iss` | The Inno Setup script for the Windows installer |
| `requirements.txt` | PyInstaller, pinned |

`.github/workflows/installers.yml` runs `build.py` on GitHub's runners.

## Building

In a **fresh** environment: PyInstaller bundles whatever it finds imported,
so an environment that also holds torch, COMET or a second Qt binding gives a
larger, different app.  Python 3.12, as the release workflow uses.

macOS (Apple Silicon):

```bash
python3.12 -m venv build-env
build-env/bin/python -m pip install ./smlm-view3d
build-env/bin/python -m pip install ".[pyqt6]" -r installer/requirements.txt
build-env/bin/python installer/build.py
```

Windows, with [Inno Setup 6](https://jrsoftware.org/isinfo.php) installed:

```bash
py -3.12 -m venv build-env
build-env\Scripts\python -m pip install ./smlm-view3d
build-env\Scripts\python -m pip install ".[pyqt6]" -r installer/requirements.txt
build-env\Scripts\python installer\build.py
```

The app lands in `dist/` (`dist/napari-storm.app`, or `dist/napari-storm/`
with `napari-storm.exe`), the installer next to it as
`napari-storm-<version>-macos-arm64.dmg` or
`napari-storm-<version>-windows-x64-setup.exe`.  `--skip-freeze`,
`--skip-smoke` and `--skip-package` leave a step out.

On GitHub, a published `v<version>` release builds both and attaches them to
the release; a pull request that touches the installer builds both and keeps
them as workflow artifacts.

## The smoke test

`build.py` starts the frozen app on `sample_data/beta_ii_spectrin.storm_4pi.h5`
with `NAPARI_STORM_SMOKE_TEST` set, and fails the build unless the app:

* printed `GL backend: gl+ (instancing available)` -- VisPy's gl+ backend
  loads PyOpenGL's platform module by name, so a frozen app that misses it
  starts anyway, silently, on the slower billboard renderer;
* opened the file the way `napari-storm FILE` does, and drew it with the
  instanced renderer (an `InstancedParticles` layer);
* sees its own napari plugin (the reader napari uses for a file dropped on
  the viewer), which a frozen app finds only if the package metadata was
  bundled;
* rendered a frame that is not blank;

and exited within the timeout.  The report is `build/smoke-report.txt`; the
workflow keeps it as an artifact, pass or fail.  A frozen app started without
a console (Windows) writes its output to
`%LOCALAPPDATA%\napari-storm\napari-storm.log`.

## Sizes

napari-storm 3.3.0, Python 3.12, PyInstaller 6.22.3, built 2026-11-26:

| | App, unpacked | Installer |
| --- | --- | --- |
| macOS arm64 | 260 MB | 113 MB (`.dmg`, zlib) |
| Windows x64 | not yet built (CI) | -- |

The largest parts of the macOS app are PyQt6 (45 MB), SciPy (34 MB), pandas
(18 MB), scikit-image (14 MB), scikit-learn (13 MB) and Pillow (10 MB).  The
app needs macOS 14.4 or newer, because PyQt6 6.12's Qt does; `Info.plist`
says so, taken from the bundled libraries at build time.

## Left out, for now

* **napari's IPython console.**  Excluded with IPython, ipykernel, ZeroMQ and
  Jupyter's client; napari says the console is not installed when its button
  is pressed.  Adding it back means dropping those excludes and checking that
  the kernel starts in a frozen app.
* **COMET** (`[comet]`).  It pulls in numba and llvmlite -- llvmlite's LLVM
  library alone can be 100 MB or more -- and needs three things a frozen app
  does not give numba by itself: COMET's `.py` sources (`cache=True` in
  `comet.core.cpu_wrapper` needs a source file to key its cache on; collect
  `comet` with `module_collection_mode="py"`), a writable `NUMBA_CACHE_DIR`
  set by the launcher, and, once the macOS app is signed with the hardened
  runtime, the `com.apple.security.cs.allow-jit` entitlement.  Then install
  `.[pyqt6,comet]` and add a COMET run to the smoke test.
* **MINFLUX `.zarr`** (`[minflux]`, zarr < 3).  Smaller: install
  `.[pyqt6,minflux]`, add `zarr` and `numcodecs` to the spec's hidden imports
  and metadata (numcodecs registers codecs through entry points), and open a
  `.zarr` store in the smoke test.
* **Intel Macs and Linux.**  An Intel `.dmg` needs an Intel runner
  (`macos-15-intel`) as a second macOS entry in the workflow matrix.
* **Opening files from the Finder or Explorer** ("Open With", double-click).
  The command line opens files already; the Finder needs
  `CFBundleDocumentTypes` in `Info.plist` and PyInstaller's `argv_emulation`
  (first start only) or a Qt `FileOpen` event handler, and Windows needs file
  associations in the Inno Setup script.

## Signing

Nothing is signed yet beyond PyInstaller's ad-hoc signature, which Apple
Silicon requires for any code to run at all.  What signing would take:

### macOS: Developer ID and notarization (Gatekeeper)

An unsigned app downloaded from the web is quarantined: Gatekeeper refuses
the first start, and the user has to allow it under **System Settings →
Privacy & Security → Open Anyway** (or `xattr -dr com.apple.quarantine
/Applications/napari-storm.app`).  To start without a warning:

1. **Apple Developer Program membership** (paid, yearly), as an individual
   or an organisation (an organisation needs a D-U-N-S number).
2. A **Developer ID Application** certificate, created by the account holder,
   exported with its private key as a `.p12`.  For CI: the `.p12`
   (base64-encoded) and its password as repository secrets, imported into a
   temporary keychain on the runner.
3. **Sign every binary in the app** with that identity, the hardened runtime
   and a secure timestamp: set `codesign_identity` and `entitlements_file` in
   the spec (PyInstaller then signs each file it collects), then sign the
   `.dmg`.  Python under the hardened runtime usually needs entitlements such
   as `com.apple.security.cs.allow-unsigned-executable-memory` (ctypes, which
   PyOpenGL uses) and `com.apple.security.cs.disable-library-validation`; which
   are needed is found by testing the signed app.
4. **Notarize**: `xcrun notarytool submit <dmg> --wait`, authenticated with an
   App Store Connect API key (issuer ID, key ID, `.p8` file) or an Apple ID
   with an app-specific password and team ID -- again as secrets.  Then
   `xcrun stapler staple` the `.dmg`, so the check also passes offline.

### Windows: a code-signing certificate (SmartScreen)

An unsigned installer gets SmartScreen's "Windows protected your PC" (**More
info → Run anyway**), and an unknown publisher at every elevation prompt.

1. A **code-signing certificate** from a certificate authority (organisation-
   or individual-validated).  Since 2023 the private key must stay in a
   hardware module, so for CI that means a cloud signing service rather than a
   `.pfx` secret: Microsoft's signing service on Azure (Trusted Signing; check
   its eligibility rules for individuals and organisations), or a CA's own
   cloud HSM (DigiCert KeyLocker, SSL.com eSigner, Certum
   SimplySign, which also sells an open-source developer certificate).
2. **Sign** `napari-storm.exe`, then the installer and its uninstaller with
   `signtool sign /fd sha256 /tr <timestamp server> /td sha256` -- Inno Setup
   does the last two through its `SignTool` directive.
3. **Reputation.**  A signature names the publisher, but SmartScreen still
   warns until the certificate has built a download reputation.  Since 2024
   an EV certificate no longer skips that either.
