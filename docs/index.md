# napari-storm

**napari-storm** is a [napari](https://napari.org) plugin for **interactive visualization and exploration of Single Molecule Localization Microscopy (SMLM) data** (STORM, PALM, MINFLUX).

Unlike voxel-based approaches, napari-storm renders each localization as a **billboarded Gaussian**, making it efficient enough to interactively explore **millions of points in 3D**.

---

## Features
- Import localizations from **Picasso HDF5, ThunderSTORM CSV, MINFLUX JSON/NPY/MFX**, or your own **custom format**.
- Read MINFLUX datasets from **Imspector 24.10 and later** as well, in every container it and [pyMINFLUX](https://pyminflux.ethz.ch/) write: `.npy`, `.json`, `.mat`, `.zarr` and `.pmx`. Which layout a file uses is worked out from the file itself, so nothing has to be selected.
- GPU-accelerated rendering of millions of points via **napari-particles**.
- Adjustable point spread functions (fixed / variable Gaussian).
- Multi-channel colormaps with per-channel contrast/opacity controls.
- Interactive histogram-based filtering.
- Overlays: grid planes, scalebars, and 3D camera views.
- Export data in multiple formats, including **calibrated OME-TIFF** at a pixel
  size you choose, which never downsamples to fit.
- **Embeddable**: a host application can render localizations through the API
  with no dock widget — see [embedding.md](embedding.md).

---

## Installation

napari-storm needs Python 3.10–3.12. We recommend its own environment:

```bash
conda create --name napari-storm python=3.11 pip
conda activate napari-storm
```

Then install from PyPI. The `[pyqt6]` extra brings in napari's Qt backend —
without an extra no Qt binding is installed and napari cannot open a window.
Use `[pyside6]` if you prefer PySide, or no extra at all if you are installing
into an application that already provides a binding:

```bash
pip install "napari-storm[pyqt6]"
```

To work on napari-storm itself, install from a clone instead:

```bash
git clone https://github.com/napari-storm/napari-storm
cd napari-storm
pip install -e ".[dev,pyqt6]"
```

Start napari and open **Plugins → napari-storm**; the
[step-by-step tutorial](step-by-step.md) takes it from there.
