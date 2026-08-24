# napari-storm

**napari-storm** is a [napari](https://napari.org) plugin for **interactive visualization and exploration of Single Molecule Localization Microscopy (SMLM) data** (STORM, PALM, MINFLUX).  

Unlike voxel-based approaches, napari-storm renders each localization as a **billboarded Gaussian**, making it efficient enough to interactively explore **millions of points in 3D**.

---

## Features
- Import localizations from **Picasso HDF5, ThunderSTORM CSV, MINFLUX JSON/NPY/MFX**, or your own **custom format**.  
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
We recommend installing in a Conda environment:

```bash
conda create --name napari-storm python==3.11
conda activate napari-storm
conda install pip

git clone https://github.com/napari-storm/napari-storm
cd napari-storm
pip install -e .
```