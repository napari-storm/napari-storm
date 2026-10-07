# Changelog

What changed for users in each release. Developers will find the full history
in the [git log](https://github.com/napari-storm/napari-storm/commits/main).

## Unreleased

* **zarr is optional.** `zarr<3` moved from the requirements into a new
  `[minflux]` extra, so napari-storm no longer holds an environment back on
  zarr 2. Only MINFLUX `.zarr` stores need it; opening one without it says
  what to install. See [Zarr stores](minflux.md#zarr-stores).
* **New documentation**, organised by task, with screenshots generated from
  the sample data by `scripts/make_doc_images.py`.

## 3.1.0 -- 2026-10-03

* **MINFLUX traces.** **Connect traces** in the Decorators tab draws each
  trace as a line through its localizations in measurement order, coloured by
  trace, time, progress or any numeric column, and saved in scenes. See
  [Traces](minflux.md#traces).
* **Gaussians are drawn at the width the settings ask for.** Before 3.1 the
  canvas drew them at about 0.88 of the width entered, and distorted
  anisotropic ones. Expect the canvas to look slightly wider than in 3.0 at
  the same settings.

## 3.0.1 -- tagged only

* A drawing fix for the fallback (billboard) renderer. This tag was not
  published to PyPI; the fix is part of 3.1.0.

## 3.0.0 -- 2026-10-02

* **Rendering styles.** The Decorators tab offers sixteen alternative
  visualisations beside the Gaussian -- markers, spheres, rings, uncertainty
  ellipses, napari-particles' sprites -- with **Use uncertainty** and
  **Smallest on screen [px]:**. See [Rendering styles](styles.md).
* **Contrast acts on the summed image.** The contrast slider's numbers now
  count overlapping localizations, with a **Cutoff** that hides sparse regions
  and a **×Range** that sets full brightness. See
  [Contrast](rendering.md#contrast).
* **A `napari-storm` command** opens napari with the dock already docked;
  `python -m napari_storm` does the same.
* The dock grows to fit a dataset's controls.
* Documentation on Read the Docs.

## 2.1.0 -- 2026-09-05

* **MINFLUX from Imspector 24.10 and later**, in `.npy`, `.json`, `.mat`,
  `.zarr` and `.pmx`, with the layout read from the file rather than the
  extension.
* HDF5 files are routed by their contents: Picasso and daxview molecule sets
  both open from `.hdf5`/`.h5`.
* A Picasso `.hdf5` without its `.yaml` says why it did not open.
* **Export** a 2-D projection onto XZ and YZ as well as XY.
* napari's own file menu offers every format the importer reads.
* The view buttons point the camera at the plane they name.
