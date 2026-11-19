# Changelog

What changed for users in each release. Developers will find the full history
in the [git log](https://github.com/napari-storm/napari-storm/commits/main).

## Unreleased

* **Scenes use the data already open.** Loading a scene no longer reads a file
  again that is open as it was loaded: split parts are cut from it, and
  filters and density the scene records are replayed onto it. **Save
  Scene…** and **Load Scene…** start in the folder of the current dataset,
  with a name taken from its file (`cells.h5` -> `cells.napari-storm.json`).
* **The view rotates about what you are looking at.** Starting a rotation in
  3-D moves the centre of rotation along the line of sight onto the
  localizations in the middle of the view, so the structure there stays put
  as you turn; double-click a structure to make it the centre. A grey dot
  marks the centre while the camera moves. See [The centre of
  rotation](navigating.md#the-centre-of-rotation).
* **Focus mode** fades out localizations further than a radius from the centre
  of the view, down to an opacity you choose, and follows the camera. It sits
  under **Reset view** on the Data Controls tab, with the rotation option, and
  is saved in scenes. See [Focus mode](navigating.md#focus-mode).
* **The XY / YZ / XZ buttons are evenly spaced** whatever the display scaling.
* **Local density.** **Calculate local density** on the Data Filter tab gives
  every localization a `local_density` parameter: how many other localizations
  lie within a search radius (35 nm by default; a sphere in 3-D, a circle in
  2-D), counting only those the filters applied so far keep -- the measure
  daxview uses. Filter on it to remove isolated localizations, or colour by
  it. See [Local density](filtering.md#local-density).
* **Split a dataset by its filters.** Apply filters, then **Split current
  dataset: shown / filtered out**: the dataset becomes two channels, what you
  see and everything the filters removed, each in its own colour -- dense and
  sparse localizations side by side, for instance. Scenes make the split again
  when they are loaded. See
  [Splitting a dataset in two](filtering.md#splitting-a-dataset-in-two).
* **Derived columns can be filtered and coloured by.** Group ids from STORM
  grouping now appear in the Data Filter tab and under **Colour code by**,
  like local density.
* **Applying a filter keeps the view.** **Apply filter** no longer re-centres
  the camera on the whole dataset, and Data adjustment arithmetic re-centres
  only when it moves the data. A click on the Data Filter tab's background no
  longer raises an error.
* **Load Scene reopens the data.** A scene is now a saved view: **Load
  Scene…** opens the files it names -- at their saved paths, or beside the
  scene if the folder moved -- replays the filters and adjustments, and
  restores the Gaussian widths, render range, grid plane, scale bar, channel
  controls, reference images and camera, perspective included. Answers typed in
  when a file was first opened, such as a pixel size, are not asked again.
  Scenes saved before this recorded no file paths -- every one said
  `"source_path": null` -- so they still apply only to data opened by hand;
  save them again to make them reopen their files. See [Scenes](export.md#scenes).
* **A hidden channel stays hidden** when a filter, crop or width change
  redraws the layers.
* **Colour coding by any property.** **Colour code by** replaces **Activate
  Rainbow colorcoding in Z**: colour by x, y, z or any numeric column -- frame,
  photons, uncertainty -- in 2-D as well as 3-D, with HSV (still the default),
  grey, perceptual or diverging colormaps. The limits are shared by every
  dataset, follow the data by default and can be typed. Overlapping
  localizations blend to their mean colour, so dense regions no longer turn
  white, and the contrast slider sets brightness from density. Colour coding
  is saved in scenes. See [Colour](rendering.md#colour).
* **Drift correction with COMET.** A new **Post-proc.** tab runs
  [COMET](https://github.com/gpufit/Comet) drift correction in the background,
  with progress, Cancel and a memory estimate before each run, and optional
  fiducial removal and STORM grouping before it. The correction is reversible
  until **Discard drift**, can be saved, loaded and applied to other channels,
  and a **Correction** slider shows raw to corrected in 10 % steps -- with the
  network of localizations that belong together, and an x, y, time view for
  2-D data. Install with the new `[comet]` extra (COMET is published as
  `comet-smlm`). See [Post-processing](post-processing.md).
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
