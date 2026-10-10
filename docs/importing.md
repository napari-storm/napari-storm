# Importing data

## Opening a file

Use **Import Localization File…** in the Data Controls tab, or drop a file
onto the napari window. napari-storm decides how to read it from its
extension and, where one extension covers several layouts, from its
contents -- there is nothing to choose.

Reading happens in the background. The window stays usable, a progress
dialog appears for anything slower than a moment, and **Cancel** abandons the
import. A cancelled read that is already under way finishes and is then
thrown away, so cancelling never leaves half a dataset loaded. Building the
napari layer happens afterwards on the interface thread, so the window pauses
briefly at the end of a large import.

Opening a file replaces what is loaded. To add one instead, see
[Several datasets at once](#several-datasets-at-once).

## Supported formats

| Extension | What it is | Notes |
|---|---|---|
| `.hdf5`, `.h5`, `.hdf` | Picasso localizations (a `locs` table) or a daxview molecule set (a `molecule_set_data` group) | Told apart by what the file contains |
| `.yaml` | Picasso's metadata file | Opens the `.hdf5` of the same name beside it |
| `.csv` | ThunderSTORM export | Column names such as `x [nm]` and `uncertainty_xy [nm]` are recognised |
| `.smlm` | The SMLM zip format | |
| `.npy`, `.json` | Abberior MINFLUX, either Imspector layout | See [MINFLUX](minflux.md) |
| `.mat`, `.pmx`, `.zarr` | Abberior MINFLUX, Imspector 24.10 and later | `.zarr` needs the `[minflux]` extra |
| `.mfx` | MINFLUX, as napari-storm itself exports it | |
| `.ns` | napari-storm's own HDF5 format | Written by the [Data adjustment](filtering.md#adjusting-values) tab |

Raw camera movies (`.tif`, `.tiff`, `.dat`, `.raw`) are refused with an
explanation: napari-storm draws localizations, it does not localize. Fit the
movie first -- in Picasso or ThunderSTORM, for example -- and open the
result. A TIFF can still be shown *beside* the data as a
[reference image](#reference-images).

If a STORM file has no pixel size recorded, napari-storm asks for one.

## Formats it does not know

Two buttons under **Advanced:** handle the rest.

**Auto-detect Format** reads `.npy`, `.txt`, `.csv` and HDF5 files of any
layout and asks you, in a series of dialogs:

1. which table in the file holds the localizations (HDF5 only),
2. what kind of data it is -- plain localizations, STORM/PALM, or MINFLUX,
3. which column is x, y, z, photons, uncertainty and so on,
4. whether the data is 3-D.

**Custom** runs a function you write yourself, `custom_import_function` in
`src/napari_storm/localization_dataset_types/Custom_Import.py`. The comments
in that file explain which dataset class to build and which columns each one
needs. Keep a copy of your version somewhere else: reinstalling
napari-storm replaces the file.

## Several datasets at once

**⊕ Merge with Additional File** adds a file to what is already loaded instead
of replacing it. Each dataset becomes its own channel, with its own card in
Data Controls: its own colormap, contrast, visibility and alignment. A merge
cannot mix 2-D and 3-D data.

Each card has:

* a checkbox to **show or hide** the dataset;
* **Unload**, which removes only that dataset and its filters;
* **Reset**, which puts its contrast and opacity back to the defaults;
* **Shift [µm]:** x, y and z boxes that move the dataset without changing a
  single localization -- for aligning two channels by eye -- and a
  **Reset** beside them that returns it to where it was measured.

The **File Infos** tab shows one card per dataset: its type, how many
localizations it has (and how many pass the current filters), how many are
drawn, its extent, pixel size and uncertainties.

## Reference images

**Import Reference Image…** overlays a widefield image, a TIFF stack or any
PNG/JPEG on the localizations. The dialog asks for its pixel size in XY and Z
and where it sits in the data's coordinates, and for a single plane, its
orientation: **XY**, **XZ** or **YZ**.

Each reference image gets its own card, above the dataset cards:

* **Colormap** and a two-handle **Contrast** slider for grayscale images
  (napari applies neither to RGB, so they are not offered there);
* **Opacity** for RGBA images, multiplying the alpha they already have;
* the pixel size and position, editable after import;
* **↶/↷** beside each position field, rotating the image by 90° about that
  axis while keeping its centre in place. Rotating about X or Y switches the
  viewer to 3-D, since the result is no longer a flat XY slice;
* **✕ Remove**.

Adding or removing a reference image does not move the camera.
