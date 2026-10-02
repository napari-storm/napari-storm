![napari-storm](https://raw.githubusercontent.com/napari-storm/napari-storm/main/resources/napari_storm_logo.png)

# napari-storm

A plugin for interactive visualization of Single Molecule Localization Microscopy (SMLM) datasets with Napari.  This package uses the (currently experimental) Napari Particles layer developed by Martin Weigert (https://github.com/maweigert).

----------------------------------


## Installation

napari-storm needs Python 3.10–3.12. It's recommended to install it into its
own environment, e.g. with conda:

    conda create --name napari-storm python=3.11 pip

    conda activate napari-storm

Then install from PyPI. The `[pyqt6]` extra brings in napari's Qt backend;
without an extra, no Qt binding is installed and napari cannot open a window
(use `[pyside6]` if you prefer PySide):

    pip install "napari-storm[pyqt6]"

To work on napari-storm itself, install from a clone instead:

    git clone https://github.com/napari-storm/napari-storm

    cd napari-storm

    pip install -e ".[dev,pyqt6]"



## Usage

### Starting napari-storm
With the environment active -- an Anaconda Prompt after `conda activate napari-storm`, for
example -- run:

    napari-storm

This opens napari with the napari-storm dock already docked on the right. `python -m napari_storm`
does the same, and from a clone so does `python napari_start.py`.

Or start napari itself and open the dock from the Plugins menu:

    napari

### Importing data into napari-storm
Drag & drop onto the dock widget supported file types (Picasso, ThunderSTORM, MINFLUX, etc.) directly into napari or use the import file dialog.

MINFLUX data is read in both Abberior layouts: the original one, and the flat
layout Imspector writes from **24.10** onwards. The newer one is accepted as
`.npy`, `.json`, `.mat`, `.zarr`, and as pyMINFLUX's own `.pmx`. Which layout a
file uses is determined from the file, so there is nothing to choose. A `.zarr`
dataset is a folder rather than a file; since a file dialog cannot select one,
pick any file inside it and the whole dataset opens.

If your file is not covered:


- one can either write a custom import function by following the instructions in
  `src/napari_storm/localization_dataset_types/Custom_Import.py`, then run it with the **Custom** button
- try the (experimental) **Auto-detect Format** button, which will try to extract the headers of your file
and lets you assign your data. This should work for any .hdf5, .csv or .npy file.

### Basic usage
When a dataset is imported you should be able to see five tabs in the widget: Data Controls, File Infos, Decorators,
Data Filter and Data adjustment. In the data controls tab you can change the render range, load
a new file, merge the currently open dataset with another file and change your view.
There is also the option to change the colormap, add a scalebar or activate rainbow colour coding (for 3D datasets),
and adjust the contrast with the slider beneath the colormap. The contrast acts on the reconstruction itself -- the
Gaussians summed where they overlap -- so its lower handle hides sparse regions while keeping dense ones, and its upper
handle sets how many overlapping localizations it takes to reach full brightness.

The File Infos tab simply displays information on the currently opened datasets.

In the decorators tab you can switch the rendering style from the scientific Gaussian to an alternative visualisation -- points, spheres,
uncertainty ellipses and more -- activate a grid plane and customize a lot of things for the grid as well as the render range box.

Last but not least is the data filter tab, which gives you the option to filter your displayed datasets by all properties available in the dataset.
There you will find two sliders, where the top one lets you change the x-range of the displayed property and the other one controls
the cut-off/cut-on of your filter. To apply the filter settings to the dataset simply press
one of the apply buttons.

### Tips
- Double click or drag the tabs anywhere to detach them from the window. This way you have an overview over all of them at the same time
- For STORM/PALM ... datasets, it is possible to change the rendering options in the data controls tab to **Variable-size gaussian**, to include the uncertainty values or photon counts for the rendering
- hold shift and drag the mouse for panning

## Documentation

Full documentation is at **https://napari-storm.readthedocs.io/** — a
step-by-step tutorial, how the renderer works, and the embedding API for
driving napari-storm from another application.

There is also a custom Q&A GPT for this repo specifically, available at
https://chatgpt.com/g/g-68aebb6371a88191877094b48513d690-napari-storm-q-a

To build the docs locally, install the pinned toolchain (once):

```bash
pip install -r docs/requirements.txt
```

Then serve them with live reload:

```bash
mkdocs serve
```

## Acknowledgements

napari-storm builds on work by others, with thanks:

- **[napari-particles](https://github.com/maweigert/napari-particles)** by Martin
  Weigert (BSD-3-Clause), the experimental Particles layer this plugin renders
  through. The modules derived from it live in
  `src/napari_storm/napari_particles/` and carry their own `NOTICE`.
- **[pyMINFLUX](https://pyminflux.ethz.ch/)** by the Single Cell Facility of the
  D-BSSE, ETH Zurich ([source](https://github.com/bsse-scf/pyMINFLUX),
  Apache-2.0). Its `MinFluxReaderV2` is the reference for the MINFLUX layout
  Imspector writes from 24.10 onwards: the version markers that tell the two
  layouts apart, the per-container quirks, and the `.pmx` structure were all
  read from it. napari-storm's reader is an independent implementation and
  contains no pyMINFLUX code, but it would not exist without theirs having
  documented the format first.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/napari-storm/napari-storm/issues
