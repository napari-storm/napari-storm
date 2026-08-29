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
To start the program, run the napari_start.py, e.g. by navigating in your anaconda prompt to the location of the 
napri_start.py (root) and run it with:

    python.exe napari_start.py 

Or simply start the napari version that was just installed into your environment, e.g. again using the conda prompt:  

    napari 

and then opening napari-storm in the plugins tab.

### Importing data into napari-storm
Drag & drop onto the dock widget supported file types (Picasso, ThunderSTORM, MINFLUX, etc.) directly into napari or use the import file dialog.

MINFLUX data is read in both Abberior layouts: the original one, and the flat
layout Imspector writes from **24.10** onwards. The newer one is accepted as
`.npy`, `.json`, `.mat`, `.zarr`, and as pyMINFLUX's own `.pmx`. Which layout a
file uses is determined from the file, so there is nothing to choose. A `.zarr`
dataset is a folder rather than a file; since a file dialog cannot select one,
pick any file inside it and the whole dataset opens.

If your file is not covered:


- one can either write a custom import function by following the instructions in the src/napari_storm/Custom_Import.py
- try the (experimental) file recognition import button, which will try to extract the headers of your file
and lets you assign your data. This should work for any .hdf5, .csv or npy. file. 

### Basic usage
When a dataset is imported you should be able to see 4 tabs in the widget: Data Controls, File Infos, Decorators and 
Data Filter. In the data controls tab you can change the render range, load 
a new file, merge the currently open dataset with another that from another file and change your view.
There is also the option to change the colormap, adjust the contrast with the slider beneath the colormap picking as well
as adding a scalebar or active rainbow colorcoding (for 3D datasets).

The File Infos tab simply displays information on the currently opened datasets.

In the decorators tab you can activate a grid plane and customize a lot of things for the grid as well as the render range box. 

Last but not least is the data filter tab, which gives you the option to filter your displayed datasets by all properties available in the dataset.
There you will find two sliders, where the top one lets you change the x-range of the displayed property and the other one controlls 
the cut-off/cut-on of your filter. To apply the filter settings to the dataset simply press 
one of the apply buttons.

### Tips
- Double click or drag the tabs anywhere to detach them from the window. This way you have an overview over all of them at the same time
- For STORM/PALM ... datasets, it is possible to change the rendering options in the data controls tab to "variabel gaussian mode", to include the uncertainty values or photon counts for the rendering
- hold shift and drag the mouse for panning

## Documentation
There is a custom Q&A GPT for this repo specifically, available at 
https://chatgpt.com/g/g-68aebb6371a88191877094b48513d690-napari-storm-q-a

To access the documentation install the following packages (once):
```bash
pip install mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions
```

Then build and serve the docs:

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

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/napari-storm/napari-storm/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
