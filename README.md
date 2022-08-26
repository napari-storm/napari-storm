![alt text](/ressources/napari_storm_logo.png)

# napari-storm

[![License](https://img.shields.io/pypi/l/napari-storm.svg?color=green)](https://github.com/napari-storm/napari-storm/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-storm.svg?color=green)](https://pypi.org/project/napari-storm)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-storm.svg?color=green)](https://python.org)
[![tests](https://github.com/napari-storm/napari-storm/workflows/tests/badge.svg)](https://github.com/napari-storm/napari-storm/actions)
[![codecov](https://codecov.io/gh/napari-storm/napari-storm/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-storm/napari-storm)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-storm)](https://napari-hub.org/plugins/napari-storm)

A plugin for interactive visualization of Single Molecule Localization Microscopy (SMLM) datasets with Napari.  This package uses the (currently experimental) Napari Particles layer developed by Martin Weigert (https://github.com/maweigert).

----------------------------------


## Installation

We recommend instaling Napari into a Conda virtual environment, e.g. using the conda prompt with 

    conda create --name napari-storm
    
    activate napari-storm
    
    conda install pip

You can install `napari-storm` by cloning the repository, using, e.g. the conda prompt with:

    git clone https://github.com/napari-storm/napari-storm
    
Next, switch to the install directory and install the python package using 

    cd napari-storm

    pip install -e .
    
To start the program, run the napari_start.py, e.g. by navigating in your anaconda prompt to the location of the 
napri_start.py and run it with:

    python.exe napari_start.py 

Or simply start the napari version that was just installed into your environment, e.g. again using the conda prompt:  

    napari 

## Usage

### Starting napari-storm
If you started napari via console and not napari_start.py first thing to do is to activate napari-storm, found in 
the menu bar under plug-ins. 

### Importing data into napari-storm
Standard file formats from picasso, thunderstorm, AI Minflux files,... can be imported by simply dragging them into
napari or the import file dialog. If that's not the case for the file:

- one can either write a custom import function by following the instructions in the src/napari_storm/Custom_Import.py
- try the (rather experimental) file recognition import button, which will try to extract the headers of your file
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
