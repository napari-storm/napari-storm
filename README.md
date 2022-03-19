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
    
To start the program, simply start the napari version that was just installed into your environment, e.g. again using the conda prompt:  

    napari 
    


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
