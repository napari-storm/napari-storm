# napari-storm

[![License](https://img.shields.io/pypi/l/napari-STORM.svg?color=green)](https://github.com/napari-storm/napari-storm/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-STORM.svg?color=green)](https://pypi.org/project/napari-storm)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-STORM.svg?color=green)](https://python.org)
[![tests](https://github.com/LREIN663/napari-STORM/workflows/tests/badge.svg)](https://github.com/napari-storm/napari-storm/actions)
[![codecov](https://codecov.io/gh/LREIN663/napari-STORM/branch/main/graph/badge.svg)](https://codecov.io/gh/napari-storm/napari-STORM)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-STORM)](https://napari-hub.org/plugins/napari-storm)

A simple Plugin to visualize STORM data with napari

----------------------------------


## Installation

I would recommend instaling this into a virtual enviorment, e.g. using conda prompt with 

    conda create --name napari-STORM
    
    activate napari-STORM

You can install `napari-STORM` via [pip]:

    pip install napari-STORM
    

    
## Getting Startet

Best functionality is provided when you start it from python (e.g. type python.exe into conda prompt) and copy the following:
    
    import napari
    from napari_storm import *
    v = napari.Viewer()
    widget=SMLMQW(v)
    print(widget)
    v.window.qt_viewer.dockLayerControls.setVisible(False)
    v.window.add_dock_widget(widget,area='left',name='napari-STORM')
    napari.run()
    
    
but in principle the widget should be available whenever you start napari in the enviorment you installed this in, e.g. using 
    
    napari
    
in you conda command prompt. For furhter information stick to https://github.com/napari/napari



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

[file an issue]: https://github.com/LREIN663/napari-STORM/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
