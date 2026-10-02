"""Start napari with the napari-storm dock from a source checkout.

Installed, the same launcher is the ``napari-storm`` command; it lives in
``src/napari_storm/__main__.py`` so that it ships with the package.
"""

import sys

from napari_storm.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
