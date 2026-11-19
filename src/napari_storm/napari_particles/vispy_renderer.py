"""Moved to ``storm_drawing.particles.vispy_renderer``, which needs no napari.

This name is the same module object, so existing imports keep working.
"""

import sys

from storm_drawing.particles import vispy_renderer as _moved

sys.modules[__name__] = _moved
