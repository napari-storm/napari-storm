"""Moved to ``storm_drawing.particles.footprint_shaders``, which needs no napari.

This name is the same module object, so existing imports keep working.
"""

import sys

from storm_drawing.particles import footprint_shaders as _moved

sys.modules[__name__] = _moved
