"""Moved to ``storm_drawing.particles.summed_contrast``, which needs no napari.

This name is the same module object, so existing imports keep working.
"""

import sys

from storm_drawing.particles import summed_contrast as _moved

sys.modules[__name__] = _moved
