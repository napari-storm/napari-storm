"""Moved to ``smlm_view3d.particles.vispy_compat``, which needs no napari.

This name is the same module object, so existing imports keep working.
"""

import sys

from smlm_view3d.particles import vispy_compat as _moved

sys.modules[__name__] = _moved
