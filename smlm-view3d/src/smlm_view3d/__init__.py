"""napari-storm's drawing core, without napari: the 3D view of SMLM localizations.

- ``smlm_view3d.core``: the localization table, the render planner, the
  renderer contract and the scene; numpy only, with no Qt, napari or VisPy.
- ``smlm_view3d.particles``: the instanced VisPy billboards
  (``billboard_filter``, ``footprint_shaders``, ``summed_contrast``,
  ``vispy_compat`` and the ``instanced`` prototype) and the plain-VisPy
  backend, ``vispy_renderer.VispyInstancedRenderer``; VisPy and numpy, with no
  napari, and Qt only through VisPy's app.

napari-storm depends on this package.  Its old names, ``napari_storm.core``,
``napari_storm.core.<module>`` and ``napari_storm.napari_particles.<module>``
for the modules moved here, are these same module objects.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__ = _distribution_version("smlm-view3d")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
