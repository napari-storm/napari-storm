from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    # Single source of truth: the version declared in setup.cfg and recorded in
    # the installed distribution metadata.  The previous ._version import came
    # from setuptools_scm, which was listed as a build requirement but never
    # configured, so __version__ was permanently "unknown".
    __version__ = _distribution_version("napari-storm")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "napari_storm"]


def __getattr__(name):
    """Load the Qt dock only when the legacy top-level export is requested.

    The base distribution deliberately does not force a Qt binding.  Eagerly
    importing the dock here made even ``import napari_storm`` and version
    inspection fail in an otherwise valid headless installation.  npe2 imports
    the dock's module directly when the widget is actually requested.
    """
    if name == "napari_storm":
        from ._dock_widget import napari_storm as dock_widget

        return dock_widget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
