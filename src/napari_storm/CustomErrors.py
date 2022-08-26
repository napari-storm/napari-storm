class DimensionError(ImportError):
    """Raise this error if something goes wrong with the data dimensions while importing"""


class ParentError(ValueError):
    """Raise when the parent of a widget gets changed"""


class StaticAttributeError(ValueError):
    """Like Parent Error, but mor general for all different kind of attributes"""


class MoreThanOneInstanceError(ImportError):
    """When importing a file, there should be only one instance of the napari dock widget"""


class PixelSizeIsNecessaryError(ValueError):
    """Raise this error when the pixelsize for a dataset is not provided"""


class FileImportAbortedError(ImportError):
    """Raise this error when the user aborts the import of a file"""

