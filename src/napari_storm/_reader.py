"""Reader hook for the localization formats handled by napari-storm."""

from pathlib import Path

import napari

from ._dock_widget import napari_storm
from .localization_dataset_types.minflux_v2 import zarr_store_root
from .ns_constants import list_of_recognized_file_formats


def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        path = path[0]

    # A MINFLUX Zarr dataset is a directory, and so has no suffix to dispatch
    # on.  The manifest accepts directories for its sake, so this has to be
    # the thing that declines every *other* directory dropped on the canvas.
    if zarr_store_root(path) is not None:
        return reader_function
    if Path(path).is_dir():
        return None

    # Suffix, not "text after the last dot": a path whose directory contains a
    # dot and whose file does not used to take the directory's.  Lowercased,
    # because a file dialog on a case-insensitive filesystem hands back
    # whatever the file was named -- .CSV is a csv.
    if Path(path).suffix.lower().lstrip(".") in list_of_recognized_file_formats:
        return reader_function
    return None


def reader_function(path):
    path = path.replace("\\", "/")
    v = napari.current_viewer()
    our_dock_widget = napari_storm.get_instance(v)
    if our_dock_widget is None:
        our_dock_widget = napari_storm(napari_viewer=v)
        v.window.add_dock_widget(our_dock_widget, area="right", name="napari-STORM")
    our_dock_widget.open_localization_data_file_and_get_dataset(file_path=path)
    return [(None,)]
