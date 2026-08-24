"""Reader hook for the localization formats handled by napari-storm."""
import napari

from ._dock_widget import napari_storm
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

    if path.split(".")[-1] in list_of_recognized_file_formats:
        return reader_function
    else:
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
