"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/dev/plugins/for_plugin_developers.html
"""
import napari
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from ._dock_widget import napari_storm
from .CustomErrors import *
from .ns_constants import *


@napari_hook_implementation
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
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]


    if path.split('.')[-1] in list_of_recognized_file_formats:
        return reader_function
    else:
        return None


def reader_function(path):
    path = path.replace('\\', '/')
    import gc
    instance_counter = 0
    for obj in gc.get_objects():
        if isinstance(obj, napari_storm):
            our_dock_widget = obj
            instance_counter += 1
    if instance_counter > 1:
        raise MoreThanOneInstanceError('When importing a file, there should'
                                       ' be only one instance of the napari dock widget')
    elif instance_counter == 0:
        v = napari.current_viewer()
        our_dock_widget = napari_storm(napari_viewer=v)
        v.window.qt_viewer.dockLayerControls.setVisible(False)
        v.window.add_dock_widget(our_dock_widget, area="right", name="napari-STORM")
    else:
        v = napari.current_viewer()
        v.window.qt_viewer.dockLayerControls.setVisible(False)
    napari_storm.open_localization_data_file_and_get_dataset(our_dock_widget, file_path=path)
    return [(None,)]

