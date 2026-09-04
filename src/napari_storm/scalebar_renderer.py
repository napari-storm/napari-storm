import napari
import numpy as np


class ScalebarRenderer:
    """Creates and manages the 3D/2D scalebar surface layer in the napari viewer."""

    def __init__(self, viewer, render_config):
        self.viewer = viewer
        self.render_config = render_config
        self.scalebar_layer = None
        self.scalebar_exists = False

    def update(self, dataset):
        """Create, update, or remove the scalebar surface layer.

        Parameters
        ----------
        dataset : LocalizationDataBaseClass
            The most recently loaded localization dataset.
        """
        v = napari.current_viewer()
        cpos = v.camera.center
        length_nm = int(self.render_config.scalebar_size_nm)

        # Use the viewer's actual layer list as the source of truth rather than
        # self.scalebar_exists, which can fall out of sync (e.g. after a widget
        # rebuild or an exception in a previous call).
        scalebar_in_viewer = "scalebar" in v.layers

        if self.render_config.scalebar_enabled and dataset.number_of_active_entries():
            if dataset.zdim_present:
                list = [length_nm, 0.125 * length_nm / 2, 0.125 * length_nm / 2]
                faces = np.asarray(
                    [
                        [0, 1, 2],
                        [1, 2, 3],
                        [4, 5, 6],
                        [5, 6, 7],
                        [0, 2, 4],
                        [4, 6, 2],
                        [1, 3, 7],
                        [1, 5, 7],
                        [2, 3, 6],
                        [3, 6, 7],
                        [4, 5, 0],
                        [0, 1, 5],
                    ]
                )

                vertices = np.asarray(
                    [
                        [-list[1], list[1], list[2]],
                        [-list[1], -list[1], list[2]],
                        [-list[1], list[1], -list[2]],
                        [-list[1], -list[1], -list[2]],
                        [length_nm - list[1], list[1], list[2]],
                        [length_nm - list[1], -list[1], list[2]],
                        [length_nm - list[1], list[1], -list[2]],
                        [length_nm - list[1], -list[1], -list[2]],
                        [list[1], -list[1], list[2]],
                        [-list[1], -list[1], list[2]],
                        [list[1], -list[1], -list[2]],
                        [-list[1], -list[1], -list[2]],
                        [list[1], length_nm - list[1], list[2]],
                        [-list[1], length_nm - list[1], list[2]],
                        [list[1], length_nm - list[1], -list[2]],
                        [-list[1], length_nm - list[1], -list[2]],
                        [list[1], list[2], -list[1]],
                        [-list[1], list[2], -list[1]],
                        [list[1], -list[2], -list[1]],
                        [-list[1], -list[2], -list[1]],
                        [list[1], list[2], length_nm - list[1]],
                        [-list[1], list[2], length_nm - list[1]],
                        [list[1], -list[2], length_nm - list[1]],
                        [-list[1], -list[2], length_nm - list[1]],
                    ]
                )
                for i in range(len(vertices)):
                    vertices[i] = vertices[i] + cpos - (length_nm / 2, 0, 0)

                faces = np.asarray(np.vstack((faces, faces + 8, faces + 16)))
            else:
                list = [length_nm, 0.05 * length_nm]

                faces = np.asarray([[0, 1, 3], [1, 2, 3], [4, 5, 7], [5, 6, 7]])
                verts = [
                    [cpos[1], cpos[2] - list[1]],
                    [cpos[1] + list[0], cpos[2] - list[1]],
                    [cpos[1] + list[0], cpos[2] + list[1]],
                    [cpos[1], cpos[2] + list[1]],
                    [cpos[1] - list[1], cpos[2]],
                    [cpos[1] + list[1], cpos[2]],
                    [cpos[1] + list[1], cpos[2] + list[0]],
                    [cpos[1] - list[1], cpos[2] + list[0]],
                ]
                vertices = np.reshape(np.asarray(verts), (8, 2))

            if scalebar_in_viewer:
                # Update geometry in-place — avoids the remove/re-add cycle
                # that caused AssertionErrors when scalebar_exists was stale.
                v.layers["scalebar"].data = (vertices, faces)
                self.scalebar_layer = v.layers["scalebar"]
            else:
                self.scalebar_layer = v.add_surface(
                    (vertices, faces), name="scalebar", shading="none"
                )
            self.scalebar_exists = True
        else:
            if scalebar_in_viewer:
                v.layers.remove("scalebar")
            self.scalebar_layer = None
            self.scalebar_exists = False
