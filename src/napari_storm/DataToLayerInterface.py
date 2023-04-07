import napari
from PIL import ImageQt, Image
from PyQt5.QtGui import QIcon, QPixmap

from .napari_particles.particles import Particles
import numpy as np
from .napari_particles.utils import generate_billboards_2d
from .CustomErrors import *


class DataToLayerInterface:  # localization always with z # switch info with channel controls #
    def __init__(self, parent, viewer, surface_layer=None, grid_plane_layer=None):

        # assert isinstance(parent, napari_storm) == True
        self._parent = parent
        self.viewer = viewer
        self.colormap, self.colormap_icons = self.colormaps()
        self.scalebar_layer = surface_layer
        self.scalebar_exists = True
        if not self.scalebar_layer:
            self.scalebar_exists = False
        self.grid_plane_layer = grid_plane_layer
        self.default_line_thickness_nm = None
        self.grid_plane_layer_opacity = .75
        self.current_grid_plane_color = "white"
        self.current_grid_plane_z_pos = None

        # Render properties for every dataset, stored in lists
        self.render_sigma = []
        self.render_size = []
        self.render_values = []
        # self.render_colormap = []
        self.render_anti_alias = 0

        self.render_range_x = [np.inf, -np.inf]
        self.render_range_y = [np.inf, -np.inf]
        self.render_range_z = [np.inf, -np.inf]
        self.offset_nm_3d = [0, 0, 0]
        self.offset_nm_2d = [0, 0]

        self.camera = [self.viewer.camera.zoom, self.viewer.camera.center, self.viewer.camera.angles]

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    @property
    def list_of_datasets(self):
        return self.parent.list_of_datasets

    @list_of_datasets.setter
    def list_of_datasets(self, value):
        raise ParentError('Cannot change parent\'s attribute from here')

    def create_remove_grid_plane_state(self, enable):
        if enable:
            if not self.current_grid_plane_z_pos:
                if self.parent.zdim:
                    self.current_grid_plane_z_pos = np.mean(self.render_range_z)
                else:
                    self.current_grid_plane_z_pos = 1
            default_line_dist_nm = self.parent.grid_plane_line_distance_um * 1000
            num_of_lines_x = int(np.floor((self.render_range_y[1] - self.render_range_y[0]) *
                                          (self.parent.render_range_slider_y_percent[1] -
                                           self.parent.render_range_slider_y_percent[0]) / 100 / default_line_dist_nm))
            num_of_lines_y = int(np.floor((self.render_range_x[1] - self.render_range_x[0]) *
                                          (self.parent.render_range_slider_x_percent[1] -
                                           self.parent.render_range_slider_x_percent[0]) / 100 / default_line_dist_nm))
            if not self.default_line_thickness_nm:
                self.default_line_thickness_nm = 0.05 / np.mean((num_of_lines_x, num_of_lines_y)) * np.mean((
                    self.render_range_y[1] - self.render_range_y[0],
                    self.render_range_x[1] - self.render_range_x[0]))

            vectors_x = np.zeros((num_of_lines_x + 1, 2, 3))
            # length of vectors
            vectors_x[:, 1, 1] = (self.render_range_x[1] * 1.01 - self.render_range_x[0]) * \
                                 (self.parent.render_range_slider_x_percent[1] -
                                  self.parent.render_range_slider_x_percent[0]) / 100
            # x and y start position of vectors
            vectors_x[:, 0, 2] = np.arange(num_of_lines_x + 1) * default_line_dist_nm + self.render_range_x[0] + \
                                 (self.render_range_y[1] - self.render_range_y[0]) \
                                 * (self.parent.render_range_slider_y_percent[0]) / 100
            vectors_x[:, 0, 1] = np.ones(num_of_lines_x + 1) * (self.render_range_x[0] +
                                                                (self.render_range_x[1] - self.render_range_x[0])
                                                                * (self.parent.render_range_slider_x_percent[
                        0]) / 100)
            vectors_x[:, 0, 0] = self.current_grid_plane_z_pos

            vectors_y = np.zeros((num_of_lines_y + 1, 2, 3))
            vectors_y[:, 1, 2] = (self.render_range_y[1] * 1.01 - self.render_range_y[0]) * \
                                 (self.parent.render_range_slider_y_percent[1] -
                                  self.parent.render_range_slider_y_percent[0]) / 100
            vectors_y[:, 0, 1] = np.arange(num_of_lines_y + 1) * default_line_dist_nm + self.render_range_x[0] + \
                                 (self.render_range_x[1] - self.render_range_x[0]) \
                                 * (self.parent.render_range_slider_x_percent[0]) / 100
            vectors_y[:, 0, 2] = np.ones(num_of_lines_y + 1) * (self.render_range_y[0] +
                                                                (self.render_range_y[1] - self.render_range_y[0])
                                                                * (self.parent.render_range_slider_y_percent[
                        0]) / 100)
            vectors_y[:, 0, 0] = self.current_grid_plane_z_pos

            vectors = np.concatenate((vectors_x, vectors_y))
            self.grid_plane_layer = self.viewer.add_vectors(vectors, edge_width=self.default_line_thickness_nm,
                                                            name="Grid_Plane", edge_color=self.current_grid_plane_color,
                                                            ndim=3,
                                                            opacity=self.grid_plane_layer_opacity)
        else:
            self.default_line_thickness_nm = self.grid_plane_layer.edge_width
            self.grid_plane_layer_opacity = self.grid_plane_layer.opacity
            self.viewer.layers.remove('Grid_Plane')

    def update_grid_plane(self, z_pos=None, line_thickness=None, line_distance_nm=None, color=None, opacity=None):
        if line_distance_nm:
            self.grid_plane_layer_opacity = self.grid_plane_layer.opacity
            default_line_thickness_nm = self.grid_plane_layer.edge_width
            default_line_dist_nm = self.parent.grid_plane_line_distance_um * 1000
            num_of_lines_x = int(np.floor((self.render_range_y[1] - self.render_range_y[0]) *
                                          (self.parent.render_range_slider_y_percent[1] -
                                           self.parent.render_range_slider_y_percent[0]) / 100 / default_line_dist_nm))
            num_of_lines_y = int(np.floor((self.render_range_x[1] - self.render_range_x[0]) *
                                          (self.parent.render_range_slider_x_percent[1] -
                                           self.parent.render_range_slider_x_percent[0]) / 100 / default_line_dist_nm))
            if num_of_lines_x < 1 or num_of_lines_y < 1:
                tmp_max_line_dist = np.round(np.floor(min((self.render_range_y[1] - self.render_range_y[0]) *
                                                          (self.parent.render_range_slider_x_percent[1] -
                                                           self.parent.render_range_slider_x_percent[0]) / 100,
                                                          (self.render_range_x[1] - self.render_range_x[0]) *
                                                          (self.parent.render_range_slider_y_percent[1] -
                                                           self.parent.render_range_slider_y_percent[0]) / 100)) * .001,
                                             3)
                self.parent.Egrid_line_distance.setText(str(tmp_max_line_dist))
                return
            self.viewer.layers.remove('Grid_Plane')
            vectors_x = np.zeros((num_of_lines_x + 1, 2, 3))
            # length of vectors
            vectors_x[:, 1, 1] = (self.render_range_x[1] * 1.01 - self.render_range_x[0]) * \
                                 (self.parent.render_range_slider_x_percent[1] -
                                  self.parent.render_range_slider_x_percent[0]) / 100
            # x and y start position of vectors
            vectors_x[:, 0, 2] = np.arange(num_of_lines_x + 1) * default_line_dist_nm + self.render_range_x[0] + \
                                 (self.render_range_y[1] - self.render_range_y[0]) \
                                 * (self.parent.render_range_slider_y_percent[0]) / 100
            vectors_x[:, 0, 1] = np.ones(num_of_lines_x + 1) * (self.render_range_x[0] +
                                                                (self.render_range_x[1] - self.render_range_x[0])
                                                                * (self.parent.render_range_slider_x_percent[
                        0]) / 100)
            vectors_x[:, 0, 0] = self.current_grid_plane_z_pos

            vectors_y = np.zeros((num_of_lines_y + 1, 2, 3))
            vectors_y[:, 1, 2] = (self.render_range_y[1] * 1.01 - self.render_range_y[0]) * \
                                 (self.parent.render_range_slider_y_percent[1] -
                                  self.parent.render_range_slider_y_percent[0]) / 100
            vectors_y[:, 0, 1] = np.arange(num_of_lines_y + 1) * default_line_dist_nm + self.render_range_x[0] + \
                                 (self.render_range_x[1] - self.render_range_x[0]) \
                                 * (self.parent.render_range_slider_x_percent[0]) / 100
            vectors_y[:, 0, 2] = np.ones(num_of_lines_y + 1) * (self.render_range_y[0] +
                                                                (self.render_range_y[1] - self.render_range_y[0])
                                                                * (self.parent.render_range_slider_y_percent[
                        0]) / 100)
            vectors_y[:, 0, 0] = self.current_grid_plane_z_pos
            vectors = np.concatenate((vectors_x, vectors_y))
            self.grid_plane_layer = self.viewer.add_vectors(vectors, edge_width=default_line_thickness_nm,
                                                            name="Grid_Plane", edge_color=self.current_grid_plane_color,
                                                            ndim=3,
                                                            opacity=self.grid_plane_layer_opacity)
            self.parent.update_grid_plane_color()

        if z_pos:
            vectors = self.grid_plane_layer.data
            if self.parent.zdim:
                self.current_grid_plane_z_pos = z_pos / 100 * (self.render_range_z[1] - self.render_range_z[0])
                vectors[:, 0, 0] = self.current_grid_plane_z_pos
            else:
                vectors[:, 0, 0] = 1
            self.grid_plane_layer.data = vectors
        if line_thickness:
            self.grid_plane_layer.edge_width = 0.05 * np.exp(line_thickness / 10 - 5) / len(
                self.grid_plane_layer.data[:, 0, 0]) / 2 \
                                               * np.mean((self.render_range_x[1] - self.render_range_x[0],
                                                          self.render_range_y[1] - self.render_range_y[0]))
        if color:
            self.current_grid_plane_color = color
            self.grid_plane_layer.edge_color = color
        if opacity:
            self.grid_plane_layer.opacity = opacity / 100

    def reset_render_range_and_offset(self):
        self.render_range_x = [np.inf, -np.inf]
        self.render_range_y = [np.inf, -np.inf]
        self.render_range_z = [np.inf, -np.inf]
        self.offset_nm_3d = [0, 0, 0]
        self.offset_nm_2d = [0, 0]

    def set_render_range_and_offset(self):
        self.reset_render_range_and_offset()
        for dataset in self.parent.list_of_datasets:
            self.set_offset(dataset=dataset)
            coords = self.get_coords_from_all_locs(dataset=dataset)
            self.set_render_range(zdim=dataset.zdim_present, coords=coords)
        self.parent.move_camera_center_to_render_range_center()

    def set_render_range(self, zdim, coords):
        if zdim:
            self.render_range_x[1] = max(np.max(coords[:, 1]), self.render_range_x[1])
            self.render_range_y[1] = max(np.max(coords[:, 2]), self.render_range_y[1])
            self.render_range_z[1] = max(np.max(coords[:, 0]), self.render_range_z[1])
            self.render_range_x[0] = min(np.min(coords[:, 1]), self.render_range_x[0])
            self.render_range_y[0] = min(np.min(coords[:, 2]), self.render_range_y[0])
            self.render_range_z[0] = min(np.min(coords[:, 0]), self.render_range_z[0])
        else:
            self.render_range_x[1] = max(np.max(coords[:, 2]), self.render_range_x[1])
            self.render_range_y[1] = max(np.max(coords[:, 1]), self.render_range_y[1])
            self.render_range_x[0] = min(np.min(coords[:, 2]), self.render_range_x[0])
            self.render_range_y[0] = min(np.min(coords[:, 1]), self.render_range_y[0])

    def set_offset(self, dataset):
        if dataset.zdim_present:
            if self.offset_nm_3d == [0, 0, 0]:
                self.offset_nm_3d = [-np.min(dataset.z_pos_nm),
                                     -np.min(dataset.x_pos_nm),
                                     -np.min(dataset.y_pos_nm)]
            else:
                self.offset_nm_3d[2] = np.max([self.offset_nm_3d[2],
                                               -np.min(dataset.y_pos_nm)])
                self.offset_nm_3d[1] = np.max([self.offset_nm_3d[1],
                                               -np.min(dataset.x_pos_nm)])
                self.offset_nm_3d[0] = np.max([self.offset_nm_3d[0],
                                               -np.min(dataset.z_pos_nm)])

        else:
            if self.offset_nm_2d == [0, 0]:
                self.offset_nm_2d = [-np.min(dataset.x_pos_nm),
                                     -np.min(dataset.y_pos_nm)]
            else:
                self.offset_nm_2d[0] = np.max([self.offset_nm_2d[0],
                                               -np.min(dataset.x_pos_nm)])
                self.offset_nm_2d[1] = np.max([self.offset_nm_2d[1],
                                               -np.min(dataset.y_pos_nm)])

    def create_new_layer(self, dataset, merge=False, layer_name='SMLM Data', idx=-1):
        """Creating a Particle Layer"""
        self.set_offset(dataset)
        coords = self.get_coords_from_locs(dataset=dataset)
        self.set_render_range(coords=coords, zdim=dataset.zdim_present)
        if merge:
            self.set_render_range_and_offset()
            dataset.restrict_locs_by_percent(self.parent.render_range_slider_x_percent,
                                             self.parent.render_range_slider_y_percent,
                                             self.parent.render_range_slider_z_percent)
            coords = self.get_coords_from_locs(dataset=dataset)
        self.set_render_sigmas(dataset=dataset, create=True)
        self.set_render_values(dataset=dataset, create=True)
        dataset.napari_layer_ref = Particles(
            coords,
            size=self.render_size[-1],
            values=self.render_values[-1],
            antialias=self.render_anti_alias,
            colormap=self.colormap[-1],
            sigmas=self.render_sigma[-1],
            filter=None,
            name=layer_name,
        )
        dataset.name = layer_name
        dataset.napari_layer_ref.add_to_viewer(self.viewer)
        self.viewer.camera.perspective = 50
        dataset.napari_layer_ref.shading = 'gaussian'
        self.viewer.camera.angles = (90, 0, -90)
        self.camera = [self.viewer.camera.zoom, self.viewer.camera.center, self.viewer.camera.angles]

    def update_layer(self, dataset, dataset_idx):
        """Updating a Particle Layer"""
        v = self.viewer
        if not dataset.x_pos_nm.size == 0:
            v.layers.remove(dataset.name)
            coords = self.get_coords_from_locs(dataset)
            self.set_render_range(dataset.zdim_present, coords)
            self.set_render_sigmas(dataset=dataset, channel_index=dataset_idx)
            self.set_render_values(dataset=dataset, channel_index=dataset_idx)
            dataset.napari_layer_ref = Particles(
                coords,
                size=self.render_size[dataset_idx],
                values=self.render_values[dataset_idx],
                antialias=self.render_anti_alias,
                colormap=self.colormap[dataset_idx],
                sigmas=self.render_sigma[dataset_idx],
                filter=None,
                name=dataset.name,
                visible=True
            )
            dataset.napari_layer_ref.add_to_viewer(v)
            self.parent.channel[dataset_idx].adjust_colormap_range()
            self.parent.channel[dataset_idx].adjust_z_color_encoding_opacity()
            self.parent.channel[dataset_idx].change_color_map()
            dataset.napari_layer_ref.shading = "gaussian"
        else:
            dataset.napari_layer_ref.visible = False

    def update_data_range(self, dataset, dataset_idx=-1):
        if dataset.zdim_present:
            x_range = np.asarray(self.parent.render_range_slider_x_percent) / 100 * np.ones(2) * (
                self.render_range_x[1]) - self.offset_nm_3d[1]
            y_range = np.asarray(self.parent.render_range_slider_y_percent) / 100 * np.ones(2) * (
                self.render_range_y[1]) - self.offset_nm_3d[2]
            z_range = np.asarray(self.parent.render_range_slider_z_percent) / 100 * np.ones(2) * (
                self.render_range_z[1]) - self.offset_nm_3d[0]
            x_indices = dataset.get_idx_of_specified_prop_all(prop="x_pos_nm", l_val=x_range[0], u_val=x_range[1])
            y_indices = dataset.get_idx_of_specified_prop_all(prop="y_pos_nm", l_val=y_range[0], u_val=y_range[1])
            z_indices = dataset.get_idx_of_specified_prop_all(prop="z_pos_nm", l_val=z_range[0], u_val=z_range[1])
            render_indices = np.intersect1d(x_indices, y_indices)
            render_indices = np.intersect1d(render_indices, z_indices)
        else:
            x_range = np.asarray(self.parent.render_range_slider_x_percent) / 100 * np.ones(2) * (
                self.render_range_x[1]) - self.offset_nm_2d[0]
            y_range = np.asarray(self.parent.render_range_slider_y_percent) / 100 * np.ones(2) * (
                self.render_range_y[1]) - self.offset_nm_2d[1]
            x_indices = dataset.get_idx_of_specified_prop_all(prop="x_pos_nm", l_val=x_range[0], u_val=x_range[1])
            y_indices = dataset.get_idx_of_specified_prop_all(prop="y_pos_nm", l_val=y_range[0], u_val=y_range[1])
            render_indices = np.intersect1d(x_indices, y_indices)
        to_be_removed_by_index = np.delete(np.arange(len(dataset.x_pos_nm_all)), render_indices)
        if len(self.parent.data_filter_itf.filter_idx_list) > dataset_idx:
            to_be_removed_by_index = np.concatenate((to_be_removed_by_index,
                                                    self.parent.data_filter_itf.filter_idx_list[dataset_idx]),
                                                    dtype=int)
        if to_be_removed_by_index.size > 1:
            self.parent.dataset_itf.pipe_indices_to_be_removed(to_be_removed_by_index, dataset=dataset, reset=True)
        else:
            self.parent.dataset_itf.reset_filters()

    def update_layers2(self):
        """Still doesn't work"""
        v = napari.current_viewer()
        for i in range(len(self.list_of_datasets)):
            self.list_of_datasets[i].update_locs()
            coords = self.get_coords_from_locs(self.list_of_datasets[i].pixelsize_nm, i)
            values = self.list_of_datasets[i].values
            size = self.list_of_datasets[i].size

            coords = np.asarray(coords)
            if np.isscalar(values):
                values = values * np.ones(len(coords))
            values = np.broadcast_to(values, len(coords))
            size = np.broadcast_to(size, len(coords))
            if coords.shape[1] == 2:
                coords = np.concatenate([np.zeros((len(coords), 1)), coords], axis=-1)

            vertices, faces, texcoords = generate_billboards_2d(coords, size=size)

            values = np.repeat(values, 4, axis=0)
            vertices_old, faces_old, values_old = v.layers[
                self.list_of_datasets[i].name
            ].data
            # print(f"before: {len(vertices_old), len(faces_old), len(values_old)}")
            # print(f"after: {len(vertices), len(faces), len(values)}")
            v.layers[self.list_of_datasets[i].name].data = (vertices, faces, values)

    def save_current_camera(self):
        self.camera = [self.viewer.camera.zoom, self.viewer.camera.center, self.viewer.camera.angles]

    def restore_camera(self):
        self.viewer.camera.angles = self.camera[2]
        self.viewer.camera.zoom = self.camera[0]
        self.viewer.camera.center = self.camera[1]
        self.viewer.camera.update({})
    def colormaps(self):
        """Creating the Custom Colormaps"""
        cmaps = []
        cmap_icons = []
        names = ["red", "green", "blue"]
        for i in range(3):
            colors = np.zeros((2, 4))
            colors[-1][i] = 1
            colors[-1][-1] = 1
            cmaps.append(
                napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
            )
            icon = np.zeros((128, 128, 4))
            icon[:, :, -1] = 1
            icon[:, :, i] = np.interp((np.arange(128)+1)/128, [0, 1], [0, 1])
            icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon*255)))))
            cmap_icons.append(icon)
        names = ["yellow", "cyan", "pink"]
        for i in range(3):
            colors = np.zeros((2, 4))
            colors[-1][i] = 1
            colors[-1][(i + 1) % 3] = 1
            colors[-1][-1] = 1
            cmaps.append(
                napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
            )
            icon = np.zeros((128, 128, 4))
            icon[:, :, -1] = 1
            icon[:, :, i] = np.interp((np.arange(128)+1)/128, [0, 1], [0, 1])
            icon[:, :, (i + 1) % 3] = np.interp((np.arange(128)+1)/128, [0, 1], [0, 1])
            icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon*255)))))
            cmap_icons.append(icon)
        names = ["orange", "mint", "purple"]
        for i in range(3):
            colors = np.zeros((2, 4))
            colors[-1][i] = 1
            colors[-1][(i + 1) % 3] = 0.5
            colors[-1][-1] = 1
            cmaps.append(
                napari.utils.colormaps.colormap.Colormap(colors=colors, name=names[i])
            )
            icon = np.zeros((128, 128, 4))
            icon[:, :, -1] = 1
            icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
            icon[:, :, (i + 1) % 3] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
            icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon*255)))))
            cmap_icons.append(icon)
        colors = np.zeros((2, 4))
        colors[-1][:] = 1
        colors[:][-1] = 1
        cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name="gray"))
        icon = np.zeros((128, 128, 4))
        icon[:, :, -1] = 1
        for i in range(4):
            icon[:, :, i] = np.interp((np.arange(128) + 1) / 128, [0, 1], [0, 1])
        icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon*255)))))
        cmap_icons.append(icon)
        colors = np.zeros((4, 4))
        colors[1:, -1] = 1
        colors[1:, 0] = 1
        colors[2:, 1] = 1
        colors[3, 2] = 1
        cmaps.append(napari.utils.colormaps.colormap.Colormap(colors=colors, name="red hot"))
        icon = np.zeros((128, 128, 4))
        icon[:, :, -1] = 1
        icon[:, 16:48, 0] = np.ones((128, 32)) * np.arange(32) / 32
        icon[:, 48:, 0] = 1
        icon[:, 48:112, 1] = np.ones((128, 64)) * np.arange(64) / 64
        icon[:, 96:, 1] = 1
        icon[:, 96:, 2] = np.ones((128, 32)) * np.arange(32) / 32
        icon = QIcon(QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon*255)))))
        cmap_icons.append(icon)
        icon = np.zeros((128, 600, 4))
        icon[:, :100, 0] = 1
        icon[:, :100, 1] = np.ones((128, 100)) * np.arange(100)/100
        icon[:, 100:200, 0] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
        icon[:, 100:200, 1] = 1
        icon[:, 200:300, 2] = np.ones((128, 100)) * np.arange(100) / 100
        icon[:, 200:300, 1] = 1
        icon[:, 300:400, 2] = 1
        icon[:, 300:400, 1] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
        icon[:, 400:500, 0] = np.ones((128, 100)) * np.arange(100) / 100
        icon[:, 400:500, 2] = 1
        icon[:, 500:, 0] = 1
        icon[:, 500:, 2] = np.ones((128, 100)) * np.arange(100)[::-1] / 100
        icon[:, :, 3] = 1
        qpm = QPixmap.fromImage(ImageQt.ImageQt(Image.fromarray(np.uint8(icon * 255))))
        cmap_icons.append(qpm)
        return cmaps, cmap_icons

    def set_render_values(self, dataset, channel_index=-1, create=False):
        """Update values, which are used to determine the rendered
        color and intensity of each localization"""
        if self.parent.render_gaussian_mode == 0:
            # Fixed gaussian mode

            tmp_values = np.ones(dataset.locs_active.size)

        elif self.parent.render_gaussian_mode == 1:
            # Variable gaussian mode

            assert dataset.uncertainty_defined is True

            if dataset.zdim_present:
                # 3D data

                if dataset.sigma_present:
                    # Sigma values present

                    sigma_x_pixels = dataset.locs_active.sigma_x_pixels
                    sigma_y_pixels = dataset.locs_active.sigma_y_pixels
                    sigma_z_pixels = dataset.locs_active.sigma_z_pixels

                    tmp_product = sigma_x_pixels * sigma_y_pixels * sigma_z_pixels

                    tmp_values = 1.0 / tmp_product

                    if not np.max(tmp_values) == np.min(tmp_values):
                        tmp_values = (tmp_values - np.min(tmp_values)) / (np.max(tmp_values) - np.min(tmp_values))

                else:
                    # Calculate sigma according to photon count

                    psf_sigma_xy_nm = self.parent.render_var_gauss_PSF_sigma_xy_nm
                    psf_sigma_z_nm = self.parent.render_var_gauss_PSF_sigma_z_nm

                    psf_sigma_xy_pixels = psf_sigma_xy_nm / dataset.pixelsize_nm
                    psf_sigma_z_pixels = psf_sigma_z_nm / dataset.pixelsize_nm

                    sigma_xy_pixels = psf_sigma_xy_pixels / np.sqrt(dataset.locs_active.photon_count)
                    sigma_z_pixels = psf_sigma_z_pixels / np.sqrt(dataset.locs_active.photon_count)

                    tmp_product = sigma_xy_pixels ** 2 * sigma_z_pixels

                    tmp_values = 1.0 / tmp_product
                    if not np.max(tmp_values) == np.min(tmp_values):
                        tmp_values = (tmp_values - np.min(tmp_values)) / (np.max(tmp_values) - np.min(tmp_values))

            else:
                # 2D data

                if dataset.sigma_present:
                    # Sigma values present

                    sigma_x_pixels = dataset.locs_active.sigma_x_pixels
                    sigma_y_pixels = dataset.locs_active.sigma_y_pixels

                    tmp_product = sigma_x_pixels * sigma_y_pixels

                    tmp_values = 1.0 / tmp_product

                    if not np.max(tmp_values) == np.min(tmp_values):
                        tmp_values = (tmp_values - np.min(tmp_values)) / (np.max(tmp_values) - np.min(tmp_values))

                else:
                    # Calculate sigma according to photon count

                    psf_sigma_xy_nm = self.parent.render_var_gauss_PSF_sigma_xy_nm

                    psf_sigma_xy_pixels = psf_sigma_xy_nm / dataset.pixelsize_nm

                    sigma_xy_pixels = psf_sigma_xy_pixels / np.sqrt(dataset.locs_active.photon_count)

                    tmp_product = sigma_xy_pixels ** 2

                    tmp_values = 1.0 / tmp_product

                    if not np.max(tmp_values) == np.min(tmp_values):
                        tmp_values = (tmp_values - np.min(tmp_values)) / (np.max(tmp_values) - np.min(tmp_values))
            # if not all the same values map 99th percentile to 1
            tmp_values /= np.percentile(tmp_values, 99)

        if self.parent.z_color_encoding_mode == 1:
            # Color the localizations according to their Z-coordinate
            #
            # In this case, it is not possible to render unit volume gaussians
            # by adjusting their "intensity" via the value parameter.  Rather,
            # the value parameter is used in conjunction with the color map to
            # assign a z-dependent color to each localization.

            assert dataset.zdim_present is True

            tmp_coords = self.get_coords_from_locs(dataset)

            tmp_values = tmp_coords[:, 0]

            if not np.max(tmp_values) == np.min(tmp_values):
                tmp_values = (tmp_values - np.min(tmp_values)) / (np.max(tmp_values) - np.min(tmp_values))

        assert np.max(tmp_values) > 0

        # Store values
        if create:
            self.render_values.append(tmp_values)
        else:
            self.render_values[channel_index] = tmp_values

    def set_render_sigmas(self, dataset, channel_index=-1, create=False):
        """Update rendered sigma values"""
        if self.parent.render_gaussian_mode == 0:
            # Fixed gaussian mode

            sigma_xy_nm = self.parent.render_fixed_gauss_sigma_xy_nm
            sigma_z_nm = self.parent.render_fixed_gauss_sigma_z_nm

            tmp_sigma_xy = sigma_xy_nm * np.ones_like(dataset.x_pos_nm)
            tmp_sigma_z = sigma_z_nm * np.ones_like(dataset.x_pos_nm)

            tmp_render_sigma_nm = np.swapaxes([tmp_sigma_z, tmp_sigma_xy, tmp_sigma_xy], 0, 1)

        elif self.parent.render_gaussian_mode == 1:
            # Variable gaussian mode

            if dataset.sigma_present:
                # Sigma values present

                sigma_x_nm = dataset.locs_active.sigma_x_pixels * dataset.pixelsize_nm
                sigma_y_nm = dataset.locs_active.sigma_y_pixels * dataset.pixelsize_nm
                sigma_z_nm = dataset.locs_active.sigma_z_pixels * dataset.pixelsize_nm

                sigma_x_nm[sigma_x_nm < self.parent.render_var_gauss_sigma_min_xy_nm] = \
                    self.parent.render_var_gauss_sigma_min_xy_nm
                sigma_y_nm[sigma_y_nm < self.parent.render_var_gauss_sigma_min_xy_nm] = \
                    self.parent.render_var_gauss_sigma_min_xy_nm
                sigma_z_nm[sigma_z_nm < self.parent.render_var_gauss_sigma_min_z_nm] = \
                    self.parent.render_var_gauss_sigma_min_z_nm

                # leave out biggest 1 percent
                sigma_x_nm[sigma_x_nm > np.percentile(sigma_x_nm, 99)] = np.percentile(sigma_x_nm, 99)
                sigma_y_nm[sigma_y_nm > np.percentile(sigma_y_nm, 99)] = np.percentile(sigma_y_nm, 99)
                sigma_z_nm[sigma_z_nm > np.percentile(sigma_z_nm, 99)] = np.percentile(sigma_z_nm, 99)

                tmp_render_sigma_nm = np.swapaxes([sigma_z_nm, sigma_y_nm, sigma_x_nm], 0, 1)

            else:
                # Calculate sigma values based on photon counts

                psf_sigma_xy_nm = self.parent.render_var_gauss_PSF_sigma_xy_nm
                psf_sigma_z_nm = self.parent.render_var_gauss_PSF_sigma_z_nm

                sigma_xy_nm = psf_sigma_xy_nm / np.sqrt(dataset.locs_active.photon_count)
                sigma_z_nm = psf_sigma_z_nm / np.sqrt(dataset.locs_active.photon_count)

                sigma_xy_nm[sigma_xy_nm < self.parent.render_var_gauss_sigma_min_xy_nm] = \
                    self.parent.render_var_gauss_sigma_min_xy_nm
                sigma_z_nm[sigma_z_nm < self.parent.render_var_gauss_sigma_min_z_nm] = \
                    self.parent.render_var_gauss_sigma_min_z_nm

                # leave out biggest 1 percent
                sigma_xy_nm[sigma_xy_nm > np.percentile(sigma_xy_nm, 99)] = np.percentile(sigma_xy_nm, 99)
                sigma_z_nm[sigma_z_nm > np.percentile(sigma_z_nm, 99)] = np.percentile(sigma_z_nm, 99)

                tmp_render_sigma_nm = np.swapaxes([sigma_z_nm, sigma_xy_nm, sigma_xy_nm], 0, 1)
        else:
            raise RuntimeError('Render Gaussian Mode undefined')
        tmp_render_sigma_norm = tmp_render_sigma_nm / np.max(tmp_render_sigma_nm)

        # Store sigma values and set render size
        if create:
            self.render_sigma.append(tmp_render_sigma_norm)
            self.render_size.append(5 * np.max(tmp_render_sigma_nm))
        else:
            self.render_sigma[channel_index] = tmp_render_sigma_norm
            self.render_size[channel_index] = 5 * np.max(tmp_render_sigma_nm)

    def get_coords_from_locs(self, dataset):
        """Calculating Particle Coordinates from Locs"""
        if dataset.zdim_present:
            num_of_locs = len(dataset.x_pos_nm)
            coords = np.zeros([num_of_locs, 3])
            coords[:, 0] = dataset.z_pos_nm + self.offset_nm_3d[0]
            coords[:, 1] = dataset.x_pos_nm + self.offset_nm_3d[1]
            coords[:, 2] = dataset.y_pos_nm + self.offset_nm_3d[2]

        else:
            num_of_locs = len(dataset.x_pos_nm)
            coords = np.zeros([num_of_locs, 3])
            coords[:, 1] = dataset.x_pos_nm + self.offset_nm_2d[0]
            coords[:, 2] = dataset.y_pos_nm + self.offset_nm_2d[1]
            coords[:, 0] = np.ones(num_of_locs)
        return coords

    def get_coords_from_all_locs(self, dataset):
        """Calculating Particle Coordinates from Locs"""
        if dataset.zdim_present:
            num_of_locs = len(dataset.x_pos_nm_all)
            coords = np.zeros([num_of_locs, 3])
            coords[:, 0] = dataset.z_pos_nm_all + self.offset_nm_3d[0]
            coords[:, 1] = dataset.x_pos_nm_all + self.offset_nm_3d[1]
            coords[:, 2] = dataset.y_pos_nm_all + self.offset_nm_3d[2]

        else:
            num_of_locs = len(dataset.x_pos_nm)
            coords = np.zeros([num_of_locs, 3])
            coords[:, 1] = dataset.x_pos_nm_all + self.offset_nm_2d[0]
            coords[:, 2] = dataset.y_pos_nm_all + self.offset_nm_2d[1]
            coords[:, 0] = np.ones(num_of_locs)
        return coords

    def scalebar(self):
        """Creating/Removing/Updating the custom Scalebar in 2 and 3D"""
        v = napari.current_viewer()
        cpos = v.camera.center
        l = int(self.parent.Esbsize.text())
        if self.parent.Cscalebar.isChecked() and not not all(self.parent.list_of_datasets[-1].locs_active):
            if self.parent.list_of_datasets[-1].zdim_present:
                displacement_list = [l, 0.125 * l / 2, 0.125 * l / 2]
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
                        [-displacement_list[1], displacement_list[1], displacement_list[2]],
                        [-displacement_list[1], -displacement_list[1], displacement_list[2]],
                        [-displacement_list[1], displacement_list[1], -displacement_list[2]],
                        [-displacement_list[1], -displacement_list[1], -displacement_list[2]],
                        [l - displacement_list[1], displacement_list[1], displacement_list[2]],
                        [l - displacement_list[1], -displacement_list[1], displacement_list[2]],
                        [l - displacement_list[1], displacement_list[1], -displacement_list[2]],
                        [l - displacement_list[1], -displacement_list[1], -displacement_list[2]],
                        [displacement_list[1], -displacement_list[1], displacement_list[2]],
                        [-displacement_list[1], -displacement_list[1], displacement_list[2]],
                        [displacement_list[1], -displacement_list[1], -displacement_list[2]],
                        [-displacement_list[1], -displacement_list[1], -displacement_list[2]],
                        [displacement_list[1], l - displacement_list[1], displacement_list[2]],
                        [-displacement_list[1], l - displacement_list[1], displacement_list[2]],
                        [displacement_list[1], l - displacement_list[1], -displacement_list[2]],
                        [-displacement_list[1], l - displacement_list[1], -displacement_list[2]],
                        [displacement_list[1], displacement_list[2], -displacement_list[1]],
                        [-displacement_list[1], displacement_list[2], -displacement_list[1]],
                        [displacement_list[1], -displacement_list[2], -displacement_list[1]],
                        [-displacement_list[1], -displacement_list[2], -displacement_list[1]],
                        [displacement_list[1], displacement_list[2], l - displacement_list[1]],
                        [-displacement_list[1], displacement_list[2], l - displacement_list[1]],
                        [displacement_list[1], -displacement_list[2], l - displacement_list[1]],
                        [-displacement_list[1], -displacement_list[2], l - displacement_list[1]],
                    ]
                )
                for i in range(len(vertices)):
                    vertices[i] = vertices[i] + cpos - (l / 2, 0, 0)

                faces = np.asarray(np.vstack((faces, faces + 8, faces + 16)))
                # vertices=np.reshape(np.asarray(vertices),(24,3))
            else:
                displacement_list = [l, 0.05 * l]

                faces = np.asarray([[0, 1, 3], [1, 2, 3], [4, 5, 7], [5, 6, 7]])
                verts = [
                    [cpos[1], cpos[2] - displacement_list[1]],
                    [cpos[1] + displacement_list[0], cpos[2] - displacement_list[1]],
                    [cpos[1] + displacement_list[0], cpos[2] + displacement_list[1]],
                    [cpos[1], cpos[2] + displacement_list[1]],
                    [cpos[1] - displacement_list[1], cpos[2]],
                    [cpos[1] + displacement_list[1], cpos[2]],
                    [cpos[1] + displacement_list[1], cpos[2] + displacement_list[0]],
                    [cpos[1] - displacement_list[1], cpos[2] + displacement_list[0]],
                ]
                vertices = np.reshape(np.asarray(verts), (8, 2))
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_layer = v.add_surface(
                    (vertices, faces), name='scalebar', shading='none'
                )
            else:
                self.scalebar_layer = v.add_surface(
                    (vertices, faces), name='scalebar', shading='none'
                )
                self.scalebar_exists = True
        else:
            if self.scalebar_exists:
                v.layers.remove('scalebar')
                self.scalebar_exists = False



