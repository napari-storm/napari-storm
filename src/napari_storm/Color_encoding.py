import napari
import numpy as np


class Colorbar:
    def __init__(self, parent, width, height):
        self.parent = parent
        self.width = width
        self.height = height
        self.offset = 30
        self.color_scale_surface_layer = None
        self.low_number = None
        self.high_number = None
        self.z_min = self.parent.data_to_layer_itf.render_range_z[1] * \
                     self.parent.render_range_slider_z_percent[0] / 1E5
        self.z_max = self.parent.data_to_layer_itf.render_range_z[1] * \
                     self.parent.render_range_slider_z_percent[1] / 1E5
        self.verts = np.array([[0, 0],
                               [0, self.height],
                               [self.width, 0],
                               [self.width, self.height],
                               [2 * self.width, 0],
                               [2 * self.width, self.height],
                               [3 * self.width, 0],
                               [3 * self.width, self.height]])

        self.faces = np.array([[0, 1, 2],
                               [1, 2, 3],
                               [2, 3, 4],
                               [3, 4, 5],
                               [4, 5, 6],
                               [5, 6, 7]
                               ])

        self.values = np.array([0, 0, .2, .2, .4, .4, .6, .6])

        surface = (self.verts, self.faces, self.values)
        self.color_scale_surface_layer = napari.current_viewer().add_surface(surface,
                                                                             colormap="hsv_r", shading='none',
                                                                             name="colormap")
        self.color_scale_surface_layer.visible = False
        self.low_number = Number(center_coord=[3 * self.width, self.height + self.offset],
                                 length=self.width / 10,
                                 width=self.height / 20,
                                 number=self.z_min)
        self.high_number = Number(center_coord=[0, self.height + self.offset],
                                  length=self.width / 10,
                                  width=self.height / 20,
                                  number=self.z_max)

    def update_numbers(self):
        self.z_min = np.round(self.parent.data_to_layer_itf.render_range_z[1] * \
                     self.parent.render_range_slider_z_percent[0] / 1E5, 2)
        self.z_max = np.round(self.parent.data_to_layer_itf.render_range_z[1] * \
                     self.parent.render_range_slider_z_percent[1] / 1E5, 2)
        for layer in napari.current_viewer().layers:
            if "tile" in layer.name:
                napari.current_viewer().layers.remove(layer.name)
        self.low_number = Number(center_coord=[3 * self.width, self.height + self.offset],
                                 length=self.width / 10,
                                 width=self.height / 20,
                                 number=self.z_min)
        self.high_number = Number(center_coord=[0, self.height + self.offset],
                                  length=self.width / 10,
                                  width=self.height / 20,
                                  number=self.z_max)
        if self.color_scale_surface_layer.visible:
            self.show()

    def show(self):
        self.color_scale_surface_layer.visible = True
        self.low_number.show()
        self.high_number.show()

    def hide(self):
        self.color_scale_surface_layer.visible = False
        self.low_number.hide()
        self.high_number.hide()


class Number:
    def __init__(self, center_coord, length, width, number):
        self.orientation = 0  # XY
        self.center_coord = center_coord
        self.length = length
        self.width = width
        self.number = np.round(number, 2)
        num_dec_digits = len(str(self.number).split('.')[-1])
        num_whole_digits = len(str(self.number).split('.')[0])

        self.dot_vertices = np.asarray([[-.5 * self.width, .5 * self.width],
                                        [.5 * self.width, .5 * self.width],
                                        [-.5 * self.width, -.5 * self.width],
                                        [.5 * self.width, -.5 * self.width],
                                        ]) + self.center_coord - np.asarray([0, self.length])
        self.dot_faces = np.asarray([[0, 1, 2], [1, 2, 3]])
        self.dot_values = np.asarray([1, 1, 1, 1])
        self.dot_surfaces = (self.dot_vertices, self.dot_faces, self.dot_values)
        self.dot = napari.current_viewer().add_surface(self.dot_surfaces, shading='none', name='dot')
        self.dot.visible = False
        self.dec_digits = []
        self.whole_digits = []
        for i in range(num_dec_digits):
            tmp_number = np.mod(int(self.number * (10 ** (i + 1))), 10)
            tmp_center_coord = np.asarray(
                [-(.5 * self.width + (i + 1) * (self.length + 1.2 * self.width) + i * self.length / 2),
                 0]) + self.center_coord
            self.dec_digits.append(Digit(center_coord=tmp_center_coord,
                                         length=self.length,
                                         width=self.width,
                                         number=tmp_number))
        for i in range(num_whole_digits):
            tmp_number = np.mod(int(self.number / (10 ** i)), 10)
            tmp_center_coord = np.asarray(
                [(.5 * self.width + (i + 1) * (self.length + 1.2 * self.width) + i * self.length / 2),
                 0]) + self.center_coord
            self.whole_digits.append(Digit(center_coord=tmp_center_coord,
                                           length=self.length,
                                           width=self.width,
                                           number=tmp_number))

    def show(self):
        for digit in self.dec_digits:
            digit.show_active_tiles()
        for digit in self.whole_digits:
            digit.show_active_tiles()
        self.dot.visible = True

    def hide(self):
        for digit in self.dec_digits:
            digit.hide_all_tiles()
        for digit in self.whole_digits:
            digit.hide_all_tiles()
        self.dot.visible = False


class Digit:
    def __init__(self, center_coord, length, width, number):
        self.number = number

        self.center_coord = center_coord
        self.length = length
        self.width = width
        self.tile_coords = np.asarray([[0, self.length],
                                       [-.5 * self.length, .5 * self.length],
                                       [.5 * self.length, .5 * self.length],
                                       [0, 0],
                                       [-.5 * self.length, -.5 * self.length],
                                       [.5 * self.length, -.5 * self.length],
                                       [0, -self.length]]) + self.center_coord
        self.digit_tiles = []
        self.tile_orientations = ["h", "v", "v", "h", "v", "v", "h"]
        for i in range(7):
            self.digit_tiles.append(DigitTile(center_coord=self.tile_coords[i],
                                              orientation=self.tile_orientations[i],
                                              length=self.length,
                                              width=self.width))
        self.active_tiles = []
        self.number_to_active_tiles()

    def number_to_active_tiles(self):
        if self.number == 0:
            self.active_tiles = np.asarray([0, 1, 2, 4, 5, 6])
        elif self.number == 1:
            self.active_tiles = np.asarray([1, 4])
        elif self.number == 2:
            self.active_tiles = np.asarray([0, 1, 3, 5, 6])
        elif self.number == 3:
            self.active_tiles = np.asarray([0, 1, 3, 4, 6])
        elif self.number == 4:
            self.active_tiles = np.asarray([1, 2, 3, 4])
        elif self.number == 5:
            self.active_tiles = np.asarray([0, 2, 3, 4, 6])
        elif self.number == 6:
            self.active_tiles = np.asarray([0, 2, 3, 4, 5, 6])
        elif self.number == 7:
            self.active_tiles = np.asarray([0, 1, 5])
        elif self.number == 8:
            self.active_tiles = np.asarray([0, 1, 2, 3, 4, 5, 6])
        elif self.number == 9:
            self.active_tiles = np.asarray([0, 1, 2, 3, 4, 6])

    def show_active_tiles(self):
        for i in range(7):
            if i in self.active_tiles:
                self.digit_tiles[i].show_tile()
            else:
                self.digit_tiles[i].hide_tile()

    def hide_all_tiles(self):
        for i in range(7):
            self.digit_tiles[i].hide_tile()


class DigitTile:
    def __init__(self, center_coord, orientation, length, width):
        self.center_coord = center_coord
        self.orientation = orientation
        self.length = length
        self.width = width

        self.vertices = np.asarray([[-.5 * (self.length - self.width), .5 * self.width],
                                    [.5 * (self.length - self.width), .5 * self.width],
                                    [-.5 * self.length, 0],
                                    [.5 * self.length, 0],
                                    [-.5 * (self.length - self.width), -.5 * self.width],
                                    [.5 * (self.length - self.width), -.5 * self.width]])
        self.faces = np.asarray([[0, 1, 4],
                                 [0, 2, 4],
                                 [1, 4, 5],
                                 [1, 3, 5]])

        if orientation == 'v':
            tmp = self.vertices[:, 0].copy()
            self.vertices[:, 0] = self.vertices[:, 1]
            self.vertices[:, 1] = tmp
        self.vertices += self.center_coord

        self.tile_surfaces = []
        self.tiles = []

        self.values = np.asarray([1, 1, 1, 1, 1, 1])

        self.change_tile_properties()
        self.add_tile_to_layer()
        self.hide_tile()

    def change_tile_properties(self):
        self.tile_surfaces = (self.vertices, self.faces, self.values)

    def add_tile_to_layer(self):
        self.tiles = napari.current_viewer().add_surface(self.tile_surfaces, shading='none', name='tile')

    def hide_tile(self):
        self.tiles.visible = False

    def show_tile(self):
        self.tiles.visible = True


if __name__ == "__main__":
    napari.Viewer()
    center_coord = np.asarray((0, 0))
    length = 25
    width = 8
    number = 18257.24234
    Number(center_coord=center_coord,
           length=length,
           width=width,
           number=number)
    napari.run()
