import numpy as np
from .ns_constants import *
from .CustomErrors import *


class LocalizationData:
    """An Object which contains the localization data,
    as well as rendering parameters used by the napari
    particles layer"""

    def __init__(
            self,
            parent,
            locs,
            name=None,
            pixelsize_nm=None,
            zdim_present=False,
            sigma_present=False,
            photon_count_present=False,
    ):

        assert isinstance(locs, np.recarray)
        assert locs.dtype == LOCS_DTYPE

        if name is None:
            name = 'Untitled'

        if pixelsize_nm is None:
            pixelsize_nm = 100.0

        self._parent = parent
        self.napari_layer_ref = None

        self.name = name
        self.locs_active = locs
        self.locs_all = locs.copy()
        self.pixelsize_nm = pixelsize_nm
        self.zdim_present = zdim_present
        self.sigma_present = sigma_present
        self.photon_count_present = photon_count_present
        self.uncertainty_defined = sigma_present or photon_count_present

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    @property
    def locs(self):
        return self.locs_active

    def update_locs(self):

        self.locs_active = self.locs_all.copy()
        coords = self.get_all_coords_rec_array()

        xcoords = coords.x_pos_pixels - np.min(coords.x_pos_pixels)
        ycoords = coords.y_pos_pixels - np.min(coords.y_pos_pixels)
        zcoords = coords.z_pos_pixels - np.min(coords.z_pos_pixels)

        render_xrange = self.parent.render_range_x_percent
        render_yrange = self.parent.render_range_y_percent
        render_zrange = self.parent.render_range_z_percent

        xmin = render_xrange[0] / 100 * np.max(xcoords)
        xmax = render_xrange[1] / 100 * np.max(xcoords)
        ymin = render_yrange[0] / 100 * np.max(ycoords)
        ymax = render_yrange[1] / 100 * np.max(ycoords)
        zmin = render_zrange[0] / 100 * np.max(zcoords)
        zmax = render_zrange[1] / 100 * np.max(zcoords)

        filter_idx = np.where((xcoords > xmax) | (xcoords < xmin) | (ycoords > ymax) |
                              (ycoords < ymin) | (zcoords > zmax) | (zcoords < zmin))
        self.locs_active = np.delete(self.locs_active, filter_idx)

    def get_all_coords_rec_array(self):
        # Returns the coordinates of the all localizations, with
        # the offset applied.

        COORDS_DTYPE = [('x_pos_pixels', 'f4'),
                        ('y_pos_pixels', 'f4'),
                        ('z_pos_pixels', 'f4')]

        tmp_x = self.locs_all.x_pos_pixels
        tmp_y = self.locs_all.y_pos_pixels
        tmp_z = self.locs_all.z_pos_pixels

        tmp_records = np.recarray((tmp_x.size,), dtype=COORDS_DTYPE)
        tmp_records.x_pos_pixels = tmp_y
        tmp_records.y_pos_pixels = tmp_x
        tmp_records.z_pos_pixels = tmp_z

        return tmp_records

    def get_active_coords_rec_array(self):
        # Returns the coordinates of the all localizations, with
        # the offset applied.

        COORDS_DTYPE = [('x_pos_pixels', 'f4'),
                        ('y_pos_pixels', 'f4'),
                        ('z_pos_pixels', 'f4')]

        tmp_x = self.locs_active.x_pos_pixels
        tmp_y = self.locs_active.y_pos_pixels
        tmp_z = self.locs_active.z_pos_pixels

        tmp_records = np.recarray((tmp_x.size,), dtype=COORDS_DTYPE)
        tmp_records.x_pos_pixels = tmp_y
        tmp_records.y_pos_pixels = tmp_x
        tmp_records.z_pos_pixels = tmp_z

        return tmp_records


