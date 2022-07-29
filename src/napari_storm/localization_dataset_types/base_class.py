
from PyQt5.QtWidgets import QFileDialog

import numpy as np


class LocalizationDataBaseClass:
    """An Object which contains most basic localization data"""

    def __init__(
            self,
            locs=None,
            name=None,
            zdim_present=False,
    ):
        self.dataset_type = "LocalizationDataBaseClass"
        if locs is None:
            self.locs_dtype = None
            self.name = None
            self.locs_active = None
            self.locs_all = None
            self.zdim_present = None
        else:
            self.locs_dtype = self.init_dtype(zdim_present)

            if name is None:
                name = 'Untitled'

            self.name = name
            self.locs_active = locs
            self.locs_all = locs.copy()
            self.zdim_present = zdim_present

    @property
    def locs(self):
        return self.locs_active

    @property
    def x_pos_nm(self):
        return self.locs_active.x_pos_nm

    @property
    def y_pos_nm(self):
        return self.locs_active.y_pos_nm

    @property
    def z_pos_nm(self):
        return self.locs_active.z_pos_nm

    @property
    def x_pos_nm_all(self):
        return self.locs_all.x_pos_nm

    @property
    def y_pos_nm_all(self):
        return self.locs_all.y_pos_nm

    @property
    def z_pos_nm_all(self):
        return self.locs_all.z_pos_nm

    def init_dtype(self, zdim_present):
        if zdim_present:
            locs_dtype = [('x_pos_nm', 'f4'),
                          ('y_pos_nm', 'f4'),
                          ('z_pos_nm', 'f4')]
        else:
            locs_dtype = [('x_pos_nm', 'f4'),
                          ('y_pos_nm', 'f4')]
        return locs_dtype

    def save_as_npy(self, filename=None):
        if filename is None:
            filename = QFileDialog.getSaveFileName(caption="Save File", filter=".npy")
            filename = filename[0]
        metadata = {'dataset_class': LocalizationDataBaseClass, 'name': self.name, 'zdim_present': self.zdim_present}
        np.save(filename + ".npy", [self.locs, metadata])

    def load_npy(self, filename=None):
        if filename is None:
            filename = QFileDialog.getOpenFileName(caption="Open File", filter=".npy")
            filename = filename[0]
        data = np.load(filename + ".npy")
        self.locs_all = data[0]
        self.locs_active = self.locs_all
        return data[1]['dataset_class'](locs=data[0], name=data[1]['name'], zdim_present=data[1]['zdim_present'])

    def locs_sanity_check(self):
        assert isinstance(self.locs, np.recarray), f"locs should be numpy rec array"
        assert self.locs.dtype == self.locs_dtype, f"locs should have {self.locs_dtype} as datatype"

    def restrict_locs_by_percent(self, x_range_pc, y_range_pc, z_range_pc=None):

        self.locs_active = self.locs_all.copy()
        xcoords_nm = self.x_pos_nm - np.min(self.x_pos_nm)
        ycoords_nm = self.y_pos_nm - np.min(self.y_pos_nm)

        xmin = x_range_pc[0] / 100 * np.max(xcoords_nm) + np.min(self.x_pos_nm)
        xmax = x_range_pc[1] / 100 * np.max(xcoords_nm) + np.min(self.x_pos_nm)
        ymin = y_range_pc[0] / 100 * np.max(ycoords_nm) + np.min(self.y_pos_nm)
        ymax = y_range_pc[1] / 100 * np.max(ycoords_nm) + np.min(self.y_pos_nm)

        if self.zdim_present and z_range_pc is not None:
            zcoords_nm = self.z_pos_nm - np.min(self.z_pos_nm)
            zmin = z_range_pc[0] / 100 * np.max(zcoords_nm) + np.min(self.z_pos_nm)
            zmax = z_range_pc[1] / 100 * np.max(zcoords_nm) + np.min(self.z_pos_nm)
            self.bandpass_locs_filter_by_property('z_pos_nm', zmin, zmax)
        self.bandpass_locs_filter_by_property('x_pos_nm', xmin, xmax)
        self.bandpass_locs_filter_by_property('y_pos_nm', ymin, ymax)

    def restrict_locs_by_absolute(self, x_range_nm, y_range_nm, z_range_nm=None):
        assert not (not self.zdim_present and not (z_range_nm is None)), "cannot use restrict in z when" \
                                                                         " z dimension not present "
        self.locs_active = self.locs_all.copy()

        if self.zdim_present:
            self.bandpass_locs_filter_by_property('z_pos_nm', z_range_nm[0], z_range_nm[1])
        self.bandpass_locs_filter_by_property('x_pos_nm', x_range_nm[0], x_range_nm[1])
        self.bandpass_locs_filter_by_property('y_pos_nm', y_range_nm[0], y_range_nm[1])

    def bandpass_locs_filter_by_property(self, prop, l_val=-np.inf, u_val=np.inf):
        if l_val == -np.inf and u_val == np.inf:
            raise ValueError('Nothing to filter here')
        else:
            filter_idx = np.where((getattr(self.locs_active, prop) < l_val) | (getattr(self.locs_active, prop) > u_val))
            self.locs_active = np.delete(self.locs_active, filter_idx)

    def value_specific_locs_filter_by_property(self, prop, values):
        try:
            len(values)
        except TypeError:
            values = [values]
        for i in range(len(values)):
            filter_idx = np.where(not (getattr(self.locs, prop) == values[i]))
            self.locs_active = np.delete(self.locs_active, filter_idx)


