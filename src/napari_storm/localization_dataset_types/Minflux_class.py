import os

import pandas as pd
from PyQt5.QtWidgets import QFileDialog

from .base_class import LocalizationDataBaseClass
from .data_formats import *
import collections
import numpy as np

MINFLUX_Z_CORRECTION_FACTOR = 0.8


class MinfluxDataBaseClass(LocalizationDataBaseClass):
    """An Object which contains MINFLUX localization data,
    Subclass of LocalizationDataBaseClass"""

    def __init__(self,
                 locs=None,
                 name=None,
                 zdim_present=False,
                 ):
        self.locs_dtype = []
        super().__init__(locs, name=name, zdim_present=zdim_present)
        self.dataset_type = "MinfluxDataBaseClass(LocalizationDataBaseClass)"
        if locs is None:
            pass
        else:
            self.add_minflux_dtype()
            self.locs_sanity_check()

    def import_recognized_data(self, data, metadata=None):
        data = np.rec.array(data, metadata["dataset_class_dtype"])
        metadata = MinfluxDataBaseClass().check_if_metadata_is_complete(metadata)
        return MinfluxDataBaseClass(locs=data, name=metadata["name"], zdim_present=metadata["zdim_present"])

    def add_minflux_dtype(self):
        self.init_dtype(self.zdim_present)
        self.locs_dtype.append(('trace_id', 'i4'))

    @property
    def sigma_present(self):
        return False

    @property
    def photon_count_present(self):
        return False

    @property
    def uncertainty_defined(self):
        return False


class MinfluxDataAIClass:
    # Collection of several minflux datasets
    def __init__(self,
                 list_of_datasets=None,
                 name=None
                 ):
        self.dataset_type = "MinfluxDataAIClass"
        self.list_of_datasets = []
        self.locs_dtype = []
        self.add_ai_mfx_dtype()
        if list_of_datasets is not None:
            if name is None:
                self.name = 'untitled'
            else:
                self.name = name
            for i in range(len(list_of_datasets)):
                self.list_of_datasets.append(list_of_datasets[i])
                assert list_of_datasets.locs.dtype == self.locs_dtype

    def add_ai_mfx_dtype(self):
        self.locs_dtype = minflux_AI_data_dtype

    def load_mfxs(self, file_path, name=None, as_list=False):
        data = np.load(file_path)
        for i in range(len(data)):
            mfx_data = np.rec.array(data[i])
            if i == 0:
                try:
                    mfx_data.z_pos_nm
                    zdim = True
                except:
                    zdim = False
            self.list_of_datasets.append(MinfluxDataAIIterationClass(locs=mfx_data, itr=i, zdim_present=zdim,
                                                                     name=name))
        if as_list:
            return self.list_of_datasets
        else:
            return self

    def load_ai_json(self, file_path, specific_itr=None, as_list=False):
        filename = file_path.split("/")[-1]
        raw_data = pd.read_json(file_path)
        raw_data = raw_data[raw_data["vld"] == True]
        mes_time_s = raw_data["tim"]
        activation = raw_data["act"]
        tid = raw_data["tid"]
        n_locs = len(raw_data.itr)
        vld_indices = raw_data.itr.keys()
        raw_locs_m = np.zeros((3, n_locs))
        eco = np.zeros(n_locs)
        ecc = np.zeros(n_locs)
        efo = np.zeros(n_locs)
        efc = np.zeros(n_locs)
        sta = np.zeros(n_locs)
        cfr = np.zeros(n_locs)
        dcr = np.zeros(n_locs)
        gvy = np.zeros(n_locs)
        gvx = np.zeros(n_locs)
        eoy = np.zeros(n_locs)
        eox = np.zeros(n_locs)
        dmz = np.zeros(n_locs)
        lcy = np.zeros(n_locs)
        lcx = np.zeros(n_locs)
        lcz = np.zeros(n_locs)
        fbg = np.zeros(n_locs)
        tic = np.zeros(n_locs)
        lnc = np.zeros((3, n_locs))
        ext = np.zeros((3, n_locs))
        if len(raw_data.itr[0]) == 10:
            zdim = True
        else:
            zdim = False
        if specific_itr is None:
            for itr in range(len(raw_data.itr[0])):
                for i in range(n_locs):
                    raw_locs_m[:, i] = raw_data.itr[vld_indices[i]][itr]["loc"]
                    eco[i] = raw_data.itr[vld_indices[i]][itr]["eco"]
                    ecc[i] = raw_data.itr[vld_indices[i]][itr]["ecc"]
                    efo[i] = raw_data.itr[vld_indices[i]][itr]["efo"]
                    efc[i] = raw_data.itr[vld_indices[i]][itr]["efc"]
                    sta[i] = raw_data.itr[vld_indices[i]][itr]["sta"]
                    cfr[i] = raw_data.itr[vld_indices[i]][itr]["cfr"]
                    dcr[i] = raw_data.itr[vld_indices[i]][itr]["dcr"]
                    gvy[i] = raw_data.itr[vld_indices[i]][itr]["gvy"]
                    gvx[i] = raw_data.itr[vld_indices[i]][itr]["gvx"]
                    eoy[i] = raw_data.itr[vld_indices[i]][itr]["eoy"]
                    eox[i] = raw_data.itr[vld_indices[i]][itr]["eox"]
                    dmz[i] = raw_data.itr[vld_indices[i]][itr]["dmz"]
                    lcy[i] = raw_data.itr[vld_indices[i]][itr]["lcy"]
                    lcx[i] = raw_data.itr[vld_indices[i]][itr]["lcx"]
                    lcz[i] = raw_data.itr[vld_indices[i]][itr]["lcz"]
                    fbg[i] = raw_data.itr[vld_indices[i]][itr]["fbg"]
                    tic[i] = raw_data.itr[vld_indices[i]][itr]["tic"]
                    lnc[:, i] = raw_data.itr[vld_indices[i]][itr]["lnc"]
                    ext[:, i] = raw_data.itr[vld_indices[i]][itr]["ext"]
                mfx_data = np.rec.array(
                    (raw_locs_m[0, :] * 1E9,
                     raw_locs_m[1, :] * 1E9,
                     raw_locs_m[2, :] * 1E9 * MINFLUX_Z_CORRECTION_FACTOR,
                     eco,
                     ecc,
                     efo,
                     efc,
                     sta,
                     cfr,
                     dcr,
                     gvy,
                     gvx,
                     eoy,
                     eox,
                     dmz,
                     lcy,
                     lcx,
                     lcz,
                     fbg,
                     tic,
                     lnc[0],
                     lnc[1],
                     lnc[2],
                     ext[0],
                     ext[1],
                     ext[2],
                     mes_time_s,
                     activation,
                     tid,
                     ), dtype=minflux_AI_data_dtype, )
                self.list_of_datasets.append(MinfluxDataAIIterationClass(itr=itr, locs=mfx_data, zdim_present=zdim,
                                                                         name=filename))
            if as_list:
                return self.list_of_datasets
            else:

                return self
        else:
            for i in range(n_locs):
                raw_locs_m[:, i] = raw_data.itr[vld_indices[i]][specific_itr]["loc"]
                eco[i] = raw_data.itr[vld_indices[i]][specific_itr]["eco"]
                ecc[i] = raw_data.itr[vld_indices[i]][specific_itr]["ecc"]
                efo[i] = raw_data.itr[vld_indices[i]][specific_itr]["efo"]
                efc[i] = raw_data.itr[vld_indices[i]][specific_itr]["efc"]
                sta[i] = raw_data.itr[vld_indices[i]][specific_itr]["sta"]
                cfr[i] = raw_data.itr[vld_indices[i]][specific_itr]["cfr"]
                dcr[i] = raw_data.itr[vld_indices[i]][specific_itr]["dcr"]
                gvy[i] = raw_data.itr[vld_indices[i]][specific_itr]["gvy"]
                gvx[i] = raw_data.itr[vld_indices[i]][specific_itr]["gvx"]
                eoy[i] = raw_data.itr[vld_indices[i]][specific_itr]["eoy"]
                eox[i] = raw_data.itr[vld_indices[i]][specific_itr]["eox"]
                dmz[i] = raw_data.itr[vld_indices[i]][specific_itr]["dmz"]
                lcy[i] = raw_data.itr[vld_indices[i]][specific_itr]["lcy"]
                lcx[i] = raw_data.itr[vld_indices[i]][specific_itr]["lcx"]
                lcz[i] = raw_data.itr[vld_indices[i]][specific_itr]["lcz"]
                fbg[i] = raw_data.itr[vld_indices[i]][specific_itr]["fbg"]
                tic[i] = raw_data.itr[vld_indices[i]][specific_itr]["tic"]
                lnc[:, i] = raw_data.itr[vld_indices[i]][specific_itr]["lnc"]
                ext[:, i] = raw_data.itr[vld_indices[i]][specific_itr]["ext"]
            mfx_data = np.rec.array(
                (raw_locs_m[0, :] * 1E9,
                 raw_locs_m[1, :] * 1E9,
                 raw_locs_m[2, :] * 1E9 * MINFLUX_Z_CORRECTION_FACTOR,
                 eco,
                 ecc,
                 efo,
                 efc,
                 sta,
                 cfr,
                 dcr,
                 gvy,
                 gvx,
                 eoy,
                 eox,
                 dmz,
                 lcy,
                 lcx,
                 lcz,
                 fbg,
                 tic,
                 lnc[0],
                 lnc[1],
                 lnc[2],
                 ext[0],
                 ext[1],
                 ext[2],
                 mes_time_s,
                 activation,
                 tid,
                 ), dtype=minflux_AI_data_dtype, )
            return mfx_data, filename, zdim

    def load_ai_npy(self, file_path, specific_itr=None, as_list=False):
        filename = file_path.split("/")[-1]
        raw_data = np.load(file_path)
        mes_time_s = raw_data["tim"]
        activation = raw_data["act"]
        tid = raw_data["tid"]
        vld = raw_data["vld"]
        raw_data = (raw_data[vld])["itr"]

        raw_locs_m = raw_data["loc"]  # shape is (n_locs, itr, dim)
        n_locs = len(raw_locs_m[:, 0, 0])
        if np.all(raw_locs_m[0, :, 2] == 0):
            zdim = False
            itr_steps = 5 
        else:
            zdim = True
            itr_steps = 10

        eco = raw_data["eco"]
        ecc = raw_data["ecc"]
        efo = raw_data["efo"]
        efc = raw_data["efc"]
        sta = raw_data["sta"]
        cfr = raw_data["cfr"]
        dcr = raw_data["dcr"]
        gvy = raw_data["gvy"]
        gvx = raw_data["gvx"]
        eoy = raw_data["eoy"]
        eox = raw_data["eox"]
        dmz = raw_data["dmz"]
        lcy = raw_data["lcy"]
        lcx = raw_data["lcx"]
        lcz = raw_data["lcz"]
        fbg = raw_data["fbg"]
        tic = raw_data["tic"]
        lnc = raw_data["lnc"].reshape((n_locs, itr_steps, 3))
        ext = raw_data["ext"].reshape((n_locs, itr_steps, 3))
        if specific_itr is None:
            for itr in range(itr_steps):
                mfx_data = np.rec.array(
                    (raw_locs_m[:, itr, 0] * 1E9,
                     raw_locs_m[:, itr, 1] * 1E9,
                     raw_locs_m[:, itr, 2] * 1E9 * MINFLUX_Z_CORRECTION_FACTOR,
                     eco[:, itr],
                     ecc[:, itr],
                     efo[:, itr],
                     efc[:, itr],
                     sta[:, itr],
                     cfr[:, itr],
                     dcr[:, itr],
                     gvx[:, itr],
                     gvy[:, itr],
                     eoy[:, itr],
                     eox[:, itr],
                     dmz[:, itr],
                     lcy[:, itr],
                     lcx[:, itr],
                     lcz[:, itr],
                     fbg[:, itr],
                     tic[:, itr],
                     lnc[:, itr, 0],
                     lnc[:, itr, 1],
                     lnc[:, itr, 2],
                     ext[:, itr, 0],
                     ext[:, itr, 1],
                     ext[:, itr, 2],
                     mes_time_s,
                     activation,
                     tid,
                     ), dtype=self.locs_dtype, )
                self.list_of_datasets.append(MinfluxDataAIIterationClass(itr=itr, locs=mfx_data, zdim_present=zdim,
                                                                         name=filename))
            if as_list:
                return self.list_of_datasets
            else:
                return self
        else:
            mfx_data = np.rec.array(
                (raw_locs_m[:, specific_itr, 0] * 1E9,
                 raw_locs_m[:, specific_itr, 1] * 1E9,
                 raw_locs_m[:, specific_itr, 2] * 1E9 * MINFLUX_Z_CORRECTION_FACTOR,
                 eco[:, specific_itr],
                 ecc[:, specific_itr],
                 efo[:, specific_itr],
                 efc[:, specific_itr],
                 sta[:, specific_itr],
                 cfr[:, specific_itr],
                 dcr[:, specific_itr],
                 gvx[:, specific_itr],
                 gvy[:, specific_itr],
                 eoy[:, specific_itr],
                 eox[:, specific_itr],
                 dmz[:, specific_itr],
                 lcy[:, specific_itr],
                 lcx[:, specific_itr],
                 lcz[:, specific_itr],
                 fbg[:, specific_itr],
                 tic[:, specific_itr],
                 lnc[:, specific_itr, 0],
                 lnc[:, specific_itr, 1],
                 lnc[:, specific_itr, 2],
                 ext[:, specific_itr, 0],
                 ext[:, specific_itr, 1],
                 ext[:, specific_itr, 2],
                 mes_time_s,
                 activation,
                 tid,
                 ), dtype=minflux_AI_data_dtype, )
            return mfx_data, filename, zdim


class MinfluxDataAIIterationClass(MinfluxDataBaseClass):
    def __init__(self,
                 locs=None,
                 itr=0,
                 name=None,
                 zdim_present=False,
                 ):

        if locs is not None:
            super().__init__(None, name, zdim_present)
            self.locs_all = locs
            self.locs_active = locs
            if name is None:
                self.name = 'untitled'
            else:
                self.name = name
        self.dataset_type = "MinfluxDataAIIterationClass(MinfluxDataBaseClass)"
        self.itr = itr
        self.locs_dtype = minflux_AI_data_dtype
        self.zdim_present = zdim_present

    def load_ns(self, dataset):
        tmp_name = dataset.attrs["name"]
        tmp_zdim_present = dataset.attrs["zdim_present"]
        tmp_itr = dataset.attrs["itr"]
        return MinfluxDataAIIterationClass(locs=np.rec.array(dataset[...]), name=tmp_name, zdim_present=tmp_zdim_present,
                                           itr=tmp_itr)

    def export_current_iteration_as_mfx_file(self, filename=None):
        if filename is None:
            filename = QFileDialog.getSaveFileName(caption="Save File", filter=".mfx")
            filename = filename[0]
        np.save(filename + ".npy", self.locs)
        os.rename(filename + ".npy", filename + f"_itr_{self.itr}.mfx")

    def load_mfx(self, file_path, name=None):
        if name is None:
            self.name = 'Untitled'
        else:
            self.name = name
        try:
            itr = int(file_path.split("_")[-1].split(".")[0])
        except ValueError:
            itr = 0
        self.locs_all = np.rec.array(np.load(file_path))
        self.locs_active = self.locs_all
        try:
            self.locs_active.z_pos_nm
            self.zdim_present = True
        except:
            self.zdim_present = False
        return self

    def load_single_itr(self, file_path, itr, name=None):
        if file_path.split('.')[-1] == 'npy':
            self.locs_all, self.name, self.zdim_present = MinfluxDataAIClass().load_ai_npy(file_path, specific_itr=itr)
        elif file_path.split('.')[-1] == 'json':
            self.locs_all, self.name, self.zdim_present = MinfluxDataAIClass().load_ai_json(file_path, specific_itr=itr)
        if name is not None:
            self.name = name
        self.locs_active = self.locs_all
        return self



