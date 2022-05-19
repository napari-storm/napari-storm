import numpy as np
import pandas as pd

import os.path as _ospath


import h5py
import yaml as _yaml
import io
import json
import logging
import struct

# load_SMLM adapted from Marting Weigerts readSmlmFile
import zipfile

from PyQt5.QtWidgets import QInputDialog, QFileDialog as fd

from .ns_constants import *
from .LocalizationData import LocalizationData
from .CustomErrors import *


class FileToLocalizationDataInterface:

    def __init__(self, parent, n_datasets=0):
        self._parent = parent
        self.n_datasets = n_datasets
        self.dataset_names = []

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError('Cannot change parent of existing Widget')

    def filetype_recognition(self):
        # Check what kind of filetype and compare with defined featrues of a list of recognized filetypes
        pass

    def open_localization_data_file_and_get_dataset(self, file_path=None):
        """Determine which file type is being opened, and call the
        corresponding importer function"""
        if not file_path:
            file_path = fd.getOpenFileName(None, 'Open mol list', '.', 'Localization files (*.h5)')[0]
        return self.recognize_storm_data_and_return_dataset(file_path)

    def recognize_storm_data_and_return_dataset(self, file_path):
        filetype = file_path.split('.')[-1]
        self.n_datasets += 1
        # Picasso hdf5 -> always needs a yaml present
        if filetype == 'hdf5':
            if _ospath.isfile(file_path[: -(len(filetype))] + 'yaml'):
                return self.load_hdf5(file_path)
        elif filetype == 'yaml':
            file_path = file_path[: -(len(filetype))] + 'hdf5'
            if _ospath.isfile(file_path):
                return self.load_hdf5(file_path)
        # Thunderstorm csv
        elif filetype == 'csv':
            return self.load_csv(file_path)
        # SMLM File
        elif filetype == 'smlm':
            return self.load_SMLM(file_path)
        # h5 -> special type of hdf5
        elif filetype == 'h5':
            return self.load_h5(file_path)
        # MINFLUX files
        elif filetype == 'json':
            return self.load_mfx_json(file_path)
        elif filetype == 'npy':
            return self.load_mfx_npy(file_path)
        elif filetype == 'test':
            return self.start_testing()
        elif filetype == 'your_custom_ending':
            return self.your_custom_import_function(file_path)
        self.n_datasets -= 1
        raise TypeError('Unknown SMLM data file extension')

    def your_custom_import_function(self, file_path):
        """Use this Function as a template to import the data you want
            Start with importing your data however you like
            After you imported your data you need to create a numpy rec array like this:

            locs_rec_array = np.rec.array(
                                (   Framenumbers,
                                    array_of_x_pos_in_pixelunits,
                                    array_of_y_pos_in_pixelunits,
                                    array_of_z_pos_in_pixelunits,
                                    array_of_uncertainties_in_x_in_pixelunits,
                                    array_of_uncertainties_in_y_in_pixelunits,
                                    array_of_uncertainties_in_z_in_pixelunits,
                                    array_of_intensity_values_for_each_loc,)
                                    , dtype=LOCS_DTYPE)

            The Z positions, Framenumbers, uncertainties and intensity values are optional,
            just set them to np.ones(len(array_of_x_pos_in_pixelunits)) if you want to skip them.

            Next you want to check if there is already a file present with the same name, for that
            just call the check_namespace function:

            filename=self.check_namespace(self,filename), where

            filename is typically file_path.split('/')[-1] (= Filepath without the path)

            Next you need to return the gathered infos, while also providing information on pixelsize,
            if z positions are present (zdim_present = True/False),
            if uncertainty values are present (sigma_present = True/False)
            if intensity values are present (photon_count_present = True/False):

             return [LocalizationData(locs=locs_rec_array, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim, sigma_present=sigma_present,
                                 photon_count_present=photon_count_present,
                                 parent=self.parent)]

            If you are sure your function is working, as a last step go into the
            recognize-storm_data_and_return_dataset function, replace 'your_custom_ending' with the ending
            or if there is already a function for the filetype, just replace the existing function with yours,
            e.g. you want to import .csv, go to

             elif filetype == 'csv':
                return self.load_csv(file_path)
             and replace self.load_csv with self.your_custom_import_function
        """


    def check_namespace(self, name, idx=1):
        # print("checking the namespace:",name)
        for i in range(self.n_datasets-1):
            if self.dataset_names[i] == name:
                return self.check_namespace( name + "_" + str(idx + 1), idx + 1)
        return name

    def load_info(self, path):
        """Loads Infos from Picassos .yaml"""
        path_base, path_extension = _ospath.splitext(path)
        filename = path_base + ".yaml"
        try:
            with open(filename) as info_file:
                info = list(_yaml.load_all(info_file, Loader=_yaml.FullLoader))
        except FileNotFoundError as e:
            print(f"\nAn error occured. Could not find metadata file:\n{filename}")
        return info

    def load_locs(self,path):
        """Loads Picassos .hdf5 files"""
        with h5py.File(path, "r") as locs_file:
            locs = locs_file["locs"][...]
        locs = np.rec.array(
            locs, dtype=locs.dtype
        )  # Convert to rec array with fields as attributes
        info = self.load_info(path)
        return locs, info

    def load_hdf5(self, file_path):
        """Wrapper for load_locs and load_infos -> picassos hdf5"""
        filename = file_path.split("/")[-1]
        locs, info = self.load_locs(file_path)
        try:
            pixelsize = locs.pixelsize_nm
        except AttributeError:
            pixelsize,ok = QInputDialog.getText(self.parent,'Pixelsize',f"Enter the pixelsize [nm]")
            if not ok:
                raise PixelSizeIsNeccessaryError('Pixelsize is mandatory')
        pixelsize = float(pixelsize)
        try:
            locs.z=locs.z/pixelsize
            zdim = True
        except AttributeError:
            locs.z=np.ones(len(locs.x))
            zdim = False

        sigma_present = False
        photon_count_present = False

        try:
            uncertainty_x_pixels = locs.lpx
            uncertainty_y_pixels = locs.lpy
            sigma_present = True
        except AttributeError:
            uncertainty_x_pixels = np.ones(len(locs.x))
            uncertainty_y_pixels = np.ones(len(locs.x))
        try:
            uncertainty_z_pixels = locs.lpz
        except AttributeError:
            uncertainty_z_pixels = 2*np.sqrt(locs.lpx**2+locs.lpy**2)

        try:
            intensity_photons = locs.photons
            photon_count_present = True
        except AttributeError:
            intensity_photons = np.ones(len(locs.x))
        locs = np.rec.array(
        (   locs.frame,
            locs.x,
            locs.y,
            locs.z,
            uncertainty_x_pixels,
            uncertainty_y_pixels,
            uncertainty_z_pixels,
            intensity_photons,)
            , dtype=LOCS_DTYPE)
        filename = self.check_namespace( filename)
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=sigma_present, photon_count_present=photon_count_present,
                                 parent=self.parent)]

    def load_h5(self, file_path):
        """Loads localizations from .h5 files"""
        filename = file_path.split("/")[-1]
        with h5py.File(file_path, "r") as locs_file:
            data = locs_file["molecule_set_data"]["datatable"][...]
            pixelsize = (
                    locs_file["molecule_set_data"]["xy_pixel_size_um"][...] * 1e3
            )  # to µm to nm
        try:
            data["Z_POS_PIXELS"]
            zdim=True
        except:
            data["Z_POS_PIXELS"]=np.ones(len(data["X_POS_PIXELS"]))
            zdim=False
        locs = np.rec.array(
        (   data["FRAME_NUMBER"],
            data["X_POS_PIXELS"],
            data["Y_POS_PIXELS"],
            data["Z_POS_PIXELS"],
            np.ones(len(data["Z_POS_PIXELS"])),
            np.ones(len(data["Z_POS_PIXELS"])),
            np.ones(len(data["Z_POS_PIXELS"])),
            data["PHOTONS"],)
            , dtype=LOCS_DTYPE)
        num_channel = max(data["CHANNEL"]) + 1
        list_of_datasets=[]
        for i in range(num_channel):
            filename_pluschannel = self.check_namespace( filename + f" Channel {i + 1}")
            self.dataset_names.append(filename_pluschannel)
            locs_in_ch = locs[data["CHANNEL"] == i]
            list_of_datasets.append(LocalizationData(locs=locs_in_ch, name=filename_pluschannel, pixelsize_nm=pixelsize,
                                                     zdim_present=zdim,
                                                     sigma_present=False, photon_count_present=True, parent=self.parent))
        return list_of_datasets

    def load_mfx_json(self, file_path):
        """Loads MFX Data from json files -> Needs to be redone when a working example file is present """
        filename = file_path.split("/")[-1]
        raw_data = pd.read_json(file_path)
        raw_data = raw_data[raw_data["vld"] == True]
        n_locs = len(raw_data.itr)
        vld_indices = raw_data.itr.keys()
        if len(raw_data.itr) == 10:
            raw_locs = np.zeros((3, n_locs))
            zdim = True
            for i in range(n_locs):
                raw_locs[:, i] = raw_data.itr[vld_indices[i]][9]["loc"]
        else:
            zdim = False
            raw_locs = np.zeros((3, n_locs))
            for i in range(n_locs):
                raw_locs[:, i] = raw_data.itr[vld_indices[i]][4]["loc"]
        frames = np.ones(n_locs)
        photons = 1000 * np.ones(n_locs)
        pixelsize = 1
        locs = np.rec.array(
            (frames, raw_locs[0, :]*1E9, raw_locs[1, :]*1E9, raw_locs[2, :]*1E9*MINFLUX_Z_CORRECTION_FACTOR,
             np.ones(n_locs), np.ones(n_locs),
             np.ones(n_locs), photons), dtype=LOCS_DTYPE,)
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=False, photon_count_present=True, parent=self.parent)]

    def load_mfx_npy(self, file_path):
        """Loads MFX Data from numpy (.npy) files"""
        filename = file_path.split("/")[-1]
        raw_data = np.load(file_path)
        locsx = []
        locsy = []
        minx = min(raw_data["itr"]["loc"][:, -1, 0])
        miny = min(raw_data["itr"]["loc"][:, -1, 1])
        try:
            minz = min(raw_data["itr"]["loc"][:, -1, 2])
            zdim = True
        except:
            zdim = False
        if zdim:
            locsz = []
            for i in range(len(raw_data)):
                if raw_data["vld"][i]:
                    locsx.append((raw_data["itr"]["loc"][i, -1, 0]) * 1e9)
                    locsy.append((raw_data["itr"]["loc"][i, -1, 1]) * 1e9)
                    locsz.append((raw_data["itr"]["loc"][i, -1, 2]) * 1e9 * MINFLUX_Z_CORRECTION_FACTOR)
            frames = np.ones(len(locsx))
            photons = 1000 * np.ones(len(locsx))
            pixelsize = 1
            locs = np.rec.array(
                (frames, locsx, locsy, locsz, np.ones(len(locsx)), np.ones(len(locsx)), np.ones(len(locsx)),  photons),
                dtype=LOCS_DTYPE,
            )
        else:
            for i in range(len(raw_data)):
                if raw_data["vld"][i]:
                    locsx.append((raw_data["itr"]["loc"][i, -1, 0] - minx) * 1e9)
                    locsy.append((raw_data["itr"]["loc"][i, -1, 1] - miny) * 1e9)
            locsz=np.ones(locsx)
            frames = np.ones(len(locsx))
            photons = 1000 * np.ones(len(locsx))
            pixelsize = 1
            locs = np.rec.array(
                (frames, locsx, locsy, locsz,np.ones(len(locsx)),np.ones(len(locsx)),np.ones(len(locsx)), photons),
                dtype=LOCS_DTYPE,
            )
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=False, photon_count_present=True, parent=self.parent)]

    def load_csv(self, file_path):
        """Loads Thunderstorm .csv files"""
        filename = file_path.split("/")[-1]
        data = {}
        photon_count_present = False
        sigma_present = False

        with open(file_path) as infile:
            header = infile.readline()
            header = header.replace("\n", "")
            header = header.split(",")
            data_list = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=float)
        for i in range(len(header)):
            data[header[i]] = data_list[:, i]
        if 'pixelsize' in header:
            pixelsize = data['"pixelsize"']
        else:
            pixelsize, ok = QInputDialog.getText(self.parent,'Pixelsize',f"Enter the pixelsize [nm]")
            if not ok:
                raise PixelSizeIsNeccessaryError('Pixelsize is mandatory')
        pixelsize = float(pixelsize)

        if '"x [nm]"' in header:
            locs_pos_x_nm = data['"x [nm]"']
            locs_pos_y_nm = data['"y [nm]"']
        elif 'x [nm]' in header:
            locs_pos_x_nm = data['x [nm]']
            locs_pos_y_nm = data['y [nm]']
        else:
            raise ImportError('Localisation Position in X or Y not found in header')

        if '"z [nm]"' in header:
            locs_pos_z_nm = data['"z [nm]"']
            zdim = True
        elif 'z [nm]' in header:
            locs_pos_z_nm = data['z [nm]']
            zdim = True
        else:
            locs_pos_z_nm=np.ones(len(locs_pos_x_nm))
            zdim=False

        # Check if frame number info is present in file
        if '"frame"' in header:
            frame_numbers=data['"frame"']
        elif 'frame' in header:
            frame_numbers = data['frame']
        else:
            frame_numbers=np.ones(len(locs_pos_x_nm))

        # Check if uncertainty info is present in file
        if 'uncertainty_xy [nm]' in header:
            uncertainty_x_nm = data['uncertainty_xy [nm]']
            uncertainty_y_nm = data['uncertainty_xy [nm]']
            intensity_photons = np.ones(len(locs_pos_x_nm))
            sigma_present = True
            if zdim:
                uncertainty_z_nm=data['uncertainty_z [nm]']
            else:
                uncertainty_z_nm=np.ones(len(locs_pos_x_nm))
        elif 'uncertainty_x [nm]' in header:
            uncertainty_x_nm = data['uncertainty_x [nm]']
            uncertainty_y_nm = data['uncertainty_y [nm]']
            intensity_photons = np.ones(len(locs_pos_x_nm))
            sigma_present = True
            if zdim and "uncertainty_z [nm]" in header:
                uncertainty_z_nm=data['uncertainty_z [nm]']
            else:
                uncertainty_z_nm=2*np.sqrt(uncertainty_x_nm**2+uncertainty_y_nm**2)
        elif '"intensity [photon]"' in header:
            photon_count_present=True
            intensity_photons=data['"intensity [photon]"']
            uncertainty_x_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_y_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_z_nm = np.ones(len(locs_pos_x_nm))
        elif 'intensity [photon]' in header:
            photon_count_present = True
            intensity_photons=data['intensity [photon]']
            uncertainty_x_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_y_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_z_nm = np.ones(len(locs_pos_x_nm))
        else:
            uncertainty_x_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_y_nm = np.ones(len(locs_pos_x_nm))
            uncertainty_z_nm = np.ones(len(locs_pos_x_nm))
            intensity_photons = np.ones(len(locs_pos_x_nm))
        locs = np.rec.array(
            (
                frame_numbers,
                locs_pos_x_nm / pixelsize,
                locs_pos_y_nm / pixelsize,
                locs_pos_z_nm / pixelsize,
                uncertainty_x_nm / pixelsize,
                uncertainty_y_nm / pixelsize,
                uncertainty_z_nm / pixelsize,
                intensity_photons,
            ),
            dtype=LOCS_DTYPE,
        )
        filename = self.check_namespace( filename)
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=sigma_present, photon_count_present=photon_count_present,
                                 parent=self.parent)]

    def load_SMLM(self, file_path):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        dtype2struct = {"uint8": "B", "uint32": "I", "float64": "d", "float32": "f"}
        dtype2length = {"uint8": 1, "uint32": 4, "float64": 8, "float32": 4}

        """Loads SMLM Files"""
        filename = file_path.split(".")[-1]
        zf = zipfile.ZipFile(file_path, "r")
        file_names = zf.namelist()
        if "manifest.json" in file_names:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["format_version"] == "0.2"
            for file_info in manifest["files"]:
                if file_info["type"] == "table":
                    logger.info("loading table...")
                    format_key = file_info["format"]
                    file_format = manifest["formats"][format_key]
                    if file_format["mode"] == "binary":
                        try:
                            table_file = zf.read(file_info["name"])
                            logger.info(file_info["name"])
                        except KeyError:
                            logger.error(
                                "ERROR: Did not find %s in zip file", file_info["name"]
                            )
                            continue
                        else:
                            logger.info("loading table file: %s bytes", len(table_file))
                            logger.info("headers: %s", file_format["headers"])
                            headers = file_format["headers"]
                            dtype = file_format["dtype"]
                            shape = file_format["shape"]
                            cols = len(headers)
                            rows = file_info["rows"]
                            logger.info("rows: %s, columns: %s", rows, cols)
                            assert len(headers) == len(dtype) == len(shape)
                            rowLen = 0
                            for i, h in enumerate(file_format["headers"]):
                                rowLen += dtype2length[dtype[i]]

                            tableDict = {}
                            byteOffset = 0
                            try:
                                import numpy as np

                                for i, h in enumerate(file_format["headers"]):
                                    tableDict[h] = np.ndarray(
                                        (rows,),
                                        buffer=table_file,
                                        dtype=dtype[i],
                                        offset=byteOffset,
                                        order="C",
                                        strides=(rowLen,),
                                    )
                                    byteOffset += dtype2length[dtype[i]]
                            except ImportError:
                                logger.warning(
                                    "Failed to import numpy, performance will drop dramatically. Please install numpy for the best performance."
                                )
                                st = ""
                                for i, h in enumerate(file_format["headers"]):
                                    st += str(shape[i]) + dtype2struct[dtype[i]]

                                unpack = struct.Struct(st).unpack
                                tableDict = {h: [] for h in headers}
                                for i in range(0, len(table_file), rowLen):
                                    unpacked_data = unpack(table_file[i: i + rowLen])
                                    for j, h in enumerate(headers):
                                        tableDict[h].append(unpacked_data[j])
                                tableDict = {
                                    h: np.array(tableDict[h]) for i, h in enumerate(headers)
                                }
                            data = {}
                            data["min"] = [tableDict[h].min() for h in headers]
                            data["max"] = [tableDict[h].max() for h in headers]
                            data["avg"] = [tableDict[h].mean() for h in headers]
                            data["tableDict"] = tableDict
                            file_info["data"] = data
                            logger.info("table file loaded: %s", file_info["name"])
                    else:
                        raise Exception(
                            "format mode {} not supported yet".format(file_format["mode"])
                        )
                elif file_info["type"] == "image":
                    if file_format["mode"] == "binary":
                        try:
                            image_file = zf.read(file_info["name"])
                            logger.info("image file loaded: %s", file_info["name"])
                        except KeyError:
                            logger.error(
                                "ERROR: Did not find %s in zip file", file_info["name"]
                            )
                            continue
                        else:
                            from PIL import Image

                            image = Image.open(io.BytesIO(image_file))
                            data = {}
                            data["image"] = image
                            file_info["data"] = data
                            logger.info("image file loaded: %s", file_info["name"])

                else:
                    logger.info("ignore file with type: %s", file_info["type"])
        else:
            raise Exception("invalid file: no manifest.json found in the smlm file")
        prop = manifest["files"][-1]["data"]["tableDict"]
        try:
            pixelsize = prop["pixelsize"]
        except:
            pixelsize, ok = QInputDialog.getText(self.parent,'Pixelsize',f"Enter the pixelsize [nm]")
            if not ok:
                raise PixelSizeIsNeccessaryError('Pixelsize is mandatory')
        pixelsize = float(pixelsize)
        if (
                not "intensity_photon_" in prop.keys()
        ):  # If the photons are not given, set them to 1k
            prop["intensity_photon_"] = 1000 * np.ones(len(prop["x"]))
        try:
            prop["z"]
            zdim = True
        except:
            prop["z"]=np.ones(len(prop["x"]))
            zdim = False

        locs = np.rec.array(
            (
                prop["frame"],
                prop["x"] / pixelsize,
                prop["y"] / pixelsize,
                prop["z"] / pixelsize,
                np.ones(len(prop["x"])),
                np.ones(len(prop["x"])),
                np.ones(len(prop["x"])),
                prop["intensity_photon_"],
            ),
            dtype=LOCS_DTYPE,
        )
        filename = self.check_namespace( filename)
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=False, photon_count_present=True, parent=self.parent)]

    ##### Lowest Order functions
    def start_testing(self):
        n = 6

        zdim = True
        locs = np.rec.array(
            (
                np.repeat(np.arange(1, n + 1), 2),
                np.repeat(np.arange(1, n + 1), 2),
                np.arange(1, 2 * n + 1),
                np.repeat(np.arange(0, 2), n),
                np.ones(2*n),
                np.ones(2 * n),
                np.ones(2 * n),
                np.repeat(np.ones(n), 2),
            ),
            dtype=LOCS_DTYPE,
        )
        locs.z_pos_pixels[0] = 2
        pixelsize = 100
        filename = self.check_namespace("a.tester")
        self.dataset_names.append(filename)
        return [LocalizationData(locs=locs, name=filename, pixelsize_nm=pixelsize,
                                 zdim_present=zdim,
                                 sigma_present=False, photon_count_present=True, parent=self.parent)]





