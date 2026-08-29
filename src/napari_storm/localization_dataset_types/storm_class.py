import io
import json
import logging
import os
import struct
import zipfile

import h5py
import numpy as np
import yaml

from ..core import (
    DATA_IN_NM,
    PHOTON_COUNT_PRESENT,
    PIXEL_SIZE_NM,
    SIGMA_PRESENT,
    ZDIM_PRESENT,
    MetadataProvider,
)
from ..CustomErrors import PixelSizeIsNecessaryError
from .base_class import LocalizationDataBaseClass
from .data_formats import storm_data_dtype


def _normalized_csv_header(field):
    """Reduce one CSV header field to a stable lookup key.

    ThunderSTORM quotes every field it writes, other exporters do not, and a
    file that has been through a spreadsheet picks up stray spaces around the
    unit.  Matching the raw text meant one branch per spelling: `"x [nm]"` and
    `x [nm]` were two cases of the same column, and `x  [nm]` -- reported in
    issue #17 -- matched neither.
    """
    return " ".join(field.strip().strip("\"'").split()).lower()


class StormDatasetCollection:
    def __init__(self, list_of_datasets=None):
        self.dataset_type = "StormDatasetCollection"
        self.list_of_datasets = []
        self.locs_dtype = []
        self.dataset_names = []
        if list_of_datasets is not None:
            self.add_storm_dtype()
            for i in range(len(list_of_datasets)):
                self.list_of_datasets.append(list_of_datasets[i])
                assert list_of_datasets.locs.dtype == self.locs_dtype

    def add_storm_dtype(self):
        self.locs_dtype = storm_data_dtype

    def load_h5(self, file_path, name):
        with h5py.File(file_path, "r") as locs_file:
            data = locs_file["molecule_set_data"]["datatable"][...]
            try:
                pixelsize = (
                    locs_file["molecule_set_data"]["xy_pixel_size_um"][...] * 1e3
                )  # to µm to nm
            except KeyError:
                pixelsize = locs_file["molecule_set_data"]["pixel_size_um"][...] * 1e3
        try:
            frames = data["FRAME_NUMBER"]
        except (KeyError, ValueError):
            frames = np.ones(len(data["X_POS_PIXELS"]))
        try:
            z_pos_px = data["Z_POS_PIXELS"]
            zdim = True
        except (KeyError, ValueError):
            z_pos_px = np.ones(len(data["X_POS_PIXELS"]))
            zdim = False
        locs = np.rec.array(
            (
                frames,
                data["X_POS_PIXELS"],
                data["Y_POS_PIXELS"],
                z_pos_px,
                np.ones(len(data["X_POS_PIXELS"])),
                np.ones(len(data["X_POS_PIXELS"])),
                np.ones(len(data["X_POS_PIXELS"])),
                data["PHOTONS"],
            ),
            dtype=storm_data_dtype,
        )
        unique_channels = np.unique(data["CHANNEL"])
        num_channel = len(unique_channels)
        list_of_datasets = []
        for i in range(num_channel):
            filename_pluschannel = name + f" Channel {i + 1}"
            self.dataset_names.append(filename_pluschannel)
            locs_in_ch = locs[data["CHANNEL"] == unique_channels[i]]
            list_of_datasets.append(
                StormDataClass(
                    locs=locs_in_ch,
                    name=filename_pluschannel,
                    pixelsize_nm=pixelsize,
                    zdim_present=zdim,
                    sigma_present=False,
                    photon_count_present=True,
                )
            )
        self.list_of_datasets = list_of_datasets
        return self


class StormDataClass(LocalizationDataBaseClass):
    """An Object which contains STORM/PALM localization data,
    Subclass of LocalizationDataBaseClass"""

    #: STORM records hold camera pixels, not nanometres.  Declaring that here is
    #: the whole of what this subclass has to say about coordinates; the
    #: conversion, caching and filtering are the table's job.
    POSITION_COLUMNS = {
        "x": "x_pos_pixels",
        "y": "y_pos_pixels",
        "z": "z_pos_pixels",
    }

    def __init__(
        self,
        locs=None,
        name=None,
        pixelsize_nm=None,
        zdim_present=False,
        sigma_present=False,
        photon_count_present=False,
    ):
        # Set before super().__init__, which assigns locs_all and therefore
        # builds the table with this scale.
        self._pixelsize_nm = pixelsize_nm
        super().__init__(locs, name, zdim_present)
        self.dataset_type = "StormDataClass(LocalizationDataBaseClass)"
        self.add_storm_dtype()
        if locs is None:
            self.pixelsize_nm = None
            self.sigma_present = None
            self.photon_count_present = None
            self.uncertainty_defined = None

        else:
            assert (
                locs.dtype == self.locs_dtype
            ), f"locs should be numpy rec array of format: {self.locs_dtype}"
            if pixelsize_nm is None:
                self.pixelsize_nm = 100.0
            else:
                self.pixelsize_nm = pixelsize_nm

            self.sigma_present = sigma_present
            self.photon_count_present = photon_count_present
            self.uncertainty_defined = sigma_present or photon_count_present

    def position_scale_nm(self):
        """STORM stores camera pixels; the pixel size converts them.

        ``.item()`` rather than ``float()``: some importers read the pixel size
        straight out of an HDF5 attribute, which arrives as a 0-d or 1-element
        array.  It raises, rather than silently picking one, if a file ever
        supplies more than one value.
        """
        if self._pixelsize_nm is None:
            return 1.0
        return float(np.asarray(self._pixelsize_nm).item())

    @property
    def pixelsize_nm(self):
        return self._pixelsize_nm

    @pixelsize_nm.setter
    def pixelsize_nm(self, value):
        self._pixelsize_nm = value
        # The nanometre columns are derived from it, so the table has to be told
        # rather than left holding coordinates computed with the old value.
        if self.table is not None:
            self.table.position_scale_nm = self.position_scale_nm()

    def add_storm_dtype(self):
        self.locs_dtype = storm_data_dtype

    def load_ns(self, dataset):
        tmp_name = dataset.attrs["name"]
        tmp_zdim_present = dataset.attrs["zdim_present"]
        tmp_photon_count_present = dataset.attrs["photon_count_present"]
        tmp_pixelsize_nm = dataset.attrs["pixelsize_nm"]
        tmp_sigma_present = dataset.attrs["sigma_present"]

        return StormDataClass(
            locs=np.rec.array(dataset[...]),
            name=tmp_name,
            zdim_present=tmp_zdim_present,
            photon_count_present=tmp_photon_count_present,
            pixelsize_nm=tmp_pixelsize_nm,
            sigma_present=tmp_sigma_present,
        )

    def check_if_imported_data_isnm_or_px(self, metadata_provider=None):
        provider = metadata_provider or MetadataProvider()
        data_in_nm = provider.ask_yes_no(DATA_IN_NM, "Is data saved in nm?")
        if data_in_nm is not None:
            assert isinstance(data_in_nm, bool)
        return data_in_nm

    def check_if_metadata_is_complete(self, metadata, metadata_provider=None):
        provider = metadata_provider or MetadataProvider()
        if "name" not in metadata:
            metadata["name"] = "Untitled"
        if "zdim_present" not in metadata:
            zdim_present = provider.ask_yes_no(ZDIM_PRESENT, "Is zdim present?")
            if zdim_present is not None:
                assert isinstance(zdim_present, bool)
                metadata["zdim_present"] = zdim_present
        if "pixelsize_nm" not in metadata:
            pixelsize_nm = provider.ask_text(
                PIXEL_SIZE_NM, "Pixelsize in nm as integer:"
            )
            if pixelsize_nm is not None:
                metadata["pixelsize_nm"] = int(pixelsize_nm)
        if "sigma_present" not in metadata:
            sigma_present = provider.ask_yes_no(
                SIGMA_PRESENT, "Are uncertainty values present?"
            )
            if sigma_present is not None:
                assert isinstance(sigma_present, bool)
                metadata["sigma_present"] = sigma_present
        if "photon_count_present" not in metadata:
            photon_count_present = provider.ask_yes_no(
                PHOTON_COUNT_PRESENT, "Are photon count values present?"
            )
            if photon_count_present is not None:
                assert isinstance(photon_count_present, bool)
                metadata["photon_count_present"] = photon_count_present
        return metadata

    def import_recognized_data(self, data, metadata=None, metadata_provider=None):
        data = np.rec.array(data, metadata["dataset_class_dtype"])
        metadata = StormDataClass().check_if_metadata_is_complete(
            metadata, metadata_provider
        )
        if StormDataClass().check_if_imported_data_isnm_or_px(metadata_provider):
            data.x_pos_pixels /= metadata["pixelsize_nm"]
            data.y_pos_pixels /= metadata["pixelsize_nm"]
            data.z_pos_pixels /= metadata["pixelsize_nm"]

        return StormDataClass(
            locs=data,
            name=metadata["name"],
            zdim_present=metadata["zdim_present"],
            pixelsize_nm=metadata["pixelsize_nm"],
            sigma_present=metadata["sigma_present"],
            photon_count_present=metadata["photon_count_present"],
        )

    def save_as_npy(self, filename):
        """Write to *filename*.  Choosing it is the caller's job, not a reader's."""
        if not filename:
            raise ValueError("save_as_npy needs a filename")
        metadata = {
            "dataset_class": StormDataClass,
            "name": self.name,
            "zdim_present": self.zdim_present,
            "pixelsize_nm": self.pixelsize_nm,
            "sigma_present": self.sigma_present,
            "photoncount_present": self.photon_count_present,
        }
        np.save(filename + ".npy", [self.locs, metadata])

    def load_npy(self, filename):
        if not filename:
            raise ValueError("load_npy needs a filename")
        data = np.load(filename + ".npy")
        self.locs_all = data[0].copy()
        return data[1]["dataset_class"](
            locs=data[0],
            name=data[1]["name"],
            zdim_present=data[1]["zdim_present"],
            pixelsize_nm=data[1]["pixelsize_nm"],
            sigma_present=data[1]["sigma_present"],
            photon_count_present=data[1]["photon_count_present"],
        )

    def restrict_locs_by_sigma_threshold(
        self, sigma_min_pixels=-np.inf, sigma_max_pixels=np.inf
    ):
        assert self.sigma_present
        try:
            tmp = len(sigma_max_pixels)
        except TypeError:
            tmp = 1

        properties = ["sigma_x_pixels", "sigma_y_pixels"]
        if self.zdim_present:
            properties.append("sigma_z_pixels")

        if tmp == 1:
            for i in range(len(properties)):
                self.bandpass_locs_filter_by_property(
                    properties[i], sigma_min_pixels, sigma_max_pixels
                )
        else:
            for i in range(len(properties)):
                self.bandpass_locs_filter_by_property(
                    properties[i], sigma_min_pixels[i], sigma_max_pixels[i]
                )

    def restrict_locs_by_photon_count(self, min_photon_count):
        # The indices used to be looked up in locs_all and then deleted from
        # locs_active, which only agreed while nothing else was filtered.
        # Intersecting masks over the canonical table cannot disagree.
        self.set_filter_mask(
            self.filter_mask & (self.locs_all.photon_count >= min_photon_count)
        )

    def load_info(self, path):
        """Loads Infos from Picassos .yaml"""
        path_base, path_extension = os.path.splitext(path)
        filename = path_base + ".yaml"
        try:
            with open(filename) as info_file:
                info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
        except FileNotFoundError:
            logging.warning("Could not find metadata file: %s", filename)
            info = []
        return info

    def load_locs(self, path):
        """Loads Picassos .hdf5 files"""
        with h5py.File(path, "r") as locs_file:
            locs = locs_file["locs"][...]
        locs = np.rec.array(
            locs, dtype=locs.dtype
        )  # Convert to rec array with fields as attributes
        info = self.load_info(path)
        return locs, info

    def load_hdf5(self, file_path, name, metadata_provider=None):
        """Wrapper for load_locs and load_infos -> picassos hdf5"""
        provider = metadata_provider or MetadataProvider()
        locs, info = self.load_locs(file_path)
        if hasattr(locs, "pixelsize"):
            pixelsize = locs.pixelsize_nm
        else:
            pixelsize = provider.ask_text(PIXEL_SIZE_NM, "Enter the pixelsize [nm]")
            if pixelsize is None:
                raise PixelSizeIsNecessaryError("Pixelsize is mandatory")
        pixelsize = float(pixelsize)
        if hasattr(locs, "z"):
            locs.z = locs.z / pixelsize
            zdim = True
        else:
            locs.z = np.ones(len(locs.x))
            zdim = False

        sigma_present = False
        photon_count_present = False

        if hasattr(locs, "lpx") and hasattr(locs, "lpy"):
            uncertainty_x_pixels = locs.lpx
            uncertainty_y_pixels = locs.lpy
            sigma_present = True
        else:
            uncertainty_x_pixels = np.ones(len(locs.x))
            uncertainty_y_pixels = np.ones(len(locs.x))
        if hasattr(locs, "lpz") and zdim:
            uncertainty_z_pixels = locs.lpz
        else:
            uncertainty_z_pixels = 2 * np.sqrt(locs.lpx**2 + locs.lpy**2)

        if hasattr(locs, "photons"):
            intensity_photons = locs.photons
            photon_count_present = True
        else:
            intensity_photons = np.ones(len(locs.x))
        locs = np.rec.array(
            (
                locs.frame,
                locs.x,
                locs.y,
                locs.z,
                uncertainty_x_pixels,
                uncertainty_y_pixels,
                uncertainty_z_pixels,
                intensity_photons,
            ),
            dtype=storm_data_dtype,
        )
        self.locs_all = locs.copy()
        self.name = name
        self.pixelsize_nm = pixelsize
        self.zdim_present = zdim
        self.sigma_present = sigma_present
        self.photon_count_present = photon_count_present
        self.uncertainty_defined = self.sigma_present or self.photon_count_present
        return self

    def load_csv(self, file_path, name):
        """Loads Thunderstorm .csv files"""
        sigma_present = False

        with open(file_path) as infile:
            header = [
                _normalized_csv_header(field)
                for field in infile.readline().rstrip("\n").split(",")
            ]
            # atleast_2d so a file holding a single localization still reads as
            # one row of columns rather than as one column of rows.
            data_list = np.atleast_2d(
                np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=float)
            )
        data = dict(zip(header, data_list.T))
        pixelsize = 1

        locs_pos_x_nm = data.get("x [nm]")
        locs_pos_y_nm = data.get("y [nm]")
        if locs_pos_x_nm is None or locs_pos_y_nm is None:
            # Only x used to be checked and y was read blind, so a file
            # carrying one without the other raised KeyError rather than this.
            raise ImportError("Localisation Position in X or Y not found in header")
        count = len(locs_pos_x_nm)

        locs_pos_z_nm = data.get("z [nm]")
        zdim = locs_pos_z_nm is not None
        if not zdim:
            locs_pos_z_nm = np.ones(count)

        frame_numbers = data.get("frame")
        if frame_numbers is None:
            frame_numbers = np.ones(count)

        # ThunderSTORM writes photon counts *alongside* uncertainties, and this
        # was an elif behind them, so the column was discarded for every file
        # that had both.  Reading it unconditionally does not change what gets
        # rendered -- variable-Gaussian mode still prefers sigma when both are
        # present -- it gives the photon-count filter something to work on.
        intensity_photons = data.get("intensity [photon]")
        photon_count_present = intensity_photons is not None
        if not photon_count_present:
            intensity_photons = np.ones(count)

        uncertainty_xy_nm = data.get("uncertainty_xy [nm]")
        uncertainty_x_nm = data.get("uncertainty_x [nm]")
        uncertainty_y_nm = data.get("uncertainty_y [nm]")
        uncertainty_z_nm = data.get("uncertainty_z [nm]")

        if uncertainty_xy_nm is not None:
            uncertainty_x_nm = uncertainty_xy_nm
            uncertainty_y_nm = uncertainty_xy_nm
            sigma_present = True
        elif uncertainty_x_nm is not None:
            if uncertainty_y_nm is None:
                # Read blind before, so x-without-y raised KeyError.  A single
                # measured lateral uncertainty describes both axes.
                uncertainty_y_nm = uncertainty_x_nm
            sigma_present = True
        else:
            uncertainty_x_nm = np.ones(count)
            uncertainty_y_nm = np.ones(count)

        if not (zdim and sigma_present):
            # A flat dataset has no z extent to describe, and with no lateral
            # uncertainty there is nothing to derive a z uncertainty from.
            uncertainty_z_nm = np.ones(count)
        elif uncertainty_z_nm is None:
            # Was an unguarded lookup in the uncertainty_xy branch: a 3D file
            # with lateral uncertainties but no z column raised KeyError.
            uncertainty_z_nm = 2 * np.sqrt(uncertainty_x_nm**2 + uncertainty_y_nm**2)

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
            dtype=storm_data_dtype,
        )
        self.name = name
        self.sigma_present = sigma_present
        self.photon_count_present = photon_count_present
        self.uncertainty_defined = sigma_present or photon_count_present
        self.locs_all = locs.copy()
        self.pixelsize_nm = pixelsize
        self.zdim_present = zdim
        return [self]

    def load_smlm(self, file_path, name, metadata_provider=None):
        provider = metadata_provider or MetadataProvider()
        photon_count_present = True
        sigma_present = False

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        dtype2struct = {"uint8": "B", "uint32": "I", "float64": "d", "float32": "f"}
        dtype2length = {"uint8": 1, "uint32": 4, "float64": 8, "float32": 4}

        """Loads SMLM Files"""

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
                                    unpacked_data = unpack(table_file[i : i + rowLen])
                                    for j, h in enumerate(headers):
                                        tableDict[h].append(unpacked_data[j])
                                tableDict = {
                                    h: np.array(tableDict[h])
                                    for i, h in enumerate(headers)
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
                            "format mode {} not supported yet".format(
                                file_format["mode"]
                            )
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
        except KeyError:
            pixelsize = provider.ask_text(PIXEL_SIZE_NM, "Enter the pixelsize [nm]")
            if pixelsize is None:
                raise PixelSizeIsNecessaryError("Pixelsize is mandatory")
        pixelsize = float(pixelsize)
        if "intensity_photon_" not in prop:  # Default missing photons to 1k
            photon_count_present = False
            prop["intensity_photon_"] = 1000 * np.ones(len(prop["x"]))
        if "z" in prop:
            zdim = True
        else:
            prop["z"] = np.ones(len(prop["x"]))
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
            dtype=storm_data_dtype,
        )
        self.name = name
        self.sigma_present = sigma_present
        self.photon_count_present = photon_count_present
        self.uncertainty_defined = sigma_present or photon_count_present
        self.locs_all = locs.copy()
        self.pixelsize_nm = pixelsize
        self.zdim_present = zdim
        return [self]
