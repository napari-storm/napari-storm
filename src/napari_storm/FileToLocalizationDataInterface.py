import os.path as _ospath

import h5py
import numpy as np
from qtpy.QtWidgets import QFileDialog

import napari_storm.localization_dataset_types as dataset_classes

from .background_loading import run_on_main_thread
from .CustomErrors import FileImportAbortedError, ParentError
from .file_and_data_recognition import file_and_data_recognition
from .localization_dataset_types.Custom_Import import custom_import_function
from .localization_dataset_types.Minflux_class import MinfluxDataAIIterationClass
from .localization_dataset_types.minflux_v2 import (
    MinfluxDataV2Class,
    is_v2_file,
    zarr_store_root,
)
from .localization_dataset_types.storm_class import (
    StormDataClass,
    StormDatasetCollection,
)
from .ns_constants import LOCS_DTYPE
from .pyqt.prompts import QtMetadataProvider


class FileToLocalizationDataInterface:
    def __init__(self, parent, n_datasets=0, metadata_provider=None):
        self._parent = parent
        self.n_datasets = n_datasets
        self.dataset_names = []
        # Readers ask this for anything a file does not carry -- a missing pixel
        # size, an unknown dimensionality.  It is supplied here, at the
        # application boundary, so no reader has to know a GUI exists.
        self.metadata_provider = metadata_provider or QtMetadataProvider()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError("Cannot change parent of existing Widget")

    def filetype_recognition(self, file_path):
        return file_and_data_recognition(file_path)

    @staticmethod
    def choose_file_path():
        """Ask for a file to open.  Always runs on the GUI thread."""
        return run_on_main_thread(lambda: QFileDialog.getOpenFileName()[0])

    def open_localization_data_file_and_get_dataset(
        self,
        file_path=None,
        file_type_recognizer=False,
        custom_import=False,
        handle=None,
    ):
        """Determine which file type is being opened, and call the
        corresponding importer function.

        *handle* is an optional :class:`~napari_storm.background_loading.LoadHandle`.
        When given, this may be running on a worker thread: progress is reported
        and cancellation is honoured at the checkpoints below.
        """
        if file_path is None:
            file_path = self.choose_file_path()
        # Cancel is a normal, side-effect-free outcome.  In particular, do not
        # increment counters or let the empty path reach the importer.
        if not file_path:
            return None

        if handle is not None:
            handle.checkpoint(f"Reading {_ospath.basename(file_path)}…")

        previous_names = list(self.dataset_names)
        try:
            if file_type_recognizer:
                datasets = [
                    file_and_data_recognition(
                        file_path, metadata_provider=self.metadata_provider
                    )
                ]
            elif custom_import:
                datasets = [custom_import_function(file_path)]
            else:
                datasets = self.open_known_filetype_and_import_dataset(file_path)

        except (FileImportAbortedError, FileNotFoundError):
            self.dataset_names = previous_names
            self.n_datasets = len(previous_names)
            return None
        except Exception:
            # Importers currently register names while loading.  Roll that
            # bookkeeping back if an importer fails part-way through, while
            # still surfacing unexpected errors to the caller.
            self.dataset_names = previous_names
            self.n_datasets = len(previous_names)
            raise

        if not datasets or any(dataset is None for dataset in datasets):
            self.dataset_names = previous_names
            self.n_datasets = len(previous_names)
            return None

        if handle is not None:
            # Last point at which a cancel costs nothing: past here the caller
            # starts building layers on the GUI thread.
            handle.checkpoint(f"Preparing {len(datasets)} dataset(s)…")

        self.sync_dataset_entries([*self.parent.localization_datasets, *datasets])
        return datasets

    def open_known_filetype_and_import_dataset(self, file_path):
        """Find out dataset type by ending or other clues and try to import the known dataset type directly"""
        # A Zarr MINFLUX dataset is a directory, not a file, so it has no
        # extension to dispatch on -- and a plain file dialog cannot select a
        # folder, so anything *inside* the store routes there too.
        if zarr_store_root(file_path) is not None:
            return self.load_mfx_v2(file_path)
        filetype = file_path.split(".")[-1]
        if filetype == "hdf5":
            if _ospath.isfile(file_path[: -(len(filetype))] + "yaml"):
                return self.load_hdf5(file_path)
            else:
                raise FileNotFoundError(
                    "Assuming .hdf5 is a picasso file, the correspoding "
                    ".yaml file couldn t be found in same directory"
                )
        elif filetype == "yaml":
            file_path = file_path[: -(len(filetype))] + "hdf5"
            if _ospath.isfile(file_path):
                return self.load_hdf5(file_path)
        # Thunderstorm csv
        elif filetype == "csv":
            return self.load_csv(file_path)
        # SMLM File
        elif filetype == "smlm":
            return self.load_smlm(file_path)
        # h5 -> special type of hdf5
        elif filetype == "h5":
            return self.load_h5(file_path)
        # MINFLUX files
        elif filetype == "json":
            return self.load_mfx_json(file_path)
        elif filetype == "npy":
            return self.load_mfx_npy(file_path)
        elif filetype == "mfx":
            return self.load_mfx(file_path)
        # Imspector >= 24.10 also writes MINFLUX data as Matlab and as
        # pyMINFLUX's own .pmx; neither has an older-layout counterpart, so
        # they route straight to the new reader.
        elif filetype in ("mat", "pmx"):
            return self.load_mfx_v2(file_path)
        elif filetype == "ns":
            return self.load_ns(file_path)
        elif filetype in ["tif", "tiff", "dat", "raw"]:
            raise FileImportAbortedError(
                "Raw image stacks are not supported: napari-storm renders "
                "localization tables rather than fitting raw movies. "
                "Localize the movie first (e.g. with Picasso or ThunderSTORM) "
                "and import the resulting localization file. A TIFF can still "
                "be overlaid as a reference image from the Data Controls tab."
            )
        elif filetype == "test":
            return self.start_testing()
        raise FileImportAbortedError(
            "Unknown data file extension, try file recognition import"
        )

    def check_namespace(self, name, idx=1):
        """Return a source name that cannot collide with an open dataset."""
        existing_names = {
            *self.dataset_names,
            *(dataset.name for dataset in self.parent.localization_datasets),
        }

        def conflicts(candidate):
            return any(
                existing == candidate or existing.startswith(candidate + " Channel ")
                for existing in existing_names
            )

        candidate = name
        suffix = max(idx + 1, 2)
        while conflicts(candidate):
            candidate = f"{name}_{suffix}"
            suffix += 1
        return candidate

    def sync_dataset_entries(self, datasets):
        """Synchronize namespace bookkeeping with datasets currently loaded."""
        self.dataset_names = [dataset.name for dataset in datasets]
        self.n_datasets = len(self.dataset_names)

    def clear_entries(self):
        self.dataset_names = []
        self.n_datasets = 0

    def load_ns(self, file_path):
        with h5py.File(file_path, "r") as f:
            class_ = getattr(dataset_classes, f["dataset"].attrs["dataset_class"])
            dataset = class_.load_ns(None, f["dataset"])
        dataset.name = self.check_namespace(dataset.name)
        self.dataset_names.append(dataset.name)
        return [dataset]

    def load_mfx(self, file_path):
        """wrapper to load .mfx files"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [
            MinfluxDataAIIterationClass().load_mfx(file_path=file_path, name=filename)
        ]

    def load_hdf5(self, file_path):
        """wrapper to load .hdf5 files in the picasso format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [
            StormDataClass().load_hdf5(
                file_path=file_path,
                name=filename,
                metadata_provider=self.metadata_provider,
            )
        ]

    def load_h5(self, file_path):
        """Loads localizations from .h5 files"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset_collection = StormDatasetCollection().load_h5(
            file_path=file_path, name=filename
        )
        return dataset_collection.list_of_datasets

    def load_mfx_v2(self, file_path, itr=-1):
        """loads localizations from the Imspector >= 24.10 MINFLUX layout"""
        # Name a Zarr dataset after its store, not after whichever file inside
        # it the user had to click to select the thing.
        root = zarr_store_root(file_path)
        source = str(root) if root is not None else file_path
        filename = _ospath.basename(source.rstrip("/\\").replace("\\", "/"))
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [MinfluxDataV2Class().load(file_path=file_path, name=filename, itr=itr)]

    def load_mfx_json(self, file_path, itr=-1):
        """loads localizations from AIs json format, in either layout"""
        if is_v2_file(file_path):
            return self.load_mfx_v2(file_path, itr=itr)
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset = MinfluxDataAIIterationClass().load_single_itr(
            name=filename, file_path=file_path, itr=itr
        )
        return [dataset]

    def load_mfx_npy(self, file_path, itr=-1):
        """loads localizations from AIs npy format, in either layout

        Which layout is decided from the file's header alone, so routing a
        multi-gigabyte export costs no more than opening it.
        """
        if is_v2_file(file_path):
            return self.load_mfx_v2(file_path, itr=itr)
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset = MinfluxDataAIIterationClass().load_single_itr(
            name=filename, file_path=file_path, itr=itr
        )
        return [dataset]

    def load_csv(self, file_path):
        """loads localizations thunderstorms csv format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return StormDataClass().load_csv(file_path=file_path, name=filename)

    def load_smlm(self, file_path):
        """loads localizations from smlm format"""
        filename = _ospath.basename(file_path.replace("\\", "/"))
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return StormDataClass().load_smlm(
            file_path=file_path,
            name=filename,
            metadata_provider=self.metadata_provider,
        )

    def start_testing(self):
        n = 6

        zdim = True
        locs = np.rec.array(
            (
                np.repeat(np.arange(1, n + 1), 2),
                np.repeat(np.arange(1, n + 1), 2),
                np.arange(1, 2 * n + 1),
                np.repeat(np.arange(0, 2), n),
                np.ones(2 * n),
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
        return [
            StormDataClass(
                locs=locs,
                name=filename,
                pixelsize_nm=pixelsize,
                zdim_present=zdim,
                sigma_present=False,
                photon_count_present=True,
            )
        ]
