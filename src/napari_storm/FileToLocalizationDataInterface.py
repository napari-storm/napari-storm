import os.path as _ospath

from .ns_constants import *
from .file_and_data_recognition import *
from .Custom_Import import *


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

    def filetype_recognition(self, file_path):
        return file_and_data_recognition(file_path)

    def open_localization_data_file_and_get_dataset(self, file_path=None, file_type_recognizer=False,
                                                    custom_import=False):
        """Determine which file type is being opened, and call the
        corresponding importer function"""
        if not file_path:
            file_path = QFileDialog.getOpenFileName()[0]
        try:
            self.n_datasets += 1
            if file_type_recognizer:
                return [file_and_data_recognition(file_path)]
            elif custom_import:
                return [custom_import_function(file_path)]

            else:
                return self.open_known_filetype_and_import_dataset(file_path)

        except FileImportAbortedError:
            self.n_datasets -= 1

    def open_known_filetype_and_import_dataset(self, file_path):
        """Find out dataset type by ending or other clues and try to import the known dataset type directly"""
        filetype = file_path.split('.')[-1]
        if filetype == 'hdf5':
            if _ospath.isfile(file_path[: -(len(filetype))] + 'yaml'):
                return self.load_hdf5(file_path)
            else:
                raise FileNotFoundError('Assuming .hdf5 is a picasso file, the correspoding '
                                        '.yaml file couldn t be found in same directory')
        elif filetype == 'yaml':
            file_path = file_path[: -(len(filetype))] + 'hdf5'
            if _ospath.isfile(file_path):
                return self.load_hdf5(file_path)
        # Thunderstorm csv
        elif filetype == 'csv':
            return self.load_csv(file_path)
        # SMLM File
        elif filetype == 'smlm':
            return self.load_smlm(file_path)
        # h5 -> special type of hdf5
        elif filetype == 'h5':
            return self.load_h5(file_path)
        # MINFLUX files
        elif filetype == 'json':
            return self.load_mfx_json(file_path)
        elif filetype == 'npy':
            return self.load_mfx_npy(file_path)
        elif filetype == 'mfx':
            return self.load_mfx(file_path)
        elif filetype == 'test':
            return self.start_testing()
        raise FileImportAbortedError('Unknown data file extension, try file recognition import')

    def check_namespace(self, name, idx=1):
        """Assure that no other dataset with the same name is already open or otherwise change the name """
        # print("checking the namespace:",name)
        for i in range(self.n_datasets - 1):
            if self.dataset_names[i] == name:
                return self.check_namespace(name + "_" + str(idx + 1), idx + 1)
        return name

    def load_mfx(self, file_path):
        """wrapper to load .mfx files"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [MinfluxDataAIIterationClass().load_mfx(file_path=file_path, name=filename)]

    def load_hdf5(self, file_path):
        """wrapper to load .hdf5 files in the picasso format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return [StormDataClass().load_hdf5(file_path=file_path, name=filename)]

    def load_h5(self, file_path):
        """Loads localizations from .h5 files"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset_collection = StormDatasetCollection().load_h5(file_path=file_path, name=filename)
        return dataset_collection.list_of_datasets

    def load_mfx_json(self, file_path, itr=-1):
        """loads localizations from AIs json format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset = MinfluxDataAIIterationClass().load_single_itr(name=filename, file_path=file_path, itr=itr)
        return [dataset]

    def load_mfx_npy(self, file_path, itr=-1):
        """loads localizations from AIs npy format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        dataset = MinfluxDataAIIterationClass().load_single_itr(name=filename, file_path=file_path, itr=itr)
        return [dataset]

    def load_csv(self, file_path):
        """loads localizations thunderstorms csv format"""
        filename = file_path.split("/")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return StormDataClass().load_csv(file_path=file_path, name=filename)

    def load_smlm(self, file_path):
        """loads localizations from smlm format"""
        filename = file_path.split(".")[-1]
        filename = self.check_namespace(filename)
        self.dataset_names.append(filename)
        return StormDataClass().load_smlm(file_path=file_path, name=filename)

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
        return [StormDataClass(locs=locs, name=filename, pixelsize_nm=pixelsize,
                               zdim_present=zdim,
                               sigma_present=False, photon_count_present=True)]
