
from .CustomErrors import *

from .localization_dataset_types.storm_class import *
from .localization_dataset_types.Minflux_class import *
from .pyqt.FiletyperecognitionDialog import *

def class_string_to_class(class_name):
    if class_name == "LocalizationDataBaseClass":
        return LocalizationDataBaseClass
    elif class_name == "MinfluxDataAIIterationClass":
        return MinfluxDataAIIterationClass
    elif class_name == "StormDataClass":
        return StormDataClass
    elif class_name == "MinfluxDataBaseClass":
        return MinfluxDataBaseClass


def choose_datasource_in_file(keys):
    def accepted(index):
        try:
            return keys[index]
        except TypeError:
            return keys

    def abort():
        raise FileImportAbortedError("User aborted import")

    window = MainWindowWrapper("where is your data saved?", keys, accepted, abort)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    if window.exec_() == QDialog.Accepted:
        return window.tobereturned


def choose_dataset_class():
    def accept(index):
        return list_of_dataset_classes[index]

    def abort():
        raise FileImportAbortedError("User aborted import")
    window = MainWindowWrapper("What type of data are you importing?", list_of_dataset_classes, accept, abort)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    if window.exec_() == QDialog.Accepted:
        return window.tobereturned


def assign_data_colums_to_datatype(raw_data, dataset_class_dtype, keys):
    tmp_data = []

    def accept(index):
        if index >= len(keys):
            tmp_data.append(np.ones(len(raw_data[keys[0]])))
        else:
            tmp_data.append(raw_data[keys[index]])

    def abort():
        raise FileImportAbortedError("User aborted import")

    for i in range(len(dataset_class_dtype)):
        if isinstance(keys, tuple):
            options = keys + ("not present",)
        elif isinstance(keys, list):
            options = keys + ["not present"]
        else:
            options = keys
        window = MainWindowWrapper(dataset_class_dtype[i][0], options, accept, abort)
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        if window.exec_() == QDialog.Accepted:
            tmp_data.append(window.tobereturned)
    return tmp_data


def zdim_present():
    def accept(index):
        return index  # 0 means no, 1 is yes

    def abort():
        raise FileImportAbortedError("User aborted import")

    window = MainWindowWrapper("Zdim Present?", ["No", "Yes"], accept, abort)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    if window.exec_() == QDialog.Accepted:
        return window.tobereturned


def user_preparation_for_import(dataset_class, raw_data, keys, filename):
    """Interact with user to find out metadata of dataset"""
    dataset_class_dtype = get_dtype_of_dataset_class(dataset_class)
    dataset_class = class_string_to_class(dataset_class)
    ordered_data = assign_data_colums_to_datatype(dataset_class_dtype=dataset_class_dtype, raw_data=raw_data,
                                                  keys=keys)
    ordered_data = [i for i in ordered_data if i is not None]
    metadata = {"dataset_class": dataset_class, "dataset_class_dtype": dataset_class_dtype, "name": filename,
                "zdim_present": zdim_present()}
    return ordered_data, metadata

def allkeys_h5py(obj):
    """Recursively find all keys in an h5py.Group."""
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys_h5py(value)
            else:
                keys = keys + (value.name,)
    return keys


def file_and_data_recognition(filepath=None):
    """Find out filetype, extract headers and data, return data and metadata obtained by file or user"""
    if filepath is None or isinstance(filepath, bool):
        filepath = QFileDialog.getOpenFileName()[0]
    file_ending = filepath.split('.')[-1]
    filename = filepath.split("/")[-1]

    if file_ending == "npy":
        raw_data, metadata = load_npy(filepath)
        if metadata is None:
            keys = raw_data.dtype.names
            dataset_class = choose_dataset_class()
            ordered_data, metadata = user_preparation_for_import(dataset_class, raw_data, keys, filename)
            return metadata["dataset_class"].import_recognized_data(None, ordered_data, metadata)
        else:
            dataset_class = metadata["dataset_class"]
            return metadata["dataset_class"].import_recognized_data(None, raw_data, metadata)

    elif file_ending == "txt":
        raw_data = load_txt(filepath)
        colums = []
        for i in range(raw_data.shape[0]):
            colums.append(f"colum {i}")
        dataset_class = choose_dataset_class()
        ordered_data, metadata = user_preparation_for_import(dataset_class, raw_data, colums, filename)
        return metadata["dataset_class"].import_recognized_data(None, ordered_data, metadata)

    elif file_ending == "csv":
        raw_data, header = load_csv(filepath)
        dataset_class = choose_dataset_class()
        ordered_data, metadata = user_preparation_for_import(dataset_class, raw_data, header, filename)
        return metadata["dataset_class"].import_recognized_data(None, ordered_data, metadata)

    elif file_ending == "hdf5" or file_ending == "h5" or file_ending == "hdf":
        with h5py.File(filepath, "r") as locs_file:
            keys = allkeys_h5py(locs_file)
            key = choose_datasource_in_file(keys)

            raw_data = locs_file[key][...]
            dataset_class = choose_dataset_class()
            ordered_data, metadata = user_preparation_for_import(dataset_class, raw_data, locs_file[key].dtype.names,
                                                                 filename)
        return metadata["dataset_class"].import_recognized_data(None, ordered_data, metadata)

    raise FileImportAbortedError('Unknown data file extension, try file recognition import')


def load_txt(filepath):
    raw_data = np.loadtxt(filepath)
    return raw_data


def load_npy(filepath):
    file_content = np.load(filepath)
    try:
        if type(file_content[0]) == np.recarray:
            data = file_content[0]
        else:
            data = file_content
        if isinstance(file_content[1], dict):
            metadata = file_content[1]
        else:
            metadata = None
    except IndexError:
        data = file_content
        metadata = None
    return data, metadata


def load_csv(file_path):
    data = {}

    with open(file_path) as infile:
        header = infile.readline()
        header = header.replace("\n", "")
        header = header.split(",")
        data_list = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=float)
    for i in range(len(header)):
        data[header[i]] = data_list[:, i]
    return data, header
