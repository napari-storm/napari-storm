import numpy as np


def file_and_data_recognition(filepath):
    file_ending = filepath.split('.')[-1]

    if file_ending == "npy":
        data, metadata = load_npy(filepath)
    elif file_ending == "txt":
        data = np.loadtxt(filepath)
        metadata = None
    elif file_ending == "csv":
        data = np.l


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
