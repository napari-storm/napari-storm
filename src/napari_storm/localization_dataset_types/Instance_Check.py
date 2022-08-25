from .base_class import *
from .storm_class import *
from .Minflux_class import *


def is_minflux_dataset(dataset):
    if isinstance(dataset, MinfluxDataBaseClass):
        return True
    elif isinstance(dataset, MinfluxDataAIClass):
        return True
    elif isinstance(dataset, MinfluxDataAIIterationClass):
        return True
    else:
        return False


def is_storm_dataset(dataset):
    if isinstance(dataset, StormDatasetCollection):
        return True
    elif isinstance(dataset, StormDataClass):
        return True
    else:
        return False


def is_localization_dataset(dataset):
    if isinstance(dataset, LocalizationDataBaseClass):
        return True
    elif is_minflux_dataset(dataset):
        return True
    elif is_storm_dataset(dataset):
        return True
    else:
        return False


def fulfills_storm_dataset_requirements(dataset):
    if (hasattr(dataset, "frame_number") and
            hasattr(dataset, "x_pos_pixels") and
            hasattr(dataset, "y_pos_pixels")):
        return True
    else:
        return False


def fulfills_localization_dataset_requirements(dataset):
    if (hasattr(dataset, "x_pos_nm") and
            hasattr(dataset, "y_pos_nm")):
        return True
    else:
        return False


def fulfills_minflux_dataset_requirements(dataset):
    if (fulfills_localization_dataset_requirements(dataset) and
            hasattr(dataset, "trace_id")):
        return True
    else:
        return False
