from .._data_constants import STORM_DATA_DTYPE

lm_base_data_dtype = [("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")]

storm_data_dtype = STORM_DATA_DTYPE

minflux_base_dtype = [
    ("x_pos_nm", "f4"),
    ("y_pos_nm", "f4"),
    ("z_pos_nm", "f4"),
    ("trace_id", "i4"),
]

minflux_AI_data_dtype = [
    ("x_pos_nm", "f4"),
    ("y_pos_nm", "f4"),
    ("z_pos_nm", "f4"),
    ("eco", "i4"),
    ("ecc", "i4"),
    ("efo", "f4"),
    ("efc", "f4"),
    ("sta", "i4"),
    ("cfr", "f4"),
    ("dcr", "f4"),
    ("gvy", "f4"),
    ("gvx", "f4"),
    ("eoy", "f4"),
    ("eox", "f4"),
    ("dmz", "f4"),
    ("lcx", "f4"),
    ("lcy", "f4"),
    ("lcz", "f4"),
    ("fbg", "f4"),
    ("tic", "i4"),
    ("lnc_x", "f4"),
    ("lnc_y", "f4"),
    ("lnc_z", "f4"),
    ("ext_x", "f4"),
    ("ext_y", "f4"),
    ("ext_z", "f4"),
    ("time_s", "f4"),
    ("activation", "i4"),
    ("trace_id", "i4"),
]

# Imspector >= 24.10 ("reader version 2") exports a flat, one-row-per-iteration
# table.  It drops the fields v1 carried per iteration -- act, tic, gvx/gvy,
# eox/eoy, dmz, lcx/lcy/lcz, ext -- and adds the sequence identifiers gri/thi/
# sqi plus a scalar itr, so it needs its own dtype rather than a subset of
# minflux_AI_data_dtype.
minflux_v2_data_dtype = [
    ("x_pos_nm", "f4"),
    ("y_pos_nm", "f4"),
    ("z_pos_nm", "f4"),
    ("eco", "i4"),
    ("ecc", "i4"),
    ("efo", "f4"),
    ("efc", "f4"),
    ("sta", "i4"),
    ("cfr", "f4"),
    ("dcr", "f4"),
    ("fbg", "f4"),
    ("lnc_x", "f4"),
    ("lnc_y", "f4"),
    ("lnc_z", "f4"),
    ("gri", "i4"),
    ("thi", "i4"),
    ("sqi", "i4"),
    ("itr", "i4"),
    ("time_s", "f4"),
    ("trace_id", "i4"),
]

list_of_dataset_classes = [
    "LocalizationDataBaseClass",
    "StormDataClass",
    "MinfluxDataBaseClass",
    "MinfluxDataAIIterationClass",
    "MinfluxDataV2Class",
]
list_of_dataset_classes_dtypes = [
    lm_base_data_dtype,
    storm_data_dtype,
    minflux_base_dtype,
    minflux_AI_data_dtype,
    minflux_v2_data_dtype,
]
names_of_dataset_classes_dtypes = [
    "lm_base_data_dtype",
    "storm_data_dtype",
    "minflux_base_dtype",
    "minflux_AI_data_dtype",
    "minflux_v2_data_dtype",
]


def get_dtype_of_dataset_class(dataset_class_str):
    for i in range(len(list_of_dataset_classes)):
        if list_of_dataset_classes[i] == dataset_class_str:
            return list_of_dataset_classes_dtypes[i]
