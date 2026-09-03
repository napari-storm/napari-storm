"""An HDF5 file is routed by what is inside it, not by its extension.

Picasso localization tables and daxview molecule sets are unrelated formats
that both ship as .h5 and as .hdf5.  Dispatching on the extension sent each to
the other's reader; standing in for that with "is there a .yaml next to it?"
answered a question about the directory rather than about the file.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from napari_storm.core import PIXEL_SIZE_NM, StaticMetadataProvider
from napari_storm.CustomErrors import (
    PixelSizeIsNecessaryError,
    UnknownFileLayoutError,
)
from napari_storm.FileToLocalizationDataInterface import (
    FileToLocalizationDataInterface,
)
from napari_storm.localization_dataset_types.storm_class import StormDataClass


def _interface(answers=None):
    return FileToLocalizationDataInterface(
        SimpleNamespace(localization_datasets=[]),
        metadata_provider=StaticMetadataProvider(answers or {}),
    )


def _picasso_locs(n=4):
    locs = np.zeros(
        n,
        dtype=[
            ("frame", "i4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ],
    )
    locs["x"] = np.linspace(1, 4, n)
    locs["y"] = np.linspace(4, 1, n)
    locs["photons"] = 100
    locs["lpx"] = 0.1
    locs["lpy"] = 0.1
    return locs


def _write_picasso(path, yaml_pixelsize=None, embedded_pixelsize=None):
    """A Picasso file: one ``locs`` table, metadata beside it and/or within."""
    with h5py.File(path, "w") as file:
        file.create_dataset("locs", data=_picasso_locs())
        if embedded_pixelsize is not None:
            file.create_dataset(
                "metadata",
                data=json.dumps([{"Frames": 10, "Pixelsize": embedded_pixelsize}]),
            )
    if yaml_pixelsize is not None:
        path.with_suffix(".yaml").write_text(
            yaml.dump(
                {
                    "Width": 128,
                    "Height": 128,
                    "Frames": 10,
                    "Pixelsize": yaml_pixelsize,
                }
            )
        )
    return str(path)


def _write_molecule_set(path, channels=(0, 0, 0, 1, 1, 1)):
    """A daxview file: a ``molecule_set_data`` group, one row per molecule."""
    data = np.zeros(
        len(channels),
        dtype=[
            ("X_POS_PIXELS", "f4"),
            ("Y_POS_PIXELS", "f4"),
            ("Z_POS_PIXELS", "f4"),
            ("PHOTONS", "f4"),
            ("CHANNEL", "i4"),
            ("FRAME_NUMBER", "i4"),
        ],
    )
    data["X_POS_PIXELS"] = np.arange(len(channels))
    data["CHANNEL"] = channels
    with h5py.File(path, "w") as file:
        group = file.create_group("molecule_set_data")
        group.create_dataset("datatable", data=data)
        group.create_dataset("xy_pixel_size_um", data=0.1)
    return str(path)


# ------------------------------------------------------------------ routing


def test_a_picasso_table_named_h5_reaches_the_picasso_reader(tmp_path):
    file_path = _write_picasso(tmp_path / "locs.h5", yaml_pixelsize=130)

    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    assert len(datasets) == 1
    assert isinstance(datasets[0], StormDataClass)
    assert datasets[0].number_of_entries() == 4


def test_a_molecule_set_named_hdf5_reaches_the_molecule_set_reader(tmp_path):
    file_path = _write_molecule_set(tmp_path / "cells.hdf5")

    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    # One dataset per channel, which is what makes this reader the right one.
    assert len(datasets) == 2
    assert [dataset.number_of_entries() for dataset in datasets] == [3, 3]


def test_a_molecule_set_beside_a_stray_yaml_is_still_a_molecule_set(tmp_path):
    """The old sidecar test would have called this a Picasso file."""
    file_path = _write_molecule_set(tmp_path / "cells.hdf5")
    (tmp_path / "cells.yaml").write_text(yaml.dump({"Pixelsize": 130}))

    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    assert len(datasets) == 2


def test_an_hdf5_of_an_unknown_layout_names_what_it_looked_for(tmp_path):
    file_path = tmp_path / "something_else.hdf5"
    with h5py.File(file_path, "w") as file:
        file.create_dataset("mystery", data=np.zeros(3))

    with pytest.raises(UnknownFileLayoutError) as raised:
        _interface().open_known_filetype_and_import_dataset(str(file_path))

    message = str(raised.value)
    assert "locs" in message and "molecule_set_data" in message


def test_the_yaml_may_be_picked_instead_of_the_data_file(tmp_path):
    _write_picasso(tmp_path / "locs.hdf5", yaml_pixelsize=130)

    datasets = _interface().open_known_filetype_and_import_dataset(
        str(tmp_path / "locs.yaml")
    )

    assert len(datasets) == 1
    assert datasets[0].pixelsize_nm == 130


def test_a_yaml_with_no_data_file_beside_it_says_so(tmp_path):
    orphan = tmp_path / "locs.yaml"
    orphan.write_text(yaml.dump({"Pixelsize": 130}))

    with pytest.raises(FileNotFoundError, match="metadata"):
        _interface().open_known_filetype_and_import_dataset(str(orphan))


# ------------------------------------------------------------- pixel size


def test_the_pixel_size_is_read_from_the_yaml_rather_than_asked_for(tmp_path):
    file_path = _write_picasso(tmp_path / "locs.hdf5", yaml_pixelsize=130)

    # An empty provider answers nothing, so any question at all is fatal here.
    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    assert datasets[0].pixelsize_nm == 130


def test_the_pixel_size_is_read_from_the_embedded_metadata_without_a_yaml(tmp_path):
    """Current Picasso keeps the same documents inside the .hdf5."""
    file_path = _write_picasso(tmp_path / "locs.hdf5", embedded_pixelsize=117)

    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    assert datasets[0].pixelsize_nm == 117


def test_the_yaml_wins_over_the_embedded_copy(tmp_path):
    file_path = _write_picasso(
        tmp_path / "locs.hdf5", yaml_pixelsize=130, embedded_pixelsize=117
    )

    datasets = _interface().open_known_filetype_and_import_dataset(file_path)

    assert datasets[0].pixelsize_nm == 130


def test_a_file_that_records_no_pixel_size_still_asks(tmp_path):
    file_path = _write_picasso(tmp_path / "locs.hdf5")

    datasets = _interface(
        {PIXEL_SIZE_NM: "100"}
    ).open_known_filetype_and_import_dataset(file_path)

    assert datasets[0].pixelsize_nm == 100


def test_a_file_with_no_pixel_size_and_no_answer_is_refused(tmp_path):
    file_path = _write_picasso(tmp_path / "locs.hdf5")

    with pytest.raises(PixelSizeIsNecessaryError):
        _interface().open_known_filetype_and_import_dataset(file_path)


def test_the_last_document_to_state_a_pixel_size_is_the_one_that_applies(tmp_path):
    """Picasso appends one document per processing step."""
    file_path = tmp_path / "locs.hdf5"
    with h5py.File(file_path, "w") as file:
        file.create_dataset("locs", data=_picasso_locs())
    with open(tmp_path / "locs.yaml", "w") as sidecar:
        yaml.dump_all(
            [
                {"Frames": 10, "Pixelsize": 160},
                {"Generated by": "Picasso Localize", "Pixelsize": 130},
            ],
            sidecar,
        )

    datasets = _interface().open_known_filetype_and_import_dataset(str(file_path))

    assert datasets[0].pixelsize_nm == 130
