"""Regression coverage for correctness fixes recovered from charming-kirch."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from napari_storm import ns_constants
from napari_storm.core import PIXEL_SIZE_NM, StaticMetadataProvider
from napari_storm.FileToLocalizationDataInterface import (
    FileToLocalizationDataInterface,
)
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.localization_dataset_types.data_formats import storm_data_dtype
from napari_storm.localization_dataset_types.Minflux_class import (
    MINFLUX_Z_CORRECTION_FACTOR,
)
from napari_storm.localization_dataset_types.storm_class import StormDataClass


def _parent():
    return SimpleNamespace(localization_datasets=[])


def test_dataset_inputs_do_not_alias_canonical_rows():
    locs = np.rec.array(np.zeros(3, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")]))
    dataset = LocalizationDataBaseClass(locs, zdim_present=False)

    locs.x_pos_nm[0] = 10

    assert dataset.locs_all.x_pos_nm[0] == 0
    # The active set is now derived from a mask over the canonical table rather
    # than being an independent copy, so a write to it is refused outright
    # instead of quietly diverging from locs_all.
    with pytest.raises(ValueError):
        dataset.locs_active.x_pos_nm[1] = 20
    assert dataset.locs_all.x_pos_nm[1] == 0


def test_storm_inputs_do_not_alias_canonical_rows():
    locs = np.rec.array(np.zeros(3, dtype=storm_data_dtype))
    dataset = StormDataClass(locs=locs, zdim_present=False)

    locs.x_pos_pixels[0] = 10

    assert dataset.locs_all.x_pos_pixels[0] == 0
    with pytest.raises(ValueError):
        dataset.locs_active.x_pos_pixels[1] = 20
    assert dataset.locs_all.x_pos_pixels[1] == 0


def test_shared_dtype_and_minflux_factor_have_one_value():
    assert ns_constants.LOCS_DTYPE is storm_data_dtype
    assert ns_constants.MINFLUX_Z_CORRECTION_FACTOR == 0.8
    assert MINFLUX_Z_CORRECTION_FACTOR == 0.8


def test_ns_loader_returns_a_list_and_applies_namespace(tmp_path):
    file_path = tmp_path / "saved.ns"
    locs = np.rec.array(np.zeros(3, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")]))
    with h5py.File(file_path, "w") as file:
        stored = file.create_dataset("dataset", data=locs)
        stored.attrs["name"] = "saved"
        stored.attrs["zdim_present"] = False
        stored.attrs["dataset_class"] = "LocalizationDataBaseClass"

    interface = FileToLocalizationDataInterface(_parent())
    first = interface.load_ns(str(file_path))
    second = interface.load_ns(str(file_path))

    assert [dataset.name for dataset in first] == ["saved"]
    assert [dataset.name for dataset in second] == ["saved_2"]


def test_smlm_loader_uses_the_source_basename(monkeypatch):
    calls = []

    def fake_load(_self, file_path, name, metadata_provider=None):
        calls.append((file_path, name, metadata_provider))
        return []

    monkeypatch.setattr(StormDataClass, "load_smlm", fake_load)
    provider = StaticMetadataProvider({PIXEL_SIZE_NM: "100"})
    interface = FileToLocalizationDataInterface(_parent(), metadata_provider=provider)
    interface.load_smlm(r"C:\measurements\sample.smlm")

    # The reader is also handed the provider it needs for a .smlm file with no
    # pixel size, rather than being left to open a dialog of its own.
    assert calls == [(r"C:\measurements\sample.smlm", "sample.smlm", provider)]


def test_missing_picasso_metadata_returns_an_empty_list(tmp_path):
    assert StormDataClass().load_info(str(tmp_path / "missing.hdf5")) == []


def test_a_picasso_hdf5_without_its_yaml_reports_why_it_did_not_open(tmp_path):
    """The missing sidecar must not be swallowed as if it were a cancel."""
    file_path = tmp_path / "locs.hdf5"
    locs = np.rec.array(np.zeros(3, dtype=[("frame", "i4"), ("x", "f4"), ("y", "f4")]))
    with h5py.File(file_path, "w") as file:
        file.create_dataset("locs", data=locs)

    interface = FileToLocalizationDataInterface(_parent())
    with pytest.raises(FileNotFoundError, match="yaml"):
        interface.open_localization_data_file_and_get_dataset(file_path=str(file_path))

    # Nothing was registered on the way out, whichever way the caller reports it.
    assert interface.dataset_names == []
    assert interface.n_datasets == 0
