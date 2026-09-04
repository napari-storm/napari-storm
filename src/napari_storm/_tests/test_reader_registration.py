"""What napari is told this plugin can open, and what it can actually open.

These have to agree in both directions. A format the dispatcher handles but
the manifest omits cannot be reached from File -> Open or by dropping a file
on the canvas, however well the dock widget's own import button handles it --
which is what had happened to .mat, .pmx, .mfx and the plugin's own .ns.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

import napari_storm
from napari_storm._reader import napari_get_reader
from napari_storm.CustomErrors import FileImportAbortedError
from napari_storm.FileToLocalizationDataInterface import (
    FileToLocalizationDataInterface,
)
from napari_storm.ns_constants import list_of_recognized_file_formats

MANIFEST = Path(napari_storm.__file__).parent / "napari.yaml"

#: Not a file format: the dispatcher's own hook for test mode, which has no
#: file behind it to probe with.
_SYNTHETIC = {"test"}


def _manifest_readers():
    with open(MANIFEST) as handle:
        manifest = yaml.safe_load(handle)
    return manifest["contributions"]["readers"][0]


def _manifest_suffixes():
    return {
        pattern.removeprefix("*.")
        for pattern in _manifest_readers()["filename_patterns"]
    }


# ------------------------------------------------------- manifest and code


def test_the_manifest_offers_exactly_what_the_dispatcher_dispatches_on():
    """The drift this guards against is silent: nothing fails, the format is
    simply absent from every menu napari builds."""
    assert _manifest_suffixes() == set(list_of_recognized_file_formats)


@pytest.mark.parametrize(
    "suffix", sorted(set(list_of_recognized_file_formats) - _SYNTHETIC)
)
def test_every_advertised_format_reaches_a_reader(suffix, tmp_path):
    """An empty file of each type: it cannot load, but it must get past
    dispatch to a reader that has an opinion about its contents."""
    path = tmp_path / f"probe.{suffix}"
    path.write_bytes(b"")

    interface = FileToLocalizationDataInterface(
        parent=type("P", (), {"localization_datasets": []})()
    )
    with pytest.raises(Exception) as raised:  # noqa: B017 - any but the one below
        interface.open_known_filetype_and_import_dataset(str(path))

    assert "Unknown data file extension" not in str(raised.value)


def test_an_unrelated_extension_is_still_refused_by_the_dispatcher(tmp_path):
    """Proving the probe above is measuring something."""
    path = tmp_path / "probe.docx"
    path.write_bytes(b"")

    interface = FileToLocalizationDataInterface(
        parent=type("P", (), {"localization_datasets": []})()
    )
    with pytest.raises(FileImportAbortedError, match="Unknown data file extension"):
        interface.open_known_filetype_and_import_dataset(str(path))


# ------------------------------------------------------------- the hook


@pytest.mark.parametrize("suffix", sorted(list_of_recognized_file_formats))
def test_the_hook_claims_every_advertised_format(suffix):
    assert napari_get_reader(f"acquisition.{suffix}") is not None


@pytest.mark.parametrize("suffix", ["MAT", "PMX", "CSV", "Hdf5"])
def test_the_hook_is_not_confused_by_case(suffix):
    """A dialog on a case-insensitive filesystem returns the name as stored."""
    assert napari_get_reader(f"acquisition.{suffix}") is not None


def test_the_hook_declines_a_format_this_plugin_does_not_read():
    assert napari_get_reader("movie.tif") is None
    assert napari_get_reader("notes.docx") is None


def test_the_hook_takes_the_suffix_of_the_file_and_not_of_its_folder(tmp_path):
    """`split(".")[-1]` on a path took the *directory's* extension."""
    folder = tmp_path / "session.csv"
    folder.mkdir()
    (folder / "README").write_bytes(b"")
    assert napari_get_reader(str(folder / "README")) is None


def test_the_hook_accepts_a_list_the_way_napari_hands_one_over():
    assert napari_get_reader(["acquisition.csv", "other.csv"]) is not None


# --------------------------------------------------------------- folders


def _zarr_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("zarr 3 cannot read structured arrays")
    root = tmp_path / "exp.zarr"
    array = np.zeros(4, dtype=[("itr", "<i4"), ("fnl", "?")])
    zarr.open_group(str(root), mode="w").create_dataset(
        "mfx", data=array, shape=array.shape, dtype=array.dtype
    )
    return root


def test_the_manifest_accepts_directories_so_a_zarr_store_can_be_opened():
    assert _manifest_readers()["accepts_directories"] is True


def test_the_hook_claims_a_zarr_store_directory(tmp_path):
    assert napari_get_reader(str(_zarr_store(tmp_path))) is not None


def test_the_hook_claims_a_path_inside_a_zarr_store(tmp_path):
    root = _zarr_store(tmp_path)
    assert napari_get_reader(str(root / "mfx" / ".zarray")) is not None


def test_the_hook_declines_an_ordinary_directory(tmp_path):
    """Accepting directories in the manifest must not claim every folder."""
    folder = tmp_path / "pictures"
    folder.mkdir()
    assert napari_get_reader(str(folder)) is None
