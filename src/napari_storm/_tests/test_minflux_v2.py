"""MINFLUX datasets in the Imspector >= 24.10 layout, in all five containers.

Imspector 24.10 flattened its export: one row per iteration, `itr` a scalar
column, and a new `fnl` flag marking the localization of each cycle. The
reader for the older layout indexes an iteration axis that no longer exists
and reads fields (`act`, `tic`, ...) that were dropped, so it fails on the
first one it touches rather than producing anything wrong.

Fixtures are written to the layout pyMINFLUX documents, since a real Abberior
export cannot be committed here. What they cannot check is whether a genuine
file matches that layout in some detail nobody wrote down.
"""

import json

import numpy as np
import pytest

from napari_storm.localization_dataset_types.minflux_v2 import (
    MinfluxDataV2Class,
    MinfluxV2FormatError,
    is_v2_file,
    read_v2_columns,
    select_iteration,
    zarr_store_root,
)

V2_DTYPE = [
    ("vld", "?"),
    ("fnl", "?"),
    ("bot", "?"),
    ("eot", "?"),
    ("sta", "u1"),
    ("tim", "<f8"),
    ("tid", "<u4"),
    ("gri", "<u4"),
    ("thi", "u1"),
    ("sqi", "u1"),
    ("itr", "<i4"),
    ("loc", "<f8", (3,)),
    ("lnc", "<f8", (3,)),
    ("eco", "<u4"),
    ("ecc", "<u4"),
    ("efo", "<f4"),
    ("efc", "<f4"),
    ("cfr", "<f2"),
    ("dcr", "<f2", (2,)),
    ("fbg", "<f4"),
]

#: The older layout, abbreviated: what matters is that `itr` is nested.
V1_DTYPE = [
    ("vld", "?"),
    ("tim", "<f8"),
    ("tid", "<u4"),
    ("act", "?"),
    (
        "itr",
        [
            ("loc", "<f8", (3,)),
            ("eco", "<u4"),
            ("efo", "<f4"),
            ("cfr", "<f4"),
            ("dcr", "<f4"),
            ("sta", "<i4"),
        ],
        (4,),
    ),
]

N_TRACES, N_ITR = 8, 4


def _v2_array(n_traces=N_TRACES, n_itr=N_ITR, z=True, invalid=0):
    rows = n_traces * n_itr
    a = np.zeros(rows, dtype=V2_DTYPE)
    a["vld"] = True
    a["itr"] = np.tile(np.arange(n_itr, dtype="<i4"), n_traces)
    a["fnl"] = a["itr"] == (n_itr - 1)
    a["tid"] = np.repeat(np.arange(n_traces, dtype="<u4"), n_itr)
    a["bot"] = a["itr"] == 0
    a["eot"] = a["fnl"]
    a["tim"] = np.arange(rows) * 1e-3
    # Metres, as Imspector writes them; a distinct value per row and axis so a
    # mix-up between axes or rows cannot pass unnoticed.
    a["loc"] = (np.arange(rows * 3).reshape(rows, 3) + 1) * 1e-9
    a["lnc"] = a["loc"]
    if not z:
        a["loc"][:, 2] = 0.0
    a["eco"] = np.arange(rows) + 10
    a["efo"] = np.arange(rows) * 1.5
    a["cfr"] = np.linspace(0, 1, rows)
    a["dcr"][:, 0] = np.linspace(0, 1, rows)
    a["dcr"][:, 1] = 1.0 - a["dcr"][:, 0]
    a["fbg"] = np.arange(rows) * 0.5
    if invalid:
        a["vld"][:invalid] = False
    return a


@pytest.fixture
def npy_file(tmp_path):
    path = tmp_path / "exp.npy"
    np.save(path, _v2_array())
    return str(path)


def test_the_new_layout_is_recognised_without_reading_the_array(npy_file):
    assert is_v2_file(npy_file)


def test_the_old_layout_is_not_mistaken_for_the_new_one(tmp_path):
    """`itr` names a nested sub-array there, not a scalar column."""
    path = tmp_path / "old.npy"
    np.save(path, np.zeros(3, dtype=V1_DTYPE))
    assert not is_v2_file(str(path))


def test_a_plain_array_is_rejected_rather_than_guessed_at(tmp_path):
    path = tmp_path / "plain.npy"
    np.save(path, np.zeros((4, 3)))
    with pytest.raises(MinfluxV2FormatError):
        is_v2_file(str(path))


def test_coordinates_arrive_in_nanometres(npy_file):
    dataset = MinfluxDataV2Class().load(npy_file)
    # Row 3 (itr 3) of trace 0 is the first final localization; its x is the
    # 10th element of the ramp above, in metres.
    assert dataset.locs_all.x_pos_nm[0] == pytest.approx(10.0, rel=1e-5)


def test_the_final_iteration_is_the_one_kept_by_default(npy_file):
    dataset = MinfluxDataV2Class().load(npy_file)
    assert len(dataset.locs_all) == N_TRACES
    assert np.all(dataset.locs_all.itr == N_ITR - 1)


def test_fnl_is_used_rather_than_the_highest_iteration_number(tmp_path):
    """A tracking run ends cycles at different iterations, so `fnl` and
    `itr == max` are not the same set of rows."""
    a = _v2_array(n_traces=3, n_itr=4)
    a["fnl"] = False
    a["fnl"][[1, 6, 11]] = True  # itr 1, 2 and 3 respectively
    path = tmp_path / "track.npy"
    np.save(path, a)
    kept = select_iteration(read_v2_columns(str(path)))
    assert sorted(kept["itr"]) == [1, 2, 3]


def test_an_iteration_can_still_be_picked_by_number(npy_file):
    dataset = MinfluxDataV2Class().load(npy_file, itr=1)
    assert np.all(dataset.locs_all.itr == 1)
    assert len(dataset.locs_all) == N_TRACES


def test_asking_for_an_iteration_that_is_not_there_says_so(npy_file):
    with pytest.raises(MinfluxV2FormatError):
        MinfluxDataV2Class().load(npy_file, itr=99)


def test_invalid_entries_are_dropped(tmp_path):
    path = tmp_path / "some_invalid.npy"
    np.save(path, _v2_array(invalid=N_ITR))  # the whole first trace
    dataset = MinfluxDataV2Class().load(str(path))
    assert len(dataset.locs_all) == N_TRACES - 1


def test_the_second_dcr_column_is_dropped(npy_file):
    """It only holds 1 - dcr[:, 0] and describes nothing on its own."""
    columns = read_v2_columns(npy_file)
    assert np.asarray(columns["dcr"]).ndim == 1


def test_a_flat_acquisition_reports_no_z(tmp_path):
    path = tmp_path / "flat.npy"
    np.save(path, _v2_array(z=False))
    assert not MinfluxDataV2Class().load(str(path)).zdim_present


def test_a_3d_acquisition_reports_z(npy_file):
    assert MinfluxDataV2Class().load(npy_file).zdim_present


def test_the_declared_dtype_is_produced_whole(npy_file):
    """The filter tab and the scene format both read `locs_dtype`, so every
    container has to yield the same columns."""
    dataset = MinfluxDataV2Class().load(npy_file)
    assert list(dataset.locs_all.dtype.names) == [
        name for name, _ in dataset.locs_dtype
    ]


# --- the other containers ----------------------------------------------------


def _reference(npy_file):
    return MinfluxDataV2Class().load(npy_file).locs_all


def test_json_reads_the_same_as_npy(tmp_path, npy_file):
    a = _v2_array()
    records = [
        {
            "vld": bool(row["vld"]),
            "fnl": bool(row["fnl"]),
            "bot": bool(row["bot"]),
            "eot": bool(row["eot"]),
            "sta": int(row["sta"]),
            "tim": float(row["tim"]),
            "tid": int(row["tid"]),
            "gri": int(row["gri"]),
            "thi": int(row["thi"]),
            "sqi": int(row["sqi"]),
            "itr": int(row["itr"]),
            "loc": [float(v) for v in row["loc"]],
            "lnc": [float(v) for v in row["lnc"]],
            "eco": int(row["eco"]),
            "ecc": int(row["ecc"]),
            "efo": float(row["efo"]),
            "efc": float(row["efc"]),
            "cfr": float(row["cfr"]),
            "dcr": [float(v) for v in row["dcr"]],
            "fbg": float(row["fbg"]),
        }
        for row in a
    ]
    path = tmp_path / "exp.json"
    path.write_text(json.dumps(records))
    assert is_v2_file(str(path))
    locs = MinfluxDataV2Class().load(str(path)).locs_all
    assert np.allclose(locs.x_pos_nm, _reference(npy_file).x_pos_nm)
    assert np.allclose(locs.z_pos_nm, _reference(npy_file).z_pos_nm)


def test_mat_reads_the_same_as_npy(tmp_path, npy_file):
    scipy_io = pytest.importorskip("scipy.io")
    a = _v2_array()
    scipy_io.savemat(
        str(tmp_path / "exp.mat"),
        {name: a[name] for name in a.dtype.names},
    )
    path = str(tmp_path / "exp.mat")
    assert is_v2_file(path)
    locs = MinfluxDataV2Class().load(path).locs_all
    assert np.allclose(locs.x_pos_nm, _reference(npy_file).x_pos_nm)


def test_zarr_reads_the_same_as_npy(tmp_path, npy_file):
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("zarr 3 cannot read structured arrays")
    a = _v2_array()
    root = tmp_path / "exp.zarr"
    group = zarr.open_group(str(root), mode="w")
    group.create_dataset("mfx", data=a, shape=a.shape, dtype=a.dtype)
    assert is_v2_file(str(root))
    locs = MinfluxDataV2Class().load(str(root)).locs_all
    assert np.allclose(locs.x_pos_nm, _reference(npy_file).x_pos_nm)


def test_a_zarr_store_is_found_from_a_path_inside_it(tmp_path):
    """A plain file dialog cannot select a folder, so picking any file in the
    store has to resolve to the same dataset."""
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("zarr 3 cannot read structured arrays")
    root = tmp_path / "exp.zarr"
    group = zarr.open_group(str(root), mode="w")
    a = _v2_array()
    group.create_dataset("mfx", data=a, shape=a.shape, dtype=a.dtype)
    assert zarr_store_root(str(root / ".zgroup")) == root.resolve()
    assert zarr_store_root(str(root / "mfx" / ".zarray")) == root.resolve()


def test_an_ordinary_directory_is_not_a_zarr_store(tmp_path):
    assert zarr_store_root(str(tmp_path)) is None
    assert not is_v2_file(str(tmp_path))


def test_pmx_reads_the_same_as_npy(tmp_path, npy_file):
    """pyMINFLUX's own save format, which is what its users actually keep."""
    h5py = pytest.importorskip("h5py")
    a = _v2_array()
    columns = {
        "vld": a["vld"].astype(float),
        "fnl": a["fnl"].astype(float),
        "tid": a["tid"].astype(float),
        "itr": a["itr"].astype(float),
        "tim": a["tim"],
        "x": a["loc"][:, 0],
        "y": a["loc"][:, 1],
        "z": a["loc"][:, 2],
        "eco": a["eco"].astype(float),
        "efo": a["efo"].astype(float),
        "cfr": a["cfr"].astype(float),
        "dcr": a["dcr"][:, 0].astype(float),
        "fbg": a["fbg"].astype(float),
    }
    path = tmp_path / "exp.pmx"
    with h5py.File(path, "w") as handle:
        handle.attrs["file_version"] = "3.0"
        handle.attrs["reader_version"] = 2
        table = handle.create_dataset(
            "/raw/df", data=np.column_stack(list(columns.values()))
        )
        table.attrs["column_names"] = list(columns)
    assert is_v2_file(str(path))
    locs = MinfluxDataV2Class().load(str(path)).locs_all
    assert np.allclose(locs.x_pos_nm, _reference(npy_file).x_pos_nm)


def test_an_older_pmx_file_is_not_read_as_the_new_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "old.pmx"
    with h5py.File(path, "w") as handle:
        handle.attrs["file_version"] = "2.0"
    assert not is_v2_file(str(path))


# --- routing -----------------------------------------------------------------


def _interface():
    from types import SimpleNamespace

    from napari_storm.FileToLocalizationDataInterface import (
        FileToLocalizationDataInterface,
    )

    return FileToLocalizationDataInterface(
        parent=SimpleNamespace(localization_datasets=[])
    )


def test_a_new_layout_npy_routes_to_the_new_reader(npy_file):
    """Both layouts are .npy, so the extension cannot decide this."""
    dataset = _interface().open_known_filetype_and_import_dataset(npy_file)[0]
    assert isinstance(dataset, MinfluxDataV2Class)
    assert len(dataset.locs_all) == N_TRACES


def test_an_old_layout_npy_still_reaches_the_old_reader(tmp_path):
    path = tmp_path / "old.npy"
    np.save(path, np.zeros(2, dtype=V1_DTYPE))
    # The stub is too thin to load; reaching that reader at all is the point,
    # and it is the one that fails on an iteration axis.
    with pytest.raises((IndexError, ValueError, KeyError)):
        _interface().open_known_filetype_and_import_dataset(str(path))


def test_a_zarr_dataset_is_named_after_its_store_not_the_file_picked(tmp_path):
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("zarr 3 cannot read structured arrays")
    root = tmp_path / "acquisition.zarr"
    a = _v2_array()
    zarr.open_group(str(root), mode="w").create_dataset(
        "mfx", data=a, shape=a.shape, dtype=a.dtype
    )
    inside = str(root / "mfx" / ".zarray")
    dataset = _interface().open_known_filetype_and_import_dataset(inside)[0]
    assert dataset.name == "acquisition.zarr"


# --- what only real files showed --------------------------------------------


def test_a_flat_acquisition_is_not_called_3d_by_float_noise(tmp_path):
    """A 2D MINFLUX run does not write z as exact zero.

    The Abberior files this was checked against carry z as float noise around
    1e-13 m -- every value non-zero, total extent 0.0003 nm. Testing `z != 0`
    therefore called every flat dataset 3D, which offers a z range slider, 3D
    widths and z colour coding for an extent six orders of magnitude below the
    localization precision. Synthetic fixtures were all exactly zero and could
    not have caught it.
    """
    a = _v2_array()
    rng = np.random.default_rng(0)
    a["loc"][:, 2] = rng.uniform(3e-14, 2.9e-13, len(a))
    path = tmp_path / "flat_but_noisy.npy"
    np.save(path, a)

    dataset = MinfluxDataV2Class().load(str(path))
    assert np.count_nonzero(dataset.locs_all.z_pos_nm) == len(dataset.locs_all)
    assert not dataset.zdim_present


def test_a_z_extent_of_hundreds_of_nanometres_is_3d(tmp_path):
    """The other side of the threshold: a real 3D run spans ~600 nm."""
    a = _v2_array()
    a["loc"][:, 2] = np.linspace(-2e-7, 6e-7, len(a))
    path = tmp_path / "really_3d.npy"
    np.save(path, a)
    assert MinfluxDataV2Class().load(str(path)).zdim_present
