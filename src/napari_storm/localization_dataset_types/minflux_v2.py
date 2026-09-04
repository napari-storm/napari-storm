"""Abberior MINFLUX datasets in the Imspector >= 24.10 layout.

Imspector 24.10 changed what it exports.  The older layout -- what
:mod:`.Minflux_class` reads -- stores one row per trace with a *nested* ``itr``
sub-array, so a localization is chosen by indexing an iteration axis.  The new
one is flat: one row **per iteration**, ``itr`` a scalar column, and the
localization of each cycle marked by a new ``fnl`` flag.  Fields the old layout
carried per iteration (``act``, ``tic``, ``gvx``/``gvy``, ``eox``/``eoy``,
``dmz``, ``lcx``/``lcy``/``lcz``, ``ext``) are gone; ``gri``, ``thi`` and
``sqi`` are new; and ``dcr`` gained a second column that is only
``1 - dcr[:, 0]``.

Reading the old reader's code against a new file fails on the first missing
field, which is why this is a separate reader rather than a widened one.

The layout, the version markers and the per-container quirks were taken from
pyMINFLUX (https://github.com/bsse-scf/pyMINFLUX, D-BSSE ETH Zurich,
Apache-2.0), which documents this format in its ``MinFluxReaderV2``.
"""

import json
from pathlib import Path

import numpy as np

from .._data_constants import MINFLUX_Z_CORRECTION_FACTOR
from .data_formats import minflux_v2_data_dtype
from .Minflux_class import MinfluxDataBaseClass

#: Metres to nanometres.  Every container stores coordinates in metres.
_M_TO_NM = 1e9

#: Smallest z extent, in nanometres, that counts as a third dimension.
#:
#: A 2D acquisition does not write z as exact zero: the Abberior files this was
#: checked against carry float noise around 1e-13 m, so *every* value is
#: non-zero and "any z != 0" calls every flat dataset 3D -- which then offers a
#: z range slider, 3D widths and z colour coding for an extent of 0.0003 nm.
#: A nanometre sits four orders of magnitude above that noise and two below the
#: extent of a real 3D dataset, so nothing plausible falls near it.
_MIN_Z_EXTENT_NM = 1.0

#: A field that only the new layout has.  ``itr`` is listed separately because
#: both layouts carry the name -- it is the *scalar* dtype that marks v2.
_V2_MARKER_FIELDS = ("fnl", "bot", "eot")

#: The layouts this package can read, and the answer for a file that is
#: neither.  A boolean could only say "v2 or not", which left every file that
#: is not v2 -- including files that are not MINFLUX at all -- indistinguishable
#: from a genuine old-layout export and sent to the old reader.
LAYOUT_V1 = "v1"
LAYOUT_V2 = "v2"
LAYOUT_UNKNOWN = "unknown"

#: How much of a JSON export to read while looking for its first record.  One
#: old-layout record, nested iterations and all, is a few kB; a megabyte is
#: generous, and the cap below keeps a pathological file from being read whole.
_JSON_PROBE_BYTES = 1 << 20
_JSON_PROBE_LIMIT = 16 << 20

#: Columns this reader keeps, mapped from the names the containers use to the
#: names :data:`minflux_v2_data_dtype` declares.
_COLUMN_ALIASES = {
    "tid": "trace_id",
    "tim": "time_s",
    "lncx": "lnc_x",
    "lncy": "lnc_y",
    "lncz": "lnc_z",
}


class MinfluxV2FormatError(Exception):
    """The file is not a MINFLUX dataset this reader can read."""


def _layout_of_fields(names_and_types):
    """The layout a (name, dtype-ish) sequence describes."""
    carries_itr = False
    for name, dtype in names_and_types:
        if name in _V2_MARKER_FIELDS:
            return LAYOUT_V2
        if name == "itr":
            carries_itr = True
            if np.dtype(dtype).kind in "iu":
                # v1 nests a structured sub-array under `itr`; v2 makes it a
                # plain integer.  The kind is the whole difference.
                return LAYOUT_V2
    return LAYOUT_V1 if carries_itr else LAYOUT_UNKNOWN


def _first_json_record(path):
    """The first object in a JSON export, without parsing the rest of it.

    An export runs to gigabytes and the reader that follows will parse it
    properly anyway, so deciding which reader that is must not cost a parse of
    its own.
    """
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as handle:
        text = handle.read(_JSON_PROBE_BYTES)
        while True:
            start = text.find("{")
            if start >= 0:
                try:
                    record, _end = decoder.raw_decode(text, start)
                except ValueError:
                    pass  # Truncated mid-record: read further and retry.
                else:
                    return record if isinstance(record, dict) else None
            if len(text) >= _JSON_PROBE_LIMIT:
                return None
            more = handle.read(_JSON_PROBE_BYTES)
            if not more:
                return None
            text += more


def json_layout(path):
    """The layout of a MINFLUX JSON export, from its first record.

    Every ``.json`` used to be called v2 outright, which sent old exports --
    which the retained v1 reader handles -- to a reader that rejects them.
    """
    record = _first_json_record(path)
    if not isinstance(record, dict):
        return LAYOUT_UNKNOWN
    if any(marker in record for marker in _V2_MARKER_FIELDS):
        return LAYOUT_V2
    iteration = record.get("itr")
    if isinstance(iteration, bool) or iteration is None:
        return LAYOUT_UNKNOWN
    if isinstance(iteration, int):
        # v2 numbers the iteration; v1 stores the iterations themselves.
        return LAYOUT_V2
    if isinstance(iteration, (list, dict)):
        return LAYOUT_V1
    return LAYOUT_UNKNOWN


def file_layout(file_path):
    """Which MINFLUX layout *file_path* holds, in any of the containers.

    Cheap by design: a ``.npy`` is decided from its header and a ``.json`` from
    its first record, so routing a multi-gigabyte export costs no more than
    opening it.
    """
    path = Path(file_path)
    if zarr_store_root(path) is not None:
        return LAYOUT_V2
    if path.is_dir():
        return LAYOUT_UNKNOWN
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _layout_of_fields(_npy_header_fields(path))
    if suffix == ".json":
        return json_layout(path)
    if suffix == ".mat":
        from scipy.io import whosmat

        return _layout_of_fields(
            (name, dtype) for name, _shape, dtype in whosmat(str(path))
        )
    if suffix == ".pmx":
        return LAYOUT_V2 if _pmx_reader_version(path) == 2 else LAYOUT_V1
    return LAYOUT_UNKNOWN


def zarr_store_root(file_path):
    """The Zarr MINFLUX store *file_path* belongs to, or None.

    Accepts the store directory itself and any path inside it.  That second
    case is what makes a Zarr dataset openable at all from a plain file
    dialog, which cannot select a folder: picking any file within the store
    -- its ``.zgroup``, a chunk -- resolves to the same root.
    """
    root = _zarr_root(file_path)
    if root is None or not (root / "mfx").is_dir():
        return None
    return root


def is_v2_file(file_path):
    """Whether *file_path* holds a MINFLUX dataset in the new layout."""
    return file_layout(file_path) == LAYOUT_V2


def _npy_header_fields(path):
    """The (name, dtype) pairs of a ``.npy`` file, without reading the array.

    ``np.lib.format.read_array_header_*`` rejects the header some Imspector
    files carry, so the dict is parsed directly -- the same workaround
    pyMINFLUX applies.
    """
    import ast

    with open(path, "rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise MinfluxV2FormatError(f"{path} is not a .npy file")
        version = handle.read(2)
        if len(version) != 2:
            raise MinfluxV2FormatError(f"the header of {path} is truncated")
        major, minor = version[0], version[1]
        if major not in (1, 2, 3):
            raise MinfluxV2FormatError(
                f"{path} is .npy format {major}.{minor}, which this reader "
                "does not know"
            )
        # Format 1.0 sizes its header with two bytes and 2.0/3.0 with four.
        # Reading two either way took half the length of a valid v2 file and
        # then decoded from the wrong offset, so the file looked corrupt.
        length_bytes = 2 if major == 1 else 4
        raw_length = handle.read(length_bytes)
        if len(raw_length) != length_bytes:
            raise MinfluxV2FormatError(f"the header of {path} is truncated")
        header_length = int.from_bytes(raw_length, byteorder="little")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise MinfluxV2FormatError(f"the header of {path} is truncated")
        # 3.0 declares its header UTF-8; 1.0 and 2.0 are latin1.
        header = raw_header.decode("utf-8" if major >= 3 else "latin1")
    try:
        descr = ast.literal_eval(header.replace("\n", "").replace(" ", ""))["descr"]
    except (SyntaxError, ValueError, KeyError) as error:
        raise MinfluxV2FormatError(f"cannot parse the header of {path}") from error
    if not isinstance(descr, list):
        raise MinfluxV2FormatError(f"{path} holds a plain array, not a table")
    return [(field[0], field[1]) for field in descr]


def _pmx_reader_version(path):
    import h5py

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("file_version") not in ("1.0", "2.0", "3.0"):
            return -1
        if handle.attrs.get("file_version") in ("1.0", "2.0"):
            return 1
        try:
            return int(handle.attrs["reader_version"])
        except (KeyError, ValueError, TypeError):
            return -1


def _zarr_root(path):
    """The root of the Zarr store *path* sits in, or None.

    A Zarr dataset is a directory, and the user may well pick a group inside
    it, so walk up until the parent stops carrying Zarr metadata.
    """
    path = Path(path).resolve()
    while path != path.parent:
        if (path / ".zgroup").exists() or (path / ".zarray").exists():
            parent = path.parent
            if not ((parent / ".zgroup").exists() or (parent / ".zarray").exists()):
                return path
            path = parent
        else:
            path = path.parent
    return None


def _columns_from_structured(array):
    """Split a structured array into named 1-D columns.

    ``loc`` and ``lnc`` arrive as ``(N, 3)``; ``dcr`` as ``(N, 2)`` whose
    second column is only ``1 - dcr[:, 0]`` and carries nothing.
    """
    columns = {}
    for name in array.dtype.names:
        values = array[name]
        if name == "loc":
            columns["x"], columns["y"], columns["z"] = (values[:, i] for i in range(3))
        elif name == "lnc":
            columns["lncx"], columns["lncy"], columns["lncz"] = (
                values[:, i] for i in range(3)
            )
        elif name == "dcr":
            columns["dcr"] = values[:, 0] if values.ndim > 1 else values
        else:
            columns[name] = values
    return columns


def _drop_invalid(array):
    """Keep the entries Imspector marked valid."""
    if "vld" in (array.dtype.names or ()):
        return array[array["vld"].astype(bool)]
    return array


def _read_npy(path):
    return _columns_from_structured(
        _drop_invalid(np.load(str(path), allow_pickle=False))
    )


def _read_zarr(path):
    import zarr

    root = zarr_store_root(path)
    if root is None:
        raise MinfluxV2FormatError(f"{path} is not a MINFLUX Zarr store")
    table = root / "mfx"
    try:
        array = np.asarray(zarr.load(str(table)))
    except ValueError as error:
        # zarr 3 rejects the structured dtype Imspector writes, for stores
        # zarr 2 reads back fine.  Say so, rather than surfacing "No Zarr data
        # type found that matches" from three frames down.
        raise MinfluxV2FormatError(
            f"could not read {table} with zarr {zarr.__version__}. MINFLUX "
            f"Zarr datasets are structured arrays, which zarr 3 does not "
            f"support; install zarr<3 to open them."
        ) from error
    if array is None or array.dtype.names is None:
        raise MinfluxV2FormatError(f"{table} is not a MINFLUX table")
    columns = _columns_from_structured(_drop_invalid(array))
    return _drop_incomplete_traces(columns)


def _drop_incomplete_traces(columns):
    """Zarr stores keep traces that were cut short; a partial cycle is not a
    localization, so they go before anything is measured from them."""
    if "tid" not in columns or "itr" not in columns:
        return columns
    expected = int(np.max(columns["itr"])) + 1
    trace_ids = np.asarray(columns["tid"])
    unique, counts = np.unique(trace_ids, return_counts=True)
    complete = np.isin(trace_ids, unique[counts >= expected])
    return {name: np.asarray(values)[complete] for name, values in columns.items()}


def _read_mat(path):
    from scipy.io import loadmat

    raw = loadmat(str(path))
    columns = {}
    for key, values in raw.items():
        if key.startswith("__"):
            continue
        values = np.asarray(values)
        if key == "loc":
            columns["x"], columns["y"], columns["z"] = (values[:, i] for i in range(3))
        elif key == "lnc":
            columns["lncx"], columns["lncy"], columns["lncz"] = (
                values[:, i] for i in range(3)
            )
        elif key == "dcr":
            columns["dcr"] = values[:, 0] if values.ndim > 1 else values.ravel()
        else:
            columns[key] = values.ravel()
    return _drop_invalid_columns(columns)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise MinfluxV2FormatError(f"{path} holds no MINFLUX records")
    layout = _layout_of_fields(
        (key, np.asarray(records[0][key]).dtype) for key in records[0]
    )
    if layout != LAYOUT_V2:
        raise MinfluxV2FormatError(f"{path} is not in the Imspector >= 24.10 layout")
    columns = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key == "loc":
            columns["x"], columns["y"], columns["z"] = (
                np.array([value[i] for value in values]) for i in range(3)
            )
        elif key == "lnc":
            columns["lncx"], columns["lncy"], columns["lncz"] = (
                np.array([value[i] for value in values]) for i in range(3)
            )
        elif key == "dcr":
            columns["dcr"] = np.array(
                [value[0] if np.ndim(value) else value for value in values]
            )
        else:
            columns[key] = np.array(values)
    return _drop_invalid_columns(columns)


def _read_pmx(path):
    """pyMINFLUX's own format: the processed table under ``/raw/df``."""
    import h5py

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("file_version") != "3.0":
            raise MinfluxV2FormatError(
                f"{path} is a .pmx file of version "
                f"{handle.attrs.get('file_version')!r}; only 3.0 holds the new layout"
            )
        dataset = handle["/raw/df"]
        values = dataset[:]
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in dataset.attrs["column_names"]
        ]
    columns = {name: np.asarray(values[:, index]) for index, name in enumerate(names)}
    return _drop_invalid_columns(columns)


def _drop_invalid_columns(columns):
    """`_drop_invalid` for the containers that arrive as separate columns."""
    if "vld" not in columns:
        return columns
    valid = np.asarray(columns["vld"]).astype(bool)
    return {name: np.asarray(values)[valid] for name, values in columns.items()}


_READERS = {
    ".npy": _read_npy,
    ".mat": _read_mat,
    ".json": _read_json,
    ".pmx": _read_pmx,
}


def read_v2_columns(file_path):
    """Every valid row of *file_path*, as named 1-D columns.

    One row per iteration: selecting localizations is the caller's job, and
    :func:`select_iteration` does it.
    """
    path = Path(file_path)
    if zarr_store_root(path) is not None:
        return _read_zarr(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise MinfluxV2FormatError(f"{path.suffix} is not a MINFLUX container")
    return reader(path)


def select_iteration(columns, itr=-1):
    """The rows that make up one localization per cycle.

    *itr* of -1 means the final iteration of each cycle, which the new layout
    marks with ``fnl`` -- and which is not simply the highest ``itr`` value,
    because a tracking acquisition ends cycles at different iterations.  Any
    other value selects that iteration by number, as the old reader's iteration
    picker did.
    """
    if itr == -1:
        if "fnl" in columns:
            mask = np.asarray(columns["fnl"]).astype(bool)
        else:
            iterations = np.asarray(columns["itr"])
            mask = iterations == iterations.max()
    else:
        mask = np.asarray(columns["itr"]) == itr
    if not mask.any():
        raise MinfluxV2FormatError(f"no localizations for iteration {itr}")
    return {name: np.asarray(values)[mask] for name, values in columns.items()}


def _locs_from_columns(columns):
    """Build the record array :data:`minflux_v2_data_dtype` describes."""
    count = len(columns["x"])
    scaled = {
        "x_pos_nm": np.asarray(columns["x"], dtype="f8") * _M_TO_NM,
        "y_pos_nm": np.asarray(columns["y"], dtype="f8") * _M_TO_NM,
        "z_pos_nm": np.asarray(columns["z"], dtype="f8")
        * _M_TO_NM
        * MINFLUX_Z_CORRECTION_FACTOR,
    }
    for source, target in _COLUMN_ALIASES.items():
        if source in columns:
            scaled[target] = columns[source]
    fields = []
    for name, _dtype in minflux_v2_data_dtype:
        if name in scaled:
            values = scaled[name]
        elif name in columns:
            values = columns[name]
        else:
            # A container that does not carry an optional column still has to
            # produce the declared dtype, or the filter tab and the scene
            # format would disagree about what a v2 dataset is.
            values = np.zeros(count)
        fields.append(np.asarray(values).reshape(count))
    if "lnc_x" in scaled:
        for axis in ("x", "y", "z"):
            index = [name for name, _ in minflux_v2_data_dtype].index(f"lnc_{axis}")
            fields[index] = np.asarray(fields[index], dtype="f8") * _M_TO_NM
    return np.rec.array(tuple(fields), dtype=minflux_v2_data_dtype)


class MinfluxDataV2Class(MinfluxDataBaseClass):
    """One MINFLUX dataset read from the Imspector >= 24.10 layout."""

    def __init__(self, locs=None, name=None, zdim_present=False, itr=-1):
        if locs is not None:
            super().__init__(None, name, zdim_present)
            self.locs_all = locs.copy()
            self.name = name if name is not None else "untitled"
        self.dataset_type = "MinfluxDataV2Class(MinfluxDataBaseClass)"
        self.itr = itr
        self.locs_dtype = minflux_v2_data_dtype
        self.zdim_present = zdim_present

    def load(self, file_path, name=None, itr=-1):
        """Read *file_path*, keeping one localization per cycle."""
        columns = select_iteration(read_v2_columns(file_path), itr=itr)
        locs = _locs_from_columns(columns)
        # A 2-D acquisition still writes a z column; see _MIN_Z_EXTENT_NM for
        # why its contents cannot be tested against zero.
        zdim_present = bool(np.ptp(locs.z_pos_nm) > _MIN_Z_EXTENT_NM)
        return MinfluxDataV2Class(
            locs=locs,
            name=name if name is not None else Path(file_path).name,
            zdim_present=zdim_present,
            itr=itr,
        )

    def load_ns(self, dataset):
        return MinfluxDataV2Class(
            locs=np.rec.array(dataset[...]),
            name=dataset.attrs["name"],
            zdim_present=dataset.attrs["zdim_present"],
            itr=dataset.attrs.get("itr", -1),
        )
