"""ThunderSTORM CSV headers, in the spellings files actually arrive with.

The loader matched header text exactly, so it carried one branch per spelling:
`"x [nm]"` and `x [nm]` were two cases of the same column, and `x  [nm]` --
the spacing reported in issue #17 -- was neither. Several of those branches
also read a companion column without checking for it, which turned a merely
unusual file into a `KeyError` from inside numpy indexing.

These pin the normalization and each of the reads that used to be blind.
"""

import numpy as np
import pytest

from napari_storm.localization_dataset_types.storm_class import (
    StormDataClass,
    _normalized_csv_header,
)


def _write(tmp_path, header, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "locs.csv"
    body = "\n".join(",".join(str(value) for value in row) for row in rows)
    path.write_text(header + "\n" + body + "\n")
    return str(path)


def _load(tmp_path, header, rows):
    return StormDataClass().load_csv(_write(tmp_path, header, rows), "locs")[0]


@pytest.mark.parametrize(
    "field",
    ['"x [nm]"', "x [nm]", "x  [nm]", " x [nm] ", "X [nm]", "'x [nm]'"],
)
def test_every_spelling_of_a_header_field_reduces_to_one_key(field):
    assert _normalized_csv_header(field) == "x [nm]"


def test_a_double_space_in_the_unit_still_finds_the_column(tmp_path):
    """Issue #17: `x  [nm]` used to raise ImportError."""
    dataset = _load(
        tmp_path,
        "x  [nm],y  [nm]",
        [(10.0, 20.0), (30.0, 40.0)],
    )
    assert list(dataset.locs_all.x_pos_pixels) == [10.0, 30.0]
    assert list(dataset.locs_all.y_pos_pixels) == [20.0, 40.0]


def test_quoted_and_unquoted_headers_load_identically(tmp_path):
    rows = [(1.0, 2.0, 3.0)]
    quoted = _load(tmp_path / "q", '"x [nm]","y [nm]","z [nm]"', rows)
    bare = _load(tmp_path / "b", "x [nm],y [nm],z [nm]", rows)
    assert quoted.locs_all.x_pos_pixels == bare.locs_all.x_pos_pixels
    assert quoted.zdim_present == bare.zdim_present is True


def test_a_missing_y_column_is_an_import_error_not_a_key_error(tmp_path):
    """It was `KeyError` before, raised from a lookup that assumed y existed."""
    with pytest.raises(ImportError):
        _load(tmp_path, "x [nm],frame", [(1.0, 0.0)])


def test_lateral_uncertainty_without_its_y_column_is_read_as_isotropic(tmp_path):
    dataset = _load(
        tmp_path,
        "x [nm],y [nm],uncertainty_x [nm]",
        [(1.0, 2.0, 7.0)],
    )
    assert dataset.sigma_present
    assert dataset.locs_all.sigma_x_pixels == dataset.locs_all.sigma_y_pixels == 7.0


def test_a_3d_file_without_a_z_uncertainty_derives_one(tmp_path):
    """The uncertainty_xy branch read `uncertainty_z [nm]` unguarded."""
    dataset = _load(
        tmp_path,
        "x [nm],y [nm],z [nm],uncertainty_xy [nm]",
        [(1.0, 2.0, 3.0, 6.0)],
    )
    assert dataset.zdim_present
    assert dataset.locs_all.sigma_z_pixels == pytest.approx(2 * np.sqrt(2 * 6.0**2))


def test_photon_counts_survive_a_file_that_also_carries_uncertainties(tmp_path):
    """They were behind an `elif`, so a real ThunderSTORM export lost them."""
    dataset = _load(
        tmp_path,
        '"x [nm]","y [nm]","uncertainty_xy [nm]","intensity [photon]"',
        [(1.0, 2.0, 5.0, 900.0)],
    )
    assert dataset.sigma_present
    assert dataset.photon_count_present
    assert dataset.locs_all.photon_count == 900.0


def test_a_single_localization_file_loads_as_one_row(tmp_path):
    """`np.loadtxt` hands back a 1-D array for one row; indexing it as
    columns raised IndexError."""
    dataset = _load(tmp_path, "x [nm],y [nm]", [(5.0, 6.0)])
    assert len(dataset.locs_all) == 1
    assert dataset.locs_all.x_pos_pixels == 5.0


def test_a_flat_file_reports_no_z_dimension(tmp_path):
    dataset = _load(tmp_path, "x [nm],y [nm]", [(1.0, 2.0), (3.0, 4.0)])
    assert not dataset.zdim_present
    assert not dataset.sigma_present
    assert not dataset.photon_count_present
