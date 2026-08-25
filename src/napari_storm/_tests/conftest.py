import numpy as np
import pytest

from napari_storm.localization_dataset_types import LocalizationDataBaseClass


@pytest.fixture
def two_d_dataset():
    """10-point 2-D SMLM dataset for use in tests."""
    n = 10
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    return LocalizationDataBaseClass(
        np.rec.array(locs), name="test_2d", zdim_present=False
    )


@pytest.fixture
def three_d_dataset():
    """10-point 3-D SMLM dataset for use in tests."""
    n = 10
    locs = np.zeros(
        n,
        dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")],
    )
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    locs["z_pos_nm"] = np.linspace(-500, 500, n)
    return LocalizationDataBaseClass(
        np.rec.array(locs), name="test_3d", zdim_present=True
    )
