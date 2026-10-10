"""Lazy COMET fiducial adapter using raw nanometres."""

import numpy as np


def detect_fiducials(coords, frames, max_drift_nm, progress=None):
    from comet_fiducials import field_bounds_from_data, find_fiducials

    if progress:
        progress("detecting fiducials", {})
    data = np.column_stack((coords, frames))
    bounds = field_bounds_from_data(data, max_drift_nm)
    _, report = find_fiducials(data, max_drift_nm, bounds)
    return report


def exclusion_mask(coords, candidates):
    mask = np.zeros(len(coords), dtype=bool)
    for candidate in candidates:
        mask |= (
            np.sum((coords[:, :2] - candidate["center_nm"]) ** 2, axis=1)
            <= candidate["radius_p99_nm"] ** 2
        )
    return mask
