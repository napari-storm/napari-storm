"""Local localization density: how crowded each localization's surroundings are.

The measure daxview uses: for every localization, the number of *other*
localizations within a fixed radius -- a sphere in 3-D, a circle in 2-D. A
point exactly at the radius counts. Background and isolated blinks score
low, labelled structures high, so a lower threshold on it removes the
scatter between structures.

Host-free, like the rest of `core`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DENSITY_COLUMN", "DEFAULT_DENSITY_RADIUS_NM", "local_density"]

#: The column a density calculation writes, beside the measured ones.
DENSITY_COLUMN = "local_density"

#: daxview's default search radius.
DEFAULT_DENSITY_RADIUS_NM = 35.0


def local_density(coordinates_nm, radius_nm, neighbours=None):
    """For every row, how many *neighbours* lie within *radius_nm* of it.

    Args:
        coordinates_nm: ``(N, D)`` positions, D being 2 or 3.
        radius_nm: the search radius. Distances equal to it count.
        neighbours: boolean mask of the rows that count as neighbours, or
            None for all of them. A row is never its own neighbour; a row
            outside the mask still gets a count, of the neighbours around it.

    Rows with a non-finite position count nothing and are counted by no one.

    Returns:
        ``(N,)`` int32 counts.
    """
    from scipy.spatial import cKDTree

    coordinates = np.asarray(coordinates_nm, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("coordinates must be an (N, 2) or (N, 3) array")
    radius = float(radius_nm)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("the density radius must be a positive number of nm")
    n = len(coordinates)
    finite = np.isfinite(coordinates).all(axis=1)
    pool = finite.copy()
    if neighbours is not None:
        neighbours = np.asarray(neighbours, dtype=bool)
        if neighbours.shape != (n,):
            raise ValueError("neighbours needs one boolean per row")
        pool &= neighbours

    counts = np.zeros(n, dtype=np.int32)
    if not pool.any():
        return counts
    tree = cKDTree(coordinates[pool])
    counts[finite] = tree.query_ball_point(
        coordinates[finite], r=radius, return_length=True, workers=-1
    )
    # Every neighbour found itself at distance zero.
    counts[pool] -= 1
    return counts
