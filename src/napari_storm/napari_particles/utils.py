from typing import Tuple, Union

import numpy as np


def generate_billboards_2d(
    coords: np.ndarray, size: Union[float, np.ndarray] = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build billboard quads that always face the camera.

    Parameters
    ----------
    coords : (N, D) float array
        Particle centres; D is 2 (xy) or ≥3 (z, …).
    size : float or (N,) array
        Length of one edge of the square quad (same units as coords).

    Returns
    -------
    vertices : (6 N, D) float array
    faces    : (2 N, 3) int  array
    texcoords: (6 N, 2) float array   ← NEW: 6 rows, not 4
    """
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)

    # broadcast size
    size = (
        np.full(n, size, dtype=np.float32)
        if np.isscalar(size)
        else np.asarray(size, dtype=np.float32)
    )
    if size.shape != (n,):
        raise ValueError("`size` must be scalar or length-N array")

    # local quad, centred at origin, edge length = 1
    quad_xy = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32
    )

    # indices that build two triangles   0-1-2  and  0-2-3
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)  # 6 per particle
    vpp = len(idx)  # vertices-per-particle = 6

    # --- vertices -----------------------------------------------------------
    verts_xy = (quad_xy[idx] * size[:, None, None]).reshape(-1, 2)  # (6N, 2)
    if coords.shape[1] > 2:  # 3-D data: prepend extra cols
        rep_coords = np.repeat(coords, vpp, axis=0)
        verts = np.concatenate([rep_coords[:, :-2], verts_xy], axis=1)
    else:  # 2-D data
        verts = verts_xy

    # --- faces --------------------------------------------------------------
    # uint32 is what the GPU index buffer wants, and it is what the offsets must
    # be built in: np.arange defaults to int64, which would silently widen the
    # whole (2N, 3) index array and double its footprint.  6 * 5e6 vertices
    # still fits comfortably below 2**32.
    faces_base = np.array([[0, 1, 2], [5, 4, 3]], dtype=np.uint32)  # (2, 3)

    # shape (n, 1, 1)  so it broadcasts against (1, 2, 3)
    particle_offset = (np.arange(n, dtype=np.uint32) * np.uint32(vpp))[:, None, None]

    faces = (faces_base[None, ...] + particle_offset).reshape(-1, 3)  # (2N, 3)

    # --- texcoords  (6 rows / particle) ------------------------------------
    base_tc = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    texcoords = np.tile(base_tc[idx], (n, 1))  # (6N, 2)

    return verts, faces, texcoords
