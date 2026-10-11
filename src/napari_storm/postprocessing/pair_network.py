"""Complete cross-window networks within a bounded, centred region."""

import numpy as np
from scipy.spatial import cKDTree

from ..core.pairs import PairSet


def find_pairs(
    coords,
    times,
    windows,
    row_ids,
    radius,
    *,
    groups=None,
    cap=200000,
    z_scale=1.0,
    chains=False,
    min_dt=0.0,
    progress=None,
    raw_coords=None,
):
    coords = np.asarray(coords, dtype=float)
    times, windows, row_ids = map(np.asarray, (times, windows, row_ids))
    if radius <= 0 or not np.isfinite(radius) or cap < 1 or z_scale <= 0:
        raise ValueError("Radius, cap and z scale must be positive")
    if not len(coords):
        return PairSet(
            np.empty((0, 2), dtype=np.intp), np.empty(0), raw_count=0, corrected_count=0
        )
    report = progress or (lambda *args: None)
    center = (coords[:, :2].min(axis=0) + coords[:, :2].max(axis=0)) / 2
    full_side = float(np.ptp(coords[:, :2], axis=0).max())
    scaled = coords.copy()
    scaled[:, 2] *= z_scale

    # A pair set larger than this many times the cap is not built at all: the
    # region is narrowed instead (only a cap's worth will be drawn anyway).
    materialise_limit = 20 * cap

    def search(side):
        if side >= full_side:
            ids = np.arange(
                len(coords)
            )  # the whole region: no edge test to lose a point to
        else:
            ids = np.flatnonzero(
                (np.abs(coords[:, :2] - center) <= side / 2).all(axis=1)
            )
        if len(ids) < 2:
            return np.empty((0, 2), dtype=np.intp)
        report("pair search", {"rows": int(len(ids))})
        tree = cKDTree(scaled[ids])
        if (tree.count_neighbors(tree, radius) - len(ids)) // 2 > materialise_limit:
            return None
        local = tree.query_pairs(radius, output_type="ndarray")
        a, b = ids[local[:, 0]], ids[local[:, 1]]
        del local
        valid = (windows[a] != windows[b]) & (windows[a] >= 0) & (windows[b] >= 0)
        if groups is not None:
            valid &= (groups[a] < 0) | (groups[a] != groups[b])
        a, b = a[valid], b[valid]
        if chains:
            # every pair in both directions, keep the later-window partners, and
            # for each localization the nearest of them
            src, dst = np.concatenate([a, b]), np.concatenate([b, a])
            later = (windows[dst] > windows[src]) & (times[dst] - times[src] > min_dt)
            src, dst = src[later], dst[later]
            distance = np.sum((scaled[dst] - scaled[src]) ** 2, axis=1)
            order = np.lexsort((distance, src))
            src, dst = src[order], dst[order]
            first = np.ones(len(src), dtype=bool)
            first[1:] = src[1:] != src[:-1]
            a, b = src[first], dst[first]
        if len(a) > cap:
            return None
        return np.column_stack((a, b)).astype(np.intp)

    result = search(full_side)
    side = full_side
    if result is None:
        # The largest centred square whose network fits the cap, to ~1 %.
        low, high = 0.0, full_side
        result = np.empty((0, 2), dtype=np.intp)
        while high - low > 0.01 * full_side:
            trial_side = (low + high) / 2
            trial = search(trial_side)
            if trial is None:
                high = trial_side
            else:
                result, low = trial, trial_side
        side = low
    raw_count = corrected_count = None
    if raw_coords is not None:
        from .pair_budget import count_pairs

        report("counting close pairs", {})
        raw_scaled = np.array(raw_coords, dtype=float, copy=True)
        raw_scaled[:, 2] *= z_scale
        raw_count = count_pairs(raw_scaled, radius)
        corrected_count = count_pairs(scaled, radius)
    return PairSet(
        row_ids[result],
        np.abs(times[result[:, 1]] - times[result[:, 0]]),
        side,
        raw_count,
        corrected_count,
    )
