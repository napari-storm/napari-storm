"""Frame linking with one observation per group per frame."""

import numpy as np
from scipy.spatial import cKDTree


def link_localizations(
    coords,
    frames,
    *,
    max_distance=30.0,
    max_dark_frames=1,
    max_frames=50,
    weights=None,
    progress=None
):
    coords, frames = np.asarray(coords, float), np.asarray(frames)
    if max_distance <= 0 or max_dark_frames < 0 or max_frames < 1:
        raise ValueError("Invalid grouping distance or frame limits")
    weights = np.ones_like(coords) if weights is None else np.asarray(weights)
    result = np.full(len(coords), -1, dtype=np.int64)
    sums, totals, first, last = [], [], [], []
    order = np.argsort(frames, kind="stable")
    batches = np.split(order, np.flatnonzero(np.diff(frames[order])) + 1)
    open_ids = []
    for index, rows in enumerate(batches):
        if not len(rows):
            continue
        frame = frames[rows[0]]
        if progress and index % 128 == 0:
            progress("grouping", {"frame": float(frame)})
        open_ids = [
            g
            for g in open_ids
            if frame - last[g] <= max_dark_frames + 1 and frame - first[g] < max_frames
        ]
        candidates = []
        if open_ids:
            means = np.array([sums[g] / totals[g] for g in open_ids])
            tree = cKDTree(means)
            for row in rows:
                for j in tree.query_ball_point(coords[row], max_distance):
                    candidates.append(
                        (
                            float(np.linalg.norm(coords[row] - means[j])),
                            int(row),
                            open_ids[j],
                        )
                    )
        used = set()
        for _, row, group in sorted(candidates):
            if result[row] >= 0 or group in used:
                continue
            result[row] = group
            used.add(group)
            sums[group] += coords[row] * weights[row]
            totals[group] += weights[row]
            last[group] = frame
        for row in rows[result[rows] < 0]:
            group = len(sums)
            result[row] = group
            sums.append(coords[row] * weights[row])
            totals.append(weights[row].copy())
            first.append(frame)
            last.append(frame)
            open_ids.append(group)
    return result


def group_means(coords, frames, groups, window=None, weights=None):
    """Average each group; optionally split at explicit frame-window boundaries."""
    weights = np.ones_like(coords) if weights is None else weights
    windows = (
        np.zeros(len(frames), dtype=np.int64)
        if window is None
        else np.floor((frames - frames.min()) / window).astype(np.int64)
    )
    # Unassigned rows remain individual observations.
    groups = np.asarray(groups).copy()
    missing = groups < 0
    groups[missing] = np.arange(missing.sum()) + max(0, int(groups.max()) + 1)
    _, inverse = np.unique(
        np.column_stack((groups, windows)), axis=0, return_inverse=True
    )
    n = inverse.max() + 1
    means = np.column_stack(
        [
            np.bincount(inverse, weights=coords[:, i] * weights[:, i], minlength=n)
            / np.bincount(inverse, weights=weights[:, i], minlength=n)
            for i in range(3)
        ]
    )
    times = np.bincount(inverse, weights=frames) / np.bincount(inverse)
    return means, np.rint(times)
