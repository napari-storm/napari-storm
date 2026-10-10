"""Deterministic blinking emitters with known drift for tests and examples."""

import numpy as np


def drifting_emitters(n_frames=200, n_emitters=50, seed=5, beads=0, nonfinite=False):
    rng = np.random.default_rng(seed)
    sites = rng.normal(0, 150, (n_emitters, 3))
    frames = np.repeat(np.arange(n_frames), 20)
    emitter = rng.integers(0, n_emitters, len(frames))
    if beads:
        bead_sites = rng.normal(0, 300, (beads, 3))
        sites = np.vstack((sites, bead_sites))
        frames = np.concatenate((frames, np.tile(np.arange(n_frames), beads)))
        emitter = np.concatenate(
            (emitter, np.repeat(np.arange(n_emitters, n_emitters + beads), n_frames))
        )
    drift = np.column_stack(
        (
            np.arange(n_frames) * 0.1,
            np.arange(n_frames) * -0.04,
            np.arange(n_frames) * 0.03,
        )
    )
    coords = sites[emitter] + drift[frames] + rng.normal(0, 1, (len(frames), 3))
    dtype = [(a + "_pos_nm", "f4") for a in ("x", "y", "z")] + [("frame_number", "i4")]
    records = np.zeros(len(frames), dtype=dtype).view(np.recarray)
    for axis, column in zip(("x", "y", "z"), coords.T):
        records[axis + "_pos_nm"] = column
    records.frame_number = frames
    if nonfinite:
        records.x_pos_nm[-1] = np.nan
    return records, drift, emitter
