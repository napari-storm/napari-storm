"""Exact allocation-free pair counts and conservative CPU memory estimates."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def count_pairs(coords, radius):
    tree = cKDTree(np.asarray(coords, dtype=float))
    return (int(tree.count_neighbors(tree, radius)) - len(coords)) // 2


@dataclass(frozen=True)
class MemoryEstimate:
    pairs: int
    additional_bytes: int
    resident_bytes: int
    physical_bytes: int
    reserve_bytes: int

    @property
    def fits(self):
        """Comfortably within memory: Run goes ahead."""
        return (
            self.resident_bytes + self.additional_bytes + self.reserve_bytes
            < self.physical_bytes
        )

    @property
    def possible(self):
        """Not comfortable, but the run alone would fit in physical memory: Run asks."""
        return self.additional_bytes < self.physical_bytes


def estimate_memory(coords, radius, *, total_rows=None, frame_count=0):
    import psutil

    pairs = count_pairs(coords, radius)
    physical = psutil.virtual_memory().total
    # COMET 1.2 (comet-smlm) finds pairs with one query_pairs call: int64 pairs (16 B)
    # and their int32 copies (8 B) at the peak, 24 B per pair; the optimisation
    # after it holds the 8 B and allocates nothing per evaluation on the CPU.
    # Per localization: COMET's copies (~60 B) and ours.
    extra = (
        pairs * 24
        + int(frame_count) * 128  # dense frame interpolation and its temporaries
        + len(coords) * 192
        + (total_rows or len(coords)) * 64
        + 256 * 1024**2
    )
    return MemoryEstimate(
        pairs,
        extra,
        psutil.Process().memory_info().rss,
        physical,
        max(1024**3, int(0.1 * physical)),
    )
