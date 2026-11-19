"""Stable row-id edges, independent of the viewer and pair-search algorithm."""

from dataclasses import dataclass

import numpy as np

from .traces import TraceVertices


@dataclass(frozen=True)
class PairSet:
    row_ids: np.ndarray
    delta_time: np.ndarray
    region_side_nm: float = 0.0
    raw_count: int = None
    corrected_count: int = None

    def vertices(self, table, coordinates_by_ids):
        edges = np.asarray(self.row_ids, dtype=np.intp).reshape(-1, 2)
        keep = np.all(table.filter_mask[edges], axis=1)
        edges = edges[keep]
        n = len(edges)
        return TraceVertices(
            coords=coordinates_by_ids(edges.ravel()),
            trace_index=np.repeat(np.arange(n), 2),
            trace_ids=np.arange(n),
            time=np.tile([0.0, 1.0], n),
            localization_ids=edges.ravel(),
            trace_column="pairs",
            time_column=None,
            columns={"delta_time": np.repeat(self.delta_time[keep], 2)},
        )
