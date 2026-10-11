"""Measure incremental COMET pair-search RSS in a fresh process.

Run with the COMET 1.2 environment. The fixture puts every point at the same x,
a dense geometry that stresses pair-search allocations. COMET 1.2 retains
1.1's query_pairs search (24 bytes per pair at the conversion peak).
"""

import json
import resource
import sys
import time

import numpy as np
import psutil
from comet import __version__
from comet.core.pair_indices import pair_indices_kdtree

rng = np.random.default_rng(12)
coords = rng.normal(size=(2000, 3))
coords[:, 0] = 0
baseline = psutil.Process().memory_info().rss
start = time.monotonic()
a, b, success = pair_indices_kdtree(coords, 100.0)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    peak *= 1024
print(
    json.dumps(
        {
            "version": __version__,
            "fixture": "2000 points, identical x",
            "pairs": len(a),
            "success": success,
            "seconds": time.monotonic() - start,
            "baseline_rss": baseline,
            "peak_rss": peak,
            "increment_bytes_per_pair": (peak - baseline) / max(1, len(a)),
        },
        indent=2,
    )
)
