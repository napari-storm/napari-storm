# Post-processing

Open **Post-proc.** after loading localizations. Select the dataset at the top.
The workflow is **Fiducials → Grouping → Drift correction → Explore**; the first
two steps are optional.

## Installation

COMET is optional. It is published on PyPI as
[comet-smlm](https://pypi.org/project/comet-smlm/); install it with napari-storm's
extra:

```shell
pip install 'napari-storm[comet]'
```

or directly with `pip install comet-smlm`. Earlier COMET releases were called
`py-comet` (up to 1.1) and install the same `comet` package: uninstall
`py-comet` first, so the two do not overwrite each other's files. The
integration runs COMET's CPU backend. Importing napari-storm does not import
COMET, torch or numba. Grouping, loading and saving drift, Undo, and Explore
work without COMET installed.

## Correcting drift

1. Set **Max drift [nm]** and **Localizations per time window**. Target sigma
   is fixed at **10 nm**. COMET's original count-based segmentation keeps whole
   frames together and handles the final window using its existing policy.
   There is no additional minimum population: 10 localizations per window is
   valid. The integration does not merge sparse windows, sample localizations,
   or override COMET's initial sigma, smoothing or optimizer defaults.
   STORM uses acquisition frames. Group-mean input counts means per window;
   MINFLUX counts trace means ordered by their measured mean time and applies
   one correction value per trace.
2. **Estimate memory** counts pairs exactly for the selected input. Filters and
   the render-range crop select the estimation region. Fiducials and invalid
   positions are excluded. The estimate never changes the input or settings.
3. **Run COMET** runs in a background worker. Progress appears in the tab;
   **Cancel** takes effect at the next COMET callback. Pair search itself cannot
   currently be interrupted. Closing a dataset rejects its outstanding result.
4. The plot shows the x/y/z correction, anchored to zero at the first acquisition
   frame in the estimation input. All rows receive the drift,
   including rows outside the estimation selection. The knots sit at the mean
   time of each window's localizations; between the first and last frame of the
   estimate's data the spline is used, as COMET does, and outside that range the
   drift is held at the nearer end. File Infos reports how many rows that is.

Correction changes the dataset coordinates. **Undo** restores the original
positions exactly, **Re-apply** applies the retained model again, and **Discard**
restores the original positions and releases the model. Position, frame/time,
trace-id and pixel-size edits remain locked until Discard. Another Run estimates
from the retained original coordinates, not already-corrected positions.

Full-field ranges widen to include raw and corrected data. Intentional crops
keep their absolute bounds, so corrected points may move outside them. Existing
parameter filters retain their row selection.

## Memory and performance

The guard adds the run's predicted allocations to the process's current
resident memory and a reserve (1 GB or 10 % of RAM, whichever is larger). Over
that, **Run** asks before going ahead; when the run alone would not fit in
physical memory, it refuses. Reduce the estimation region or the maximum drift and estimate again. This reduces risk; running in the same process cannot guarantee
that the operating system will not terminate it.

COMET 1.2 retains the tested `query_pairs` search: int64 pairs and their
int32 copies coexist at a peak of roughly **24 bytes per pair**. The guard
also includes coordinate copies, dense per-frame interpolation, a fixed working
allowance and the reserve above. These are memory safeguards, not scientific
population thresholds. `python scripts/benchmark_comet_memory.py` measures a
dense pair search on your machine.

## Fiducials and grouping

Detect uses original coordinates and estimates the field bounds from the data.
Review the ticked candidates before excluding them. Discs extend through the
full z range. Restore clears only fiducial exclusions: invalid coordinates remain
excluded, and ordinary filters still apply.

STORM grouping links observations by distance with at most one observation per
frame per group. One allowed dark frame permits a link across a frame difference
of two. A duration limit closes persistent groups. Group ids are available through
**Connect traces** and **Group means** as COMET input. Each group contributes one weighted mean before COMET forms count-based
windows. Groups are not exported as a separate dataset.

MINFLUX uses its existing trace ids. Rank is the canonical correction clock.
Transfer to another dataset requires finite, strictly increasing mean trace times;
float32 timestamp ties disable transfer rather than inventing a clock.

## Exploring a correction

The slider blends the display in 10% steps. **The data stays fully corrected**:
exports always use the applied state, even while the screen shows 60%. Playback
sweeps from raw to corrected. For 2D data, the time view draws acquisition time
as height; its scale is labelled and switching it off restores the camera.

Pair search uses corrected coordinates and draws those same row-id edges while
the slider moves. The region shrinks to a centred square when needed to keep
at most 200,000 edges. Chains select one nearest later-window partner. Search
uses a single worker and replaces any pending request with the latest settings.
Deleting the pair layer leaves the real trace layer alone.

Pairs are **not an independent accuracy check**: they were selected to be close
on corrected data. Radius defaults to 3.5 times the lateral uncertainty in 2D,
4 times in 3D, with uncertainty-scaled z where available.

## Saving and sharing

**Save drift** writes the window knots, anchor, interpolation, the frame range
of the estimate, provenance and any MINFLUX rank-to-time mapping to HDF5.
**Load drift** works without COMET and refuses a drift whose time range the
dataset does not reach. A correction-details file written by COMET itself has
no clock; it is read with the selected dataset's clock, and windows COMET's
quality control flagged (NaN) are skipped as COMET skips them.

**Apply to other datasets** asks you to confirm simultaneous acquisition on the
same clock, checks range and clock compatibility, and accounts for world scale.
The datasets share Undo, Re-apply, Discard and preview fraction.

Both `.ns` save routes write applied coordinates, drop excluded rows, and retain
correction provenance. A saved corrected dataset does not contain the original
position stash, so reopening it does not recreate the interactive Undo history.

## Embedding

```python
import numpy as np
from napari_storm.core import DatasetTraits, GaussianSettings, RenderPlanner
from napari_storm.postprocessing.comet_runner import CometParameters, run_comet

# table is a LocalizationTable; the frame column contains acquisition frames.
coords = np.column_stack([table.coordinate_nm(a) for a in ('x', 'y', 'z')])
frames = table.column('frame_number')
model = run_comet(coords, frames, CometParameters(window=60))
table.apply_position_delta(model.evaluate(frames))
request = RenderPlanner().plan(table, GaussianSettings(),
                              DatasetTraits(zdim_present=True), name='corrected')
renderer.update(dataset_id, request)
# Exact restoration remains available until the stash is discarded.
table.restore_positions()
```

The model and algorithms use NumPy/SciPy without Qt or napari. Display offsets
and `PairSet` payloads are optional planner arguments; exporters omit the offsets.

When publishing results, cite COMET's
[method preprint](https://doi.org/10.64898/2026.03.27.714864).
