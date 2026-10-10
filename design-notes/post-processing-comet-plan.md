> Implementation update (2026-10-04): the user superseded the scientific controls
> in this draft. The adapter uses original COMET 1.x localization-count
> segmentation, maximum drift, and fixed target sigma 10 nm. No custom sparse-window
> merging, minimum population, automatic sampling, or smoothing override is applied.
> CPU execution is supported. Drift is anchored at the first acquisition frame,
> without averaging an arbitrary fraction of windows. COMET 1.2 retained its
> original pair search; budget its 24 B/pair peak. See docs/post-processing.md
> for current behavior. Historical design alternatives below are not requirements.

# Post-processing and COMET drift correction — plan for 3.2

Status: **v5, accepted for implementation** — the maintainer's decisions of
4 October recorded, six review rounds (Appendix B lists what each round
changed). Written against napari-storm 3.1.0
(`526c42c`) and py-comet 1.1.0.

What is asked for:

1. a **Post-processing** tab: grouping, fiducial detection and removal (mainly
   as preparation for COMET, e.g. for data already corrected on beads), COMET
   drift correction;
2. after COMET, an optional pair search at the target sigma, drawn as vectors
   between the paired localizations on the *uncorrected* data — the network of
   localizations that presumably belong together;
3. a slider that blends the drift from raw to corrected in 10 % steps, in 3D,
   with the pair vectors following, using the track machinery.

All three fit the existing machinery. What the measurements (Appendix A) say:

* **Memory is the constraint, not CPU time.** py-comet 1.1's numba CPU kernel
  costs ~4.6 ns per pair per evaluation (COMET's measurement on an M-series
  Mac; our M2 run agrees), and a run took 86 evaluations at 56–205 windows:
  62 M pairs in ~36 s. But 1.1's pair search peaks at ~24 B per pair in the
  process that runs it, and COMET's 2D test set (1 M localizations) has 173 M
  pairs at its README's 300 nm max drift — ~4.3 GB. One test run was killed
  and one spent most of its time swapping on 8 GB. py-comet 1.2 (built in
  phase 0, §9) holds ~8 B per pair and no per-evaluation copies: measured on
  62 M pairs, peak 1.61 → 0.69 GB and optimisation 43 → 19 s; the 300 nm run
  would need ~1.7 GB. Larger data still meets the same wall, so the tab's
  estimate-and-guard before Run is essential.
* **On non-NVIDIA hardware the CPU backend is the one to recommend.** COMET's
  own measurement has torch on Apple MPS ~6× slower than its CPU kernel per
  evaluation; with 1.2 the whole 62 M-pair run took 415 s on MPS (138
  evaluations, float32) against 19 s on the CPU (86 evaluations) — 22×. ROCm (AMD) torch
  reports itself as CUDA, so COMET would auto-select torch there; that path is
  unmeasured.
* **More windows is not free.** ~2 100 windows needed 381 evaluations (113 s)
  against 86 at 56–205 windows; the finest σ step alone took 206 iterations.
* **The pair network is drawable.** At a few target sigmas there are 14–92×
  fewer pairs than in COMET's set (≈ 17× at the default radius in 2D, 24× in
  3D), but still millions; a complete network of a region, capped at 200 k
  vectors, costs ~0.3 s per slider step.

---

## 1. Decisions

The answers of 4 October are recorded. Rows marked *recommended* had no answer
and stand unless they are overruled.

| # | Decision | Status |
|---|---|---|
| D1 | Release **py-comet 1.2** first (§9); 3.2 requires it and waits for it. The module-wrapping shim of §9.3 is for development only. | decided |
| D2 | `comet_fiducials` ships in py-comet 1.2. | decided |
| D3 | COMET is optional: `pip install "napari-storm[comet]"` enables Fiducials and running COMET. Grouping, Load / Save drift, Undo / Re-apply / Discard and Explore work without it. | recommended |
| D4 | The dataset **holds the corrected positions** while a drift is applied (exports, filters and `.ns` see them); the raw positions are kept, so Undo is exact. The slider is a **viewing aid** and never changes the data. No duplicate "original" dataset. | recommended |
| D5 | Fiducial removal is an **exclusion mask** with a reason per row, kept apart from the user's selection: reversible, row ids unchanged, no filter can bring the rows back. | recommended |
| D6 | Exports write the applied state (D4), whatever the slider shows. | recommended |
| D7 | Pair vectors through the track machinery: a second, parametrised `TraceOverlay` fed by a `pairs` payload on the render request. | recommended |
| D8 | Grouping in 3.2 as the last feature phase, the only one allowed to slip to 3.3: group ids and group means as COMET input; merged localizations as a dataset of their own are deferred. | recommended |
| D9 | The drift's mean is arbitrary, so it is **anchored at the start of the acquisition** (the mean of the first ~2 % of windows, at least one). The anchor belongs to the drift, not to a dataset: every dataset sharing it — the channels of a multi-channel acquisition — moves identically. | decided |
| D10 | Workflow order **Fiducials → Grouping → Drift correction → Explore**; Fiducials and Grouping are optional. | decided |
| D11 | **No z weighting** in COMET. | decided |
| D12 | **MINFLUX: one COMET point per trace** (its mean) and **one drift value per trace**. Only one trace is measured at a time, so trace order is time order; drift within a trace is not estimated — on those time scales it would fight the localization uncertainty and pull every trace into a point. | decided |
| D13 | Pair radius from the **localization uncertainty**: 3.5 σ in 2D, 4 σ in 3D, anisotropic, the target σ only as a fallback — *decided*. Default view: all cross-window pairs, complete within a region (cap 200 k); chains as an option — *recommended*. | partly decided |
| D14 | An **(x, y, time) view** for 2D data in Explore (§6.4). | decided |

---

## 2. The tab, top to bottom

Built like the Decorators tab: section headers, `QHSeperationLine`, grey
word-wrapped notes, inside a `VerticalScrollArea`. Added to the tab bar once,
at the first load, after the existing tabs (`show_avaiable_widgets` re-adds
tabs on every load, which would pull a detached tab back mid-job). The label
may get elided in the 420 px dock next to five other tabs; "Post-proc." is the
fallback. Numbers below are COMET's 2D test set (1 077 377 localizations,
13 651 frames, 12 286 of them populated) with max drift 120 nm and target σ
5 nm as the user's choice, measured or derived from Appendix A; memory figures
assume py-comet 1.2. The bead and group lines are placeholders.

```
Dataset [ test_dataset ▾ ]
Status:  drift applied · anchored at start · 205 windows · 62 M pairs · cpu · 37 s
         no rows outside the estimated range · no fiducials excluded
───────────────────────────────────────────────────────────────
1  Fiducials                                          (needs py-comet)
   Max drift [nm]  [ 120 ]            (shared with section 3)
   [ Detect ]   → <n> beads, <m> to review
   ☑ bead 1  (x, y µm)  r <nm>  coverage <0–1>  high
   ☐ bead 2  ...                                review
   [ Exclude ticked ]  [ Restore ]
   grey: discs cover the full z range in 3D data · <k> localizations
───────────────────────────────────────────────────────────────
2  Grouping                                    (STORM; MINFLUX has traces)
   Max distance [nm] [ 30 ]   Max dark frames [ 1 ]   Max frames [ 50 ]
   [ Group ]  → <g> groups (use "Connect traces" to see them)
───────────────────────────────────────────────────────────────
3  Drift correction (COMET)                           (needs py-comet)
   Backend  [ cpu ▾ ]  ⓘ no NVIDIA GPU (describe_backends())
   Estimate from [ localizations ▾ ]  (group means / trace means)
      N = 1 077 377 — filters and render-range crop applied, fiducials excluded
   Window    [ 60 ] frames  → 205 windows, ≈ 5 300 localizations each
   Max drift [ 120 ] nm     largest distance the sample moved during the run
   Target σ  [ 5 ] nm       ≈ localization precision (FWHM 11.8 nm)
   ▸ Advanced: initial σ, smoothing, cap per window
   Before running:  62 M pairs · ≈ 0.8 GB peak · cpu ≈ 20 – 45 s
      (at the default 300 nm: 173 M pairs · ≈ 1.7 GB · cpu ≈ 1–2 min)
   [ Run ]  ▓▓▓▓▓▓░░░░ step 5 of ~7 (≤ 11) · σ 7.9 nm · ETA 12 s   [ Cancel ]
   ┌ drift plot: dx, dy, dz, knots marked, clamped ends dashed ──┐
   └─────────────────────────────────────────────────────────────┘
   [ Undo ] / [ Re-apply ]   [ Discard ]   [ Save drift… ]   [ Load drift… ]
   [ Apply to other datasets… ]   [ Save corrected localizations… ]
───────────────────────────────────────────────────────────────
4  Explore                                         (once a drift is applied)
   Correction   raw ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤ corrected   100 %
                [ ▶ Play ]
   ☐ x, y, time view (2D data)    1 µm of height = 650 frames
   ☐ Show pair network    radius [ 17.5 ] nm  (3.5 × σ, 95 % of pairs)
     [ all pairs in the crop ▾ ]  (chains: one partner each, larger region)
     → 3.5 M cross-window pairs; showing ~200 k in a ~9 µm square at the
       crop centre (cap 200 k)
     close pairs within 17.5 nm: raw 2.85 M → corrected 3.60 M
```

State outside the tab: the status line also appears in File Infos (which gets
a per-card refresh), and the dataset's Channel Controls card carries a small
badge while the slider is not at 100 %. Data Filter shows a grey line
"N localizations excluded as fiducials", since its "Reset all filtering"
deliberately does not bring them back.

Locking: during a COMET run, Fiducials/Grouping/Run/Undo and Data
adjustment's position and time edits are disabled for that dataset; unloading
it or clearing the session cancels the run. During Play, Run, Undo and Exclude
are disabled; the view can still be rotated.

---

## 3. Architecture

```
napari_storm/postprocessing/          host-free: numpy, scipy; comet imported lazily
    drift.py          DriftModel: knots, time mapping, anchor, provenance;
                      evaluate(times) -> (N,3) nm
    comet_runner.py   run_comet(locs_nm, times, params, progress, cancel) -> DriftModel
    pair_budget.py    pair counts (exact and sampled), memory and time prediction
    pair_network.py   find_pairs(...) -> core PairSet
    grouping.py       link_localizations(...) -> group ids; group_means(...)
    fiducials.py      adapter over comet_fiducials (nm, bounds, bead tracks)

napari_storm/pyqt/postprocessing_tab.py   PostProcessingWindow (widgets)
napari_storm/PostProcessing.py            PostProcessingInterface (state, jobs)
```

* `postprocessing/` is outside `core/`, whose docstring promises numpy and the
  standard library only; it is added to `test_core_is_host_free.py` (which
  blocks Qt, napari and vispy, not scipy).
* `core/` gets small, general additions (§4): exclusion reasons, a position
  stash with a guarded write path, a display offset in the planner, a
  `PairSet` type and a `pairs` payload on the render request with
  `Changed.PAIRS`, a drift view in `DatasetState`, and (phase 5) side columns.
* The tab keys its state by `dataset_id` through store events, not combo
  positions. `close_session` unsubscribes listeners before clearing the store,
  so the tab cancels its job and its Play timer there explicitly; a job's
  result is discarded unless `store.get(id) is dataset` still holds.

---

## 4. Data model

### 4.1 The drift

`DriftModel` (host-free) holds what COMET estimated and how to evaluate it:

* **Knots**: the centre time and drift of every COMET window (from the run
  details, §9.1), and the interpolation method. The dense per-frame table is
  derived lazily and cached; it is not stored.
* **Time mapping**: STORM — the frame column (`frame_number`) and its offset
  (frames may be 0- or 1-based); spans above ~10⁷ frames are refused.
  MINFLUX — one COMET frame per trace, in trace order, which is time order
  (D12): the correction lives in the **rank domain**, where COMET fitted it.
  Where the format has a time column, the model also stores the source's
  rank → mean-time table (`time_s` cast to float64 first). It is used only to
  map *another* dataset's trace into the source's rank domain: its mean time →
  a fractional rank by linear interpolation in that table, clamped at the
  ends. That needs the table's mean times to be finite and **strictly
  increasing**, which sequential acquisition implies but float32 storage does
  not guarantee (two distinct times can round to the same value); the table is
  checked when the drift is made, and if it fails, the source keeps its
  rank-domain correction and transfer is disabled with the reason. Relabelling
  the knots with times and interpolating in time would change the correction
  when trace durations vary, so it is never done. MINFLUX v1-base, which has no
  time column, keeps ranks only and cannot be transferred.
* **Evaluation per row**, done here rather than with COMET's `drift[frame]`
  (which raises past the last frame, wraps for negative ones and truncates
  non-integers): COMET's interpolation from the knots inside the range of
  frames COMET was given (`RunDetails.frame_range`), the value at the nearer
  end outside it — the rule py-comet 1.2 itself uses (§9; 1.1's cubic spline
  extrapolated, 2.7·10⁷ nm in one review probe, §A.4) — with the number of
  affected rows (rows of other datasets, or rows outside the estimate's crop)
  in the status line.
* **MINFLUX** is evaluated once per trace, at its rank — its own rank for the
  source dataset, the mapped fractional rank for any other — and that value is
  applied to all of the trace's localizations (D12). A test checks that
  applying the source's drift to the source through time → rank reproduces its
  rank-domain correction exactly.
* **Anchor** (D9) applied in `evaluate`, so everything downstream — every
  dataset in the drift group included — sees one drift, written **d** below.
* **Provenance**: COMET version, parameters, backend, pair count, accepted σ,
  steps, evaluations, timings, and an input fingerprint: row count, time
  range, a hash of the raw positions, and for MINFLUX a hash of the trace
  membership (the resolved trace-id column) and of the rank → time table.

`DatasetState.drift` holds a small view: the model, `applied`, the preview
fraction `alpha`, and the drift group (the datasets sharing this model). It is
set through `store.set_drift(...)`, which emits `DriftChanged`.

### 4.2 Applying, undoing, re-running

* **Stash.** On the first Apply the table copies its three position columns
  (12 B per localization). `apply_position_delta(delta_nm)` writes
  `stash − delta / position_scale_nm` per axis; `restore_positions()` writes
  the stash back bit for bit. Only `zdim_present` decides whether z is
  written (2D STORM carries a z column of ones).
* **Lifecycle.** A drift and its stash live from the first Apply until Discard:

  | State | Positions | Writes to positions, time, pixel size, records | Enabled |
  |---|---|---|---|
  | No drift | raw | allowed | Run, Load drift |
  | Applied | raw − d | refused | Undo, Discard, Run, slider, pairs |
  | Undone | raw, from the stash | **refused** | Re-apply, Discard, Run |

  Run, from any state, estimates from the raw positions (the stash, or the
  table when there is none) and ends in Applied with the new drift. Discard
  restores raw and drops drift and stash. Writes stay refused while Undone
  because Re-apply writes from the stash, so an edit made in between would be
  overwritten; the refusal names Discard as the way out. Every transition is a
  test (§11).
* **Guard.** While a stash exists, and during a COMET run on the dataset,
  the public `set_column` and `adjust_column` refuse position and time columns
  and the resolved trace-id column (for MINFLUX the trace decides which drift
  value a row receives, D12); `set_records` refuses, the `locs_all` setter
  refuses once a dataset has an id, and the STORM pixel-size setter refuses.
  Data adjustment gets an error path that shows the reason.
* **Render range** on Apply, Undo, Re-apply and Discard, per axis, on the
  range all loaded datasets share:
  * **full field** (that axis's range slider at 0–100 %): widen the range to
    the envelope of all datasets' raw and corrected extents, finite rows only,
    and keep 0–100 %, so nothing is clipped;
  * **an intentional crop**: keep its absolute bounds (the percentages are
    re-expressed against the widened range). Corrected points that cross them
    are clipped — that is what a crop means — and a grey note gives the count.

  Either way the camera does not move; then `refresh_dataset`. Without this,
  `update_data_range` crops to the stored range first, and points moved past
  the old extent disappear (a review probe lost 100 of 1 001 points to a
  100 nm shift, §A.4); `set_render_range_and_offset` would fix that but
  re-centres the camera. Both cases are tests, with two datasets sharing the
  range.
* Undo, Re-apply, Discard and the slider act on the whole **drift group**, so
  channels sharing a drift stay registered. Run on a group member estimates
  for that dataset only and takes it out of the group; the button says so.
* **Filters** applied before the correction keep the rows they removed (Data
  Filter stores row ids, not bands — that is what a filter means today); a
  grey note says so when such filters exist.

### 4.3 The slider is a display offset

`RenderPlanner.plan(..., position_offset=...)` takes an optional function of
row ids returning per-row displacements in nm, added before the world
transform. It is passed only by `DataToLayerInterface._render_request`, which
reads the drift view from `DatasetState` on **every** plan — filter Apply,
trace-style replans and appearance updates re-plan too, and must not snap a
60 % view back to 100 %. Exports call `plan` directly without it, and the
render-range bookkeeping (`get_coords_from_locs`, `get_coords_from_all_locs`)
never sees it.

* With the table holding `raw − d`, the offset is `(1 − α) · d`: 100 % shows
  the corrected data, 0 % shows `raw − d + d` = raw (to ~2 float32 ulp: one
  rounding when the table is written, one when the offset is added), and every
  frame in between is the linear blend `raw − α · d`. The slider is enabled
  only while the drift is applied.
* An α-only `DriftChanged` takes a **positions-only refresh**: no
  `update_data_range`, `changed = POSITIONS | TRACES | PAIRS`. The crop,
  filters and display subsample therefore cannot change during a sweep —
  nothing pops in or out at a crop edge, nothing sparkles above the display
  budget. (Today's `extend_range=False` refresh sends `SIGMAS | VALUES`, which
  the trace overlay ignores.)
* Z colour coding reads the table's z; the offset is applied there too, so
  colour follows the displayed depth.
* Tooltip: 0 % shows the raw positions of the rows selected under the current
  (corrected) crop and filters — at crop edges that is not exactly what Undo
  would show.
* The x, y, time view (§6.4) adds a z component to the same offset.

### 4.4 Exclusion mask

The table keeps the **user selection** — what filters, the render-range crop
and resets produce; today's `filter_mask` — apart from `excluded`, one byte per
row with reason bits (`FIDUCIAL`, `NON_FINITE`). The effective selection is
`selection & (excluded == 0)` — not `selection & ~excluded`, which on a byte
array is a bitwise complement and lets a row carrying only one of the bits
through — and `filter_mask` returns it; the display limit,
exports and planning read it. Every mask write goes through `set_filter_mask`
(reset, apply_filters, bandpass, keep_values, deactivate, restrict by percent,
the photon filter) and sets the user selection; operations that narrow the
current selection read the user selection, not the effective one, so excluded
rows keep their place in it. **Restore** clears the fiducial bit, and the rows
come back exactly where the user selection has them; a row with both bits
stays out. `set_records` (the embedders' live-append path) pads both masks
when the new records extend the old ones and resets them otherwise. Tested:
Exclude → change filters → Restore; each reason bit on its own; rows carrying
both bits.

**Non-finite positions** are a 3.1 bug of their own, fixed first in 3.1.x. One
NaN makes the render range `[nan, nan]` at load (`get_coords_from_all_locs`
reads every row and `set_render_range` takes plain min/max), and "Reset render
range" then leaves only the NaN row active. The fix needs both halves:
non-finite rows stay out of the selection through every mask write, and every
bounds computation — at load, on Reset render range, and later for the
correction envelope — uses finite rows only. A dataset without a single finite
position is refused at load, with a message. In 3.2 the `NON_FINITE` bit
replaces the 3.1.x mechanism.

### 4.5 Side columns (phase 5)

Group ids need a column without rebuilding the records (`set_records` resets
the filter mask, doubles peak memory, and STORM classes assert a fixed dtype).
The table gets **side columns**: named arrays that `has_field` and `column`
consult; `field_names` stays records-only and `side_column_names` lists them;
`.ns` does not write them. Grouping writes `group_id` (−1 for none) and sets
the dataset's `trace_column` to it, then re-syncs the trace controls, so
"Connect traces" shows groups. That is an explicit choice, consistent with
`core/traces.py` refusing to treat a column called `group` as a trace by
default (in Picasso it is a pick id).

---

## 5. COMET drift correction

### 5.1 Input

* **Localizations (STORM)**: the filtered rows minus excluded ones (filters and
  the render-range crop apply — the crop is an implicit region of interest and
  the tab says so), in nm, z = 0 for 2D.
* **Group means (STORM)**: one point per group at its precision-weighted mean
  position and its rounded mean frame. The tab segments before COMET (§5.2),
  so the window boundaries are known, and groups are **split at them** first —
  a two-frame group straddling a boundary becomes two points — and then at a
  maximum duration. Fewer points and far fewer same-molecule pairs within a
  window, which cost COMET memory and time and carry no drift information
  (pairs across a window boundary do, and averaging loses them; worth
  measuring before group means become a default). Never Picasso's `group` (a
  pick id spanning the acquisition: its mean averages the drift away).
* **Trace means (MINFLUX, D12)**: one point per trace, one COMET frame per
  trace, windows of whole traces — a trace is never split — and one drift
  value per trace. Raw MINFLUX localizations are not offered as COMET input.
* The drift is applied to **all** rows, including rows that were not in the
  estimate: STORM by frame, MINFLUX by trace.

### 5.2 Parameters

| Control | STORM default | MINFLUX default | Source |
|---|---|---|---|
| Estimate from | localizations | trace means, the only choice (D12) | COMET's MINFLUX batch script |
| Window | **60 frames**, enlarged so each holds ≥ 200 localizations | **50 traces** (one frame per trace) | README quickstart (mode 2, 60 frames); docs (~200 per segment); batch script. Measured: ~2 100 windows cost 4.4× the evaluations of 205 (§A.4). |
| Max drift | 300 nm; lowered only if no cap per window fits the pair budget | 100 nm | README; batch script |
| Target σ | 10 nm, or the median localization precision when the data has one | 10 nm | README; batch script. Shown with its FWHM — every other width in the dock is a FWHM. |
| Initial σ | max drift // 3 | same | COMET default |
| Smoothing (boxcar) | 1 | 3 | docs; batch script |
| Cap per window | off (integer only) unless the budget needs it | off | a fractional cap raises in COMET's mode 2 |
| Backend | `best_backend()` | same | MPS selectable, labelled slower than CPU and memory-heavy |

**Windows.** For STORM data (localizations or group means) the window is a
**duration** in frames and stays fixed when the input type changes. COMET's
mode 2 counts only frames that contain localizations (12 286 of 13 651 in the
test set), so the tab computes the windows itself, by true duration, and
passes them pre-segmented — which needs 1.2's fix of the pre-segmented path
(§9.2); through the 1.1 shim it falls back to mode 2. For MINFLUX trace means
the window is a count (50 traces, mode 2 over one frame per trace). The
derived numbers are shown ("→ 205 windows, ≈ 5 300 localizations each"). COMET's own
defaults disagree with each other (function: target 1 nm; CLI: 30; docs: 1–5;
README and batch script: 10), so every default cites its source. COMET's QC
modes are not exposed (experimental, GPU-only, they write NaN for flagged
windows).

**Pair budget** = the memory the guard (§5.3) leaves for the pair-search and
optimisation stages ÷ the predicted bytes per pair for the chosen backend. It is not a time limit: time
is shown, not capped. When the parameters exceed it, the tab pre-selects a cap
per window that fits (pairs fall roughly with its square; the cap is applied by
the tab with a seeded generator, so the count stays exact), and lowers max
drift only if no cap fits. Both are shown before Run, with the alternatives
and their computed pair counts.

### 5.3 Before Run: estimate and guard

* **Pair count**: exact at Run, in the worker, with
  `cKDTree.count_neighbors(tree, r)` (= 2P + N, no allocation; 0.28 s for
  28.7 M pairs) on exactly the points COMET will pair (any cap is applied by
  the tab beforehand). While parameters are edited, a debounced sampled count
  on a subsample, scaled. (COMET's `estimate_pairs` under-counts by up to 2×.)
* **Memory** (1.2 measured in phase 0, §A.4; 1.1 from the code, §A.1):

  | | py-comet 1.2 (shipped) | py-comet 1.1 (shim only) |
  |---|---|---|
  | pair search, peak | 8 B/pair + ~0.2 GB (one slab), measured | ~24 B/pair |
  | CPU optimisation | the 8 B/pair already held; ~0 per evaluation, measured | 8 B/pair + 16 B/pair per evaluation |
  | torch | 8 B/pair (int32) + chunks ≤ 1 GB | 16 B/pair + ~150 B/pair transient (system RAM on MPS) |
  | CUDA (device) | 8 B/pair + a scratch buffer ≤ chunk (not run on hardware yet) | 8 B/pair + 800 MB |
  | per localization | ~60 B, plus the caller's copy (32 B) | ~120 B |

* **Guard.** It reduces the risk of the process being killed; it cannot rule
  it out while COMET runs inside napari. Before Run the tab predicts the peak
  of each stage — input copy, segmentation, pair search, optimisation,
  interpolation, apply — as an increment over the process's current resident
  memory (`psutil`, declared as a dependency; it already comes with napari).
  That figure already holds the loaded datasets, render buffers, the stash,
  any pair overlay and napari itself. Current + largest increment + a reserve
  (1 GB or 10 % of RAM, whichever is larger) must stay below physical RAM:
  over that, Run asks; when the increment alone exceeds physical RAM, it
  refuses. macOS "available" memory is misleading under compression, so it is
  not used. GPU backends get a second check against free device memory
  (`cuda.current_context().get_memory_info()`, `torch.cuda.mem_get_info()`);
  MPS shares system RAM and uses the first. The CPU constants above were
  measured on the 1.2 build (phase 0); torch and CUDA still come from the code
  and are to be measured on a CUDA machine before the guard relies on them.
* **Time** ≈ evaluations × pairs × per-backend cost + pairs × ~60 ns (1.2's
  pair search: an exact count, then the slabs). Evaluations = steps ×
  evaluations per step: steps from the expected ceil(log(σ0 / target) /
  log 1.5) + 1 up to K (§5.4); 12–15 per step up to a few hundred windows, up
  to ~55 near 2 000 (§A.4). The CPU cost is ~3.6 ns per pair per evaluation on
  the M2 with 1.2 (19.3 s for 86 × 62 M), with no overhead left to model — in
  1.1 a third of the time was the per-evaluation copies. Replaced by a live ETA
  after the first step.

### 5.4 Running it

* **Non-modal**: an inline progress bar and Cancel in the tab, not
  `load_in_background`'s window-modal dialog, which would freeze napari for the
  whole run. Same `_thread_worker` underneath; one COMET job at a time (numba's
  threading layer is not re-entrant).
* **In a thread**: the numba kernels are `nogil`, torch and CUDA release the
  GIL, `query_pairs` mostly does. A subprocess would duplicate the data and
  complicate frozen hosts.
* **Progress** is posted by the worker into a thread-safe holder that a QTimer
  reads; `run_on_main_thread` blocks the worker per call, and napari waits for
  workers without a timeout on quit.
* **Stages shown**: "Loading COMET…" (imports torch if installed, probes CUDA),
  "Compiling kernels (first run only)…", "Windows…", "Pair search (62 M
  pairs)…", "Step k of ~E (≤ K) · σ …", "Interpolating".
* **Steps, not σ**: COMET keeps refining below the target while updates shrink
  and returns the previous step's estimate, so the accepted σ lies between
  ~1 nm and 1.5 × target. The expected number of steps is E =
  ceil(log(σ0 / target) / log 1.5) + 1 (7 measured at σ0 40 nm, target 5 nm);
  at most K = ceil(log(σ0 / 1 nm) / log 1.5) + 1 (11 at max drift 120 nm, 13 at
  300 nm), more if L-BFGS-B fails and retries. Progress shows "step k of ~E
  (≤ K)".
* **Cancel** takes effect at the next cost evaluation, except during the pair
  search (one call). The worker then drops the exception and its traceback
  (it pins the pair arrays), collects, and empties torch / numba-CUDA caches.
* **A cancelled or failed run never changes the dataset.** COMET gets a copy
  (it corrects its input in place).
* COMET is never allowed near the UI: `display=False` (it calls `plt.show()`),
  no save path left empty (that opens a Tk dialog), never `interactive`.

### 5.5 Result

* **Drift plot** of the anchored drift over the estimated range: dx, dy, dz
  against frame or time, window knots marked, clamped ends dashed with their
  row counts, the run's parameters and the COMET citation. Bead tracks, when
  beads were detected, overlaid as an independent check (§7).
* **Save drift…**: HDF5 in COMET's correction-details layout plus provenance
  and, for MINFLUX, the rank → time table.
  **Load drift…** (works without py-comet) checks the time mapping and range
  against the dataset and refuses a STORM (frame) drift for MINFLUX (trace)
  data and vice versa.
* **Apply to other datasets…**: datasets of the same kind whose time range
  lies inside the drift's — STORM by frame, MINFLUX by the traces' mean times
  (so it needs a time column) — after an explicit "recorded simultaneously,
  same clock" confirmation (a restarted frame clock would be mis-corrected
  silently). If their world-transform scales differ, the drift is mapped
  through the scale ratio. They join the drift group and share its anchor
  (D9), so the channels move identically.
* **Save corrected localizations…**: `.ns` with the applied correction,
  excluded rows dropped, drift knots and provenance as attributes. (Today's
  `.ns` export writes every row with no provenance.) There is no scene entry
  in 3.2: scenes match datasets by name and do not reload files yet; when they
  do, the knots are small enough to embed.

### 5.6 Failures

| Situation | What the user sees | Offered |
|---|---|---|
| py-comet missing | Fiducials disabled; in Drift only Backend, estimate and Run, with `pip install "napari-storm[comet]"` — Load / Save drift, Undo / Re-apply / Discard stay | link to install docs (GPU notes) |
| No CUDA | backend list without cuda; `describe_backends()` reason as tooltip | — |
| CUDA / MPS out of memory | "ran out of GPU memory at N pairs" | Retry on CPU · cap per window · smaller max drift |
| Over the memory budget | the guard (§5.3) | the ranked alternatives |
| L-BFGS-B failed after retries | COMET's message in words | larger max drift · longer windows |
| Fewer than 2 usable windows | "window longer than the data" | the window that would give 2+ |
| A STORM file without frames (one distinct frame), or a table with neither frames nor trace ids | section disabled, reason given (MINFLUX v1-base, with trace ids but no time, is corrected by trace order) | — |
| Cancelled | "cancelled, nothing changed" | — |
| Fiducials never checked, pair count dominated by hot spots | a grey warning above Run | Detect fiducials |

---

## 6. Explore: slider and pair network

### 6.1 Slider

* "Correction: raw … corrected", 0–100 % in 10 % steps with a % readout,
  enabled while a drift is applied, acting on the drift group. Drags are
  coalesced with a single-shot timer; the latest value wins.
* **Play**: one sweep 0 → 100 %, each step after the previous refresh has
  returned and one event-loop turn; it always ends at 100 %.
* Cost per step: splats ~0.18 s per 5 M localizations on the instanced backend
  (2.5 s on the billboard fallback, §A.2); "Connect traces" adds ~0.6 s per
  1 M MINFLUX rows (a full refresh measured 0.65 s with traces on, 0.03 s
  without); pair vectors ~1.5 s per million.
* 3D data opens in 3D; 2D data stays 2D unless the x, y, time view is on
  (§6.4). Moving positions costs the same upload in both.
* A test checks that a sweep leaves camera, render range and active rows
  unchanged.
* Later: blending on the GPU (a per-instance drift buffer and a uniform in the
  instanced shader, the same for the lines) would make it continuous at any
  size.

### 6.2 Pair network

"Show pair network" (off by default; ticking it moves the slider to 0 %,
which is the view asked for). Computed on the corrected positions of the
filtered, non-excluded rows — not COMET's possibly downsampled or averaged
input:

1. **Radius** `r = k · σ`, σ = the data's median localization precision,
   falling back to the target σ; k = 3.5 in 2D, 4 in 3D, which catches ~95 %
   of same-emitter pairs (k = 3 catches 90 % / 79 %). z is scaled by σxy/σz
   (from the σ columns) so the radius is anisotropic; with σz = 2.5 σxy an
   isotropic k = 4 would catch only 66 %. The predicted capture is shown.
   Without σ columns z is not rescaled, and the note says so.
2. **Windows**: each row belongs to the window whose centre (knot) is nearest
   in time — defined for every row, for drifts loaded from COMET's own files
   (which store only centres) too; MINFLUX rows take their trace's window.
   Rows outside the estimated range get no pairs.
3. **Cross-window pairs only** (the cost depends on μ[wᵢ] − μ[wⱼ] alone; it
   removes 2–5 % here); pairs inside one group or trace are dropped.
4. **Default (D13), all pairs, complete within a region**: the render-range
   crop, narrowed to a square centred on the crop — the largest side with at
   most the cap (200 k pairs ≈ 0.3 s per step), found by bisection — when the
   crop holds more; the label says so, and moving the crop chooses the region.
   **Option, chains**: each localization to its nearest partner in any later
   window, skipping windows without one, and not in the same group or trace;
   for ungrouped STORM data additionally Δt > max dark frames + 1 (1 when
   grouping was not run), so a blink is not linked to itself. At most N edges,
   capped by the same region rule (a chain network covers a larger region for
   the same cap), and at 0 % each molecule's localizations trace the drift
   path.
5. **Colour by Δt** between the two localizations (window gaps are a
   segmentation artefact: windows differ in duration).
6. **A sharpness number**: close pairs within r on raw vs corrected positions
   (two exact counts). It is not independent of COMET — COMET maximises a
   smooth version of it — and neither is the network: pairs are chosen on the
   corrected positions, so they are tight at 100 % by construction, and at 0 %
   each vector is d(tⱼ) − d(tᵢ) plus at most r. The docs and tooltip say so;
   the bead tracks (§7) are the independent check.

### 6.3 Drawing

* A `PairSet` type in `core/` (row ids, Δt), held per dataset by
  `DataToLayerInterface` and passed to every `plan`, so any re-plan includes
  it. `RenderPlanner.plan(..., pairs=...)` turns it into a `pairs` payload:
  two vertices per pair, coordinates from a coordinates-by-ids helper with the
  same transform, flat-z rule and display offset as the splats (so vectors stay
  registered after a channel shift and reach endpoints the display budget did
  not draw). A pair is drawn while both endpoints are filtered.
* **When the pairs are recomputed.** Membership is rebuilt when the drift is
  replaced (Run, Load drift; Discard drops the set), when the radius, mode or
  cap changes, when grouping or the exclusion mask changes, and when the
  selection changes (filters, render-range crop, Reset) — filtering an old set
  can remove pairs but never add the ones a widened crop or a released filter
  makes eligible. α changes, the x, y, time view, Undo / Re-apply (the drift
  is unchanged; the overlay is hidden while undone), camera and appearance
  changes reuse the set. The search runs in a worker (the KD-tree takes 2.7 s
  at 5 M points), **one at a time**: a request made while a search runs
  replaces the single pending request (latest settings win) and asks the
  running search to stop at its next checkpoint (after the tree build,
  between query chunks); a generation counter drops any result that is no
  longer the latest. Until the new set arrives the old one is drawn filtered
  to the current selection, with "updating…" in its label. Run cancels a
  pending pair request, and the COMET guard (§5.3) counts a running search's
  predicted peak as already allocated.
* `Changed.PAIRS` joins `EVERYTHING`; backends short-circuit when the change is
  only `TRACES | PAIRS`, so toggling pairs does not rebuild the real trace
  layer.
* `TraceOverlay` is parametrised — request field, appearance keys, layer
  suffix (" pairs"; the fixed " traces" would collide and napari would rename
  the two against each other), a fixed colour-by, its own removal callback
  (deleting the pairs layer must not switch the real traces off) — and each
  backend composes a second one. `draws_pairs` joins the renderer contract and
  `NullRenderer`, pairs join `host_bytes`, and `LayerAppearance.pairs` switches
  it (embedders capability-check it via `fields(LayerAppearance)`).
* Tracks rather than Vectors: updates cost the same up to 1 M segments and
  Tracks is faster above (§A.4); creation is ~3× slower at 100 k (1.2 vs
  0.4 s), paid once when the network is shown.
* The slider stays interactive with the network shown because of the cap;
  Play with traces on as well costs the sum (§6.1).

### 6.4 Time as z for 2D data (D14)

For 2D datasets Explore offers **x, y, time**: the viewer switches to 3D and
every localization is drawn at a height proportional to its time,
`z = FLAT_DATA_Z_NM + s · (t − t₀)`. It is a display offset like the slider's
(§4.3), z component only — data and exports are unchanged — so splats, traces
and pair vectors take it alike. At 0 % each molecule's localizations form a
column slanted and bent by the drift; at 100 % the columns are straight, and
the pair vectors run between time levels.

* **Scale** `s` (nm per frame): the time span maps to half the larger xy extent
  by default; one value for every loaded dataset, so channels stay comparable;
  shown as "1 µm of height = N frames" next to the toggle, because the scale
  bar reads nm.
* **Camera**: toggling on saves the 2D camera and switches `ndisplay` to 3 with
  an oblique view; toggling off restores both. A new load resets `ndisplay`
  from the data's dimensionality (`GUI.adjust_available_options_to_data_dimension`),
  and the toggle follows.
* **What does not follow**: the grid plane and reference images stay at the
  localization plane (z = 1 nm, the bottom of the stack); the z render-range
  slider stays hidden (there is no z data to crop); Z colour coding stays
  unavailable (traces already colour by time).
* 2D and 3D datasets cannot be loaded together today, so the toggle applies to
  the whole session.

---

## 7. Fiducials

* **Detect**: `find_fiducials(locs_nm, max_drift_nm, field_bounds_nm)` on the
  **raw** positions (from the stash when a drift is applied), x/y in nm, frame
  in column 3. The field bounds should be the camera field; napari-storm knows
  no camera size, so `field_bounds_from_data` is used with a grey note that a
  bead in an empty corner can be missed. Seconds for millions of rows
  (§A.1).
* **Review**: accepted and needs-review candidates in a tick list (centre,
  radius, temporal coverage, confidence), circles on the view (a Shapes
  layer). Rejected candidates can run to thousands; only their count is shown.
* **Exclude**: `fiducial_mask` → exclusion mask (reason `FIDUCIAL`). Discs in
  xy over all frames; in 3D they remove a full-height column, which the note
  says. **Restore** clears it.
* **Why it matters before COMET**: a bead present in every frame adds ~n²/2
  pairs — one localization per frame over 13 651 frames is ~93 M, more than the
  whole 2D test sample at 120 nm — and in data already stabilised on beads they
  pin the estimate near zero. The Drift section warns when detection was never
  run.
* **Max drift** is shared with COMET and sets the detection's bin size, which
  is also its resolution (too large merges beads, too small splits them).
  `radius_p99_nm` is a radius, so the peak-to-peak drift is about twice it:
  when the largest accepted radius is below ~½ bin, the Drift section suggests
  max drift ≈ 2 × radius plus a margin. After COMET, "Detect again" uses the
  measured peak-to-peak drift.
* **Bead tracks**: the centroid of each accepted bead's localizations per
  window gives an independent drift curve, overlaid on COMET's plot.
* Not in 3.2: the beads as the drift reference instead of removing them.

---

## 8. Grouping

* **STORM linking**, frame by frame: a localization joins an open group whose
  running precision-weighted mean is within `max distance` (anisotropic in 3D)
  and whose last frame is at most `max dark frames` back. A group takes at most
  one localization per frame; within a frame, candidates are matched greedily
  by distance. Excluded rows are skipped; `max frames` closes a group so a
  sticky emitter does not become one acquisition-long group.
* Pure numpy/scipy: a prototype took 0.84 s for 121 k localizations over
  20 k frames (~7 µs per localization), so ~35–70 s at 5–10 M — a worker job with
  progress, like COMET.
* **MINFLUX**: the trace id is the group; nothing to link.
* **Output**: the `group_id` side column (§4.5), and group means as a COMET
  input (§5.1): per-axis 1/σ²-weighted position (photons as the weight
  without σ), σ = (Σσ⁻²)^−½ (assumes a static emitter), photons summed,
  first / last / mean frame and n.
* Not in 3.2: the merged groups as a dataset of their own.

---

## 9. COMET 1.2

**Done in phase 0** on the local branch `release/1.2.0` (worktree
`../Comet-1.2`, from `master`), 10 commits, not pushed or tagged. 365 tests
pass from a wheel installed in a clean Python 3.12 venv (numpy 2.5, scipy 1.18,
numba 0.68), the suite passes with torch (MPS) and on Python 3.9, and the
package source passes the 3.6 syntax gate. The CUDA changes have not run on an
NVIDIA GPU. Two things differ from what is written below, both found while
building it:

* **Clamping at the data, not at the knots.** The per-frame drift is clamped
  outside the data's own first and last frame (`min_max_frames`, recorded as
  `RunDetails.frame_range`), not outside the window centres: clamping at the
  centres would change the correction of real localizations in the first and
  last half window. With the data's range every localization keeps exactly its
  1.1 correction and only rows with no data change. `DriftModel.evaluate`
  (§4.1) follows the same rule.
* **Pre-segmented input returns drift per window id.** With mode −1 the knots'
  frames are the ids; napari-storm, which made the windows, maps them to the
  true window centres itself.

### 9.1 The hook

```python
comet_run_kd(..., progress=None, return_details=False, random_state=None)
```

`progress(stage, info)` is called at each stage; raising from it aborts, and
the dataset is untouched before `apply`.

| stage | info |
|---|---|
| `segmentation` | n_segments, n_locs_valid, median locs per window (ticks inside the loop) |
| `pairs_start` / `pairs_done` | n_locs_valid, radius / n_pairs, auto-downsampled, max_locs_per_segment |
| `run_start` / `run_end` | run, sigma, fails / success, nit, nfev, delta, accepted |
| `evaluation` | run, sigma, n_evaluations, cost |
| `interpolation`, `apply`, `done` | — / — / details |

Details: the `SegmentationResult` (segment per localization, validity, window
start/end/centre frames), the knots (per-window drift, NaN included), accepted
and last σ, runs, evaluations, failures, pairs, backend, timings.
`random_state` makes the per-window downsampling reproducible (global
`np.random` today). `optimize_3d_chunked_better_moving_avg_kd` takes the same
`progress`.

### 9.2 Fixes that matter inside a GUI process

* **Pair search** in spatial chunks writing int32 directly: ~8 B/pair peak
  instead of ~25 — the largest single saving for the guard.
  *Tried and withdrawn (maintainer's decision): 1.2 keeps 1.1's `query_pairs`
  search, ~24 B/pair at its peak; slabs are for another time.*
* **CPU wrapper**: drop the per-evaluation int32 → int64 index casts (16 B/pair
  allocated and freed every evaluation; int32 gives bit-identical results) and
  convert coordinates and segment ids once.
* **torch**: size chunks from a memory budget (one chunk of up to 10⁸ pairs
  materialises ~150 B/pair); import it lazily (today it is imported whenever
  installed, and a broken install breaks `import comet`).
* **CUDA**: size the 800 MB scratch buffer as min(chunk, pairs).
* **Segmentation** modes 0 and 1 are O(N·S): ~3 min at 5 M localizations with
  200 per window. Vectorise.
* **Pre-segmented input** (mode −1) fails today on an unassigned `result` in
  `segmentation_and_pair_indices_wrapper`; fix it, so a caller can pass
  windows of true duration (§5.2).
* **Interpolation**: clamp outside the window centres instead of extrapolating,
  so COMET's CLI and the tab agree.
* `estimate_pairs` → `count_neighbors` (exact, no allocation).
* Ship `comet_fiducials`.

All backwards compatible.

### 9.3 Development shim (not shipped)

To build the tab before 1.2 exists: call the public `comet_run_kd` and, for the
duration of one call (one job at a time, restored in `finally`), wrap module
globals it looks up at call time — `minimize` (wrap its objective: count
evaluations, read σ from its arguments, raise on cancel; verified to cancel
cleanly on the cpu backend — the same call path serves cuda and torch, untried
here — and used for the measurements in §A.4),
`segmentation_and_pair_indices_wrapper` (capture the segmentation),
`pair_indices_kdtree` (count before allocating) and `interpolate_drift`
(capture the knots). It binds to names outside `comet.__all__` and gets none of
§9.2's memory fixes.

---

## 10. Embedders

Host-free functions only in 3.2: `run_comet` (a plain progress callable and a
cancel token), `DriftModel.evaluate`, the table's stash and
`apply_position_delta`, the planner's `position_offset` and `pairs`,
`find_pairs`, `link_localizations`, the fiducial adapter. One recipe in
`embedding.md`, executed by a test like the other examples: run → apply →
`renderer.update`, plus pairs. No live or streaming correction (ImSwitch2
renders while it acquires; that is a different problem).

---

## 11. Testing and docs

* **Fixture first** (phase 1): blinking emitters on a 3D structure, a known
  random-walk drift, optional beads, optional NaN rows. Every phase, demo and
  documentation screenshot uses it.
* COMET (cpu backend, tiny sets so numba compilation fits the CI budget)
  recovers the anchored drift to a stated tolerance; pairs collapse at 100 %;
  beads are found and excluded, and their tracks match the drift; linking
  reproduces the generating emitters.
* Guarantees as tests: `import napari_storm` imports neither comet, torch nor
  numba; no tkinter import and no `plt.show` during a run; a cancelled or
  failed run leaves positions bit-identical; Undo restores them bit for bit;
  0 % equals raw to 2 ulp; a sweep leaves camera, render range and active rows
  unchanged; filter Apply at 60 % stays at 60 %; Play ends at 100 %; the guard
  refuses position and time writes while a drift is applied; the exact pair
  count matches `query_pairs`; MPS is not auto-selected; deleting the pairs
  layer leaves traces on; every lifecycle transition of §4.2, including edits
  refused while Undone; full field and crop on Apply, with two datasets
  sharing the range; Exclude → change filters → Restore, and rows with both
  exclusion bits; bounds with NaN rows, and an all-NaN dataset refused; group
  means never straddle a window; trace means never split a trace; the time
  view leaves data and exports unchanged and gives the 2D camera back; the
  trace-id column is refused while a stash exists and during a run; MINFLUX
  through time → rank reproduces the rank-domain correction; the pair set is
  rebuilt on each invalidating change, reused on α changes, and a superseded
  search result is dropped.
* Tab tests with a fake runner (progress, cancel, unload and clear mid-run), on
  `NullRenderer` where possible.
* Docs: `docs/post-processing.md`, drafted with phase 2 because its words are
  also the tooltips; a step "Post-process" in `step-by-step.md`; the `[comet]`
  install with GPU notes (CPU recommended without NVIDIA); the COMET citation;
  the embedding recipe.

---

## 12. Order of work

| Phase | Content | Depends on |
|---|---|---|
| 0 | py-comet 1.2 (§9), and a memory measurement of the release for the guard (§5.3) | D1, D2 |
| 0′ | NaN positions fix, selection and bounds, as a 3.1.x bug fix | — |
| 1 | Fixture; exclusion reasons; stash and guard; display offset and positions-only refresh; `DriftModel`, drift view, `DriftChanged`; non-modal job runner; the tab with dataset selector and status line; `psutil` | — |
| 2 | Fiducials (with bead tracks); COMET section (budget, estimate and guard, run, cancel, plot, undo / re-apply / discard, save / load, apply to others, save corrected); docs draft | 0, 1 |
| 3 | Explore: slider, Play, the x, y, time view for 2D data | 2 |
| 4 | Pair network: `PairSet`, `pairs` payload, `Changed.PAIRS`, parametrised overlay, both modes | 2 |
| 5 | Grouping: side columns, linking, group means as COMET input | 1, 2 — may slip to 3.3 |
| 6 | Embedding recipe, docs, release | all |

**Deferred**: estimating on the union of several channels; the merged-group dataset;
scene entries for drift; GPU blending; more pair colourings; beads as the drift
reference; COMET's QC modes; running COMET in a subprocess; a two-pass
"coarse, then refine with a small max drift" run (would cut pairs ~10× but
needs measuring and composing drifts).
**Cut**: a duplicate "original" dataset; shipping the §9.3 shim; Play's loop;
a second drift file format; z weighting in COMET (D11); raw MINFLUX
localizations as COMET input (D12).

---

## 13. Risks

* **Memory inside napari.** The guard reduces the risk; it does not protect
  against the process being killed. COMET's own MemoryError fallback is rarely
  reached on macOS/Linux, where the kernel kills the process instead — taking
  napari and any unsaved state with it. Running COMET in a subprocess would be
  the protection; it is deferred.
* **Quit during a run** waits for the current evaluation, or the whole pair
  search if it is in one.
* **COMET's run length is not known in advance** (7 steps and 86–381
  evaluations measured; up to 13 steps by the formula at 300 nm), so time
  estimates are ranges until the first step.
* **Slider steps with traces on** are bounded by the Tracks rebuilds, not the
  splats.
* **The display offset must reach every re-plan and nothing else** (§4.3); the
  tests in §11 pin both sides.

---

## Appendix A — facts and measurements

### A.1 COMET 1.1

* `comet_run_kd(dataset (N,4) [x, y, z, frame] nm, segmentation_mode,
  segmentation_var, ...)` returns the per-frame drift `(F,4)` for frames
  0..max_frame and corrects `dataset` in place.
* Pairs: `cKDTree.query_pairs(max_drift_nm)` over the kept localizations,
  same-window pairs included (they add a constant). Cost per pair
  `exp(−d²/(4σ²))/σ` — the overlap of two σ-Gaussians, so σ ≈ localization
  precision is the natural target.
* Outer loop: one L-BFGS-B run per σ, σ /= 1.5 per success; stops when σ ≤
  target **and** (the update grew, or σ ≤ 1 nm); returns the previous run's
  estimate. On failure it retries at the same σ (deterministic on CPU, so it
  repeats), doubles σ after two, raises after five; the failure count is never
  reset.
* The estimate is zero-mean over windows (x0 = 0, and the gradient is
  antisymmetric between the two windows of a pair). Measured: estimated mean ≈
  0 against a true mean of (15, 20) nm.
* `CubicSpline` extrapolates before the first and after the last window centre.
* Mode 2 counts only frames that contain localizations; its last window can be
  a single frame; a fractional per-window cap raises.
* No progress, cancel or logging hook; `display=True` calls `plt.show()`;
  missing save paths open Tk dialogs; `interactive=True` calls `input()`.
* `import comet` imports h5py, `matplotlib.pyplot`, `numba.cuda`, and torch if
  installed; CUDA is not initialised at import.
* py-comet 1.1 replaced the CPU path's Python loop with a numba kernel,
  380–500× faster (changelog).
* `comet_fiducials` (unreleased): `find_fiducials(dataset, max_drift_nm,
  field_bounds_nm, ...) -> centers_nm, report`; `fiducial_mask(dataset,
  centers_nm, radius_nm) -> (N,) bool`; xy only; numpy and scipy only. Its
  docs report 17/17 beads on a 3.94 M-localization crop in 1.7–3.6 s on one
  CPU.
* COMET's MINFLUX batch script: trace means in sorted trace-id order, the
  trace ordinal as the frame, mode 2 with 50 per window, max drift 100 nm,
  target 10 nm, boxcar 3, one drift value subtracted per trace.
* **Memory, from the code.** Pair search: `query_pairs` returns int64 pairs
  (16 B/pair), then COMET copies them to two int32 arrays (8 B/pair) → ~24
  B/pair peak. CPU evaluation: the int32 indices stay resident (8 B/pair) and
  are cast to int64 on every call (16 B/pair, allocated and freed). torch:
  int64 indices (16 B/pair) and about a dozen (P,3) float32 temporaries per
  chunk (COMET's `torch_accelerated` docstring) → ~150 B/pair transient, with
  one chunk of up to 10⁸ pairs. CUDA: int32 indices on the device plus a fixed
  10⁸ × float64 scratch buffer (800 MB). Per localization during optimisation
  ~120 B: the float64 input (32 B), the segmented copy held for the run
  (32 B), float32 coordinates and int32 segment ids (16 B), the float64 /
  int64 copies made per CPU evaluation (32 B), segmentation arrays (~9 B).

### A.2 napari-storm 3.1

* Tabs: Data Controls, File Infos, Decorators, Data Filter, Data adjustment;
  the last two are the pattern for a dataset-specific tab (`GUI.py:84-85,
  710-713`, `_dock_widget.py:163-173, 532-534, 580-583, 1160-1162`).
* Background work: `load_in_background` (window-modal, cancel via
  `LoadHandle.checkpoint`); `image_export.run_export` (determinate bar,
  `should_cancel`); `run_on_main_thread` is a blocking queued call.
* Positions are float32 columns: STORM in camera pixels (scale = pixel size),
  MINFLUX in nm. STORM time is `frame_number` (all ones when a file has none);
  MINFLUX v1-AI and v2 have `time_s` (float32); MINFLUX v1-base has no time.
* `set_column` writes and drops caches; `refresh_dataset` re-crops and sends
  `Changed.EVERYTHING`; a full refresh of 5 M STORM rows measured 0.18 s on the
  instanced backend; the docs give 0.16 s instanced and 2.47 s on the billboard
  fallback for an update of 5 M (`docs/api.md`). Exports call
  `planner.plan(FILTERED)` directly.
* Display budget: 2048 MB / 304 B ≈ 6.7 M drawn localizations, split across
  datasets, strided.
* Traces: `TraceOverlay` (napari Tracks, all vertices at t = 0), a full napari
  rebuild per update; 8 px line-width cap on macOS.
* `dataset.file_path` is never set, so a scene's `source_path` is None; scenes
  match datasets by name and open no files.

### A.3 COMET backend costs

COMET's own measurement (`backends.py`), one cost + gradient evaluation on an
M-series Mac:

| pairs | windows | numba CPU | torch MPS |
|---|---|---|---|
| 0.5 M | 150 | 3.4 ms | 23.8 ms |
| 4 M | 1 000 | 19.7 ms | 108.6 ms |
| 12.5 M | 2 500 | 57.0 ms | 349.5 ms |

≈ 4.6 ns per pair per evaluation on the CPU. CUDA: "at least 2× faster" than
torch on the same NVIDIA GPU; no absolute numbers published.

### A.4 Measurements for this plan

Apple M2, 8 GB, under memory pressure from other work; py-comet 1.1.0 (cpu
backend), napari 0.7.1. Scripts are not checked in.

**Pair counts at target σ.** COMET (mode 2) on the two datasets in COMET's
`test_dataset/`, then pairs counted on the corrected positions:

| | 3D set | 2D set |
|---|---|---|
| localizations / windows | 200 000 / 50 | 1 077 377 / 62 |
| max drift / target σ | 60 nm / 3 nm | 120 nm / 5 nm |
| COMET's pairs (exact) | **142 M** | **62 M** |
| COMET's `estimate_pairs` | 71 M (−50 %) | 48 M (−23 %) |
| COMET run | 459 s (swapping: 390 s sys) | 38 s (200 frames per window) |
| peak RSS | 1.9 GB (under swapping, so an undercount) | |
| pairs at 2 σ, all / cross-window | 1.54 M / 1.50 M | 1.29 M / 1.23 M |
| pairs at 3 σ, all / cross-window | 3.58 M / 3.50 M | 2.73 M / 2.62 M |
| pairs at 4 σ, all / cross-window | 5.80 M / 5.68 M | 4.53 M / 4.36 M |
| nearest cross-window partner, 3 σ | 144 k edges (99 % of locs) | 602 k edges (83 %) |
| pair search at target σ | ≤ 0.2 s | ≤ 0.2 s |

The 2D set at other max drifts (exact counts, raw positions): 172.6 M at
300 nm, 81.6 M at 150 nm, 62.2 M at 120 nm; and 9.7 M / 12.1 M at 35 / 40 nm.
After a run at 205 windows (60 frames), pairs within 17.5 nm: 2.85 M on raw
positions, 3.60 M corrected (3.53 M cross-window); a centred square of 8 µm
holds 148 k of them, 10 µm 242 k.
A first attempt at 200 nm on the 3D set was killed.

**Evaluations against windows.** 2D set, max drift 120 nm, target 5 nm,
62 M pairs, evaluations counted through the §9.3 shim:

| windows | time | steps | evaluations | iterations per step |
|---|---|---|---|---|
| 56 (mode 2, 220 frames) | 35.3 s | 7 | 86 | 8, 9, 9, 8, 9, 9, 12 |
| 205 (mode 2, 60 frames, README) | 37.4 s | 7 | 86 | 9, 10, 10, 9, 9, 10, 15 |
| 2 104 (mode 1, 431 per window) | 113.1 s | 7 | 381 | 13, 12, 14, 15, 19, 65, 206 |

σ per step 40 → 26.7 → 17.8 → 11.9 → 7.9 → 5.3 → 3.5 nm in all three: the
3.5 nm step is discarded, so the accepted σ is 5.3 nm for a 5 nm target.
86 × 62 M × 4.6 ns ≈ 25 s of the 35 s are cost evaluations. Mode 2 counts
populated frames only: 12 286 of the set's 13 651 frames hold localizations.

**py-comet 1.2, phase 0.** The same 62 M-pair run (2D set, 120 nm, 205
windows), per stage, through the progress hook (tracemalloc for NumPy
allocations; resident memory under heavy swapping on this machine is an
undercount):

| | 1.1.0 | 1.2.0, cpu | 1.2.0, torch (MPS) |
|---|---|---|---|
| pair search | 1.9 s, ~24 B/pair | 3.8 s, 8 B/pair + ~0.2 GB | 3.8 s |
| optimisation | 43.4 s, 86 evaluations | 19.3 s, 86 evaluations | 414.8 s, 138 evaluations |
| allocated per evaluation | 16.7 B/pair (1.04 GB) | ~0 | — |
| peak resident memory | 1.61 GB | 0.69 GB | 0.72 GB (Metal buffers not counted) |

**Drawing segments in napari.** M random two-vertex segments in 3D,
`viewer.screenshot()` after each call (every figure includes one frame):

| M segments | Tracks: create | Tracks: one update | Vectors: create | Vectors: one update |
|---|---|---|---|---|
| 100 k | 1.20 s | 0.13 s | 0.40 s | 0.12 s |
| 1 M | 3.30 s | 1.50 s | 2.61 s | 1.49 s |
| 3 M | 12.2 s | 5.7 s | 9.5 s | 12.7 s |

**From the reviews** (probes on small synthetic data): `count_neighbors`
0.28 s for 28.7 M pairs vs 0.36 s for `query_pairs`; `cKDTree` build 0.33 s
at 1 M and 2.7 s at 5 M points; mode-1 segmentation 0.10 s at 100 k / 487
windows and 1.19 s at 400 k / 1 951 (quadratic); a linking prototype 0.84 s
for 121 k localizations over 20 k frames; one full refresh 0.18 s at 5 M STORM
rows, 0.65 s at 1 M MINFLUX rows with traces on (0.03 s without); COMET's
cubic spline extrapolated to 2.7·10⁷ nm at frame 0 for MINFLUX-like
pseudo-frames starting at 300 000; a 100 nm `set_column` shift followed by
`refresh_dataset` left 901 of 1 001 points active. Capture fraction of
same-emitter pairs within k·σ (per-axis separation sd √2·σ; analytic and
Monte Carlo agree):

| k | 2D | 3D isotropic | 3D, σz = 2.5 σxy |
|---|---|---|---|
| 2 | 0.63 | 0.43 | 0.20 |
| 3 | 0.90 | 0.79 | 0.45 |
| 3.5 | 0.95 | | |
| 4 | 0.98 | 0.95 | 0.66 |

---

## Appendix B — review log

**Round 1** (three reviewers on v1, each with one lens).
*Integration*: moved points past the old render range are dropped on refresh;
the NaN bug is real; one exclusion hook suffices; the stash belongs on the
table; progress must not block the worker; the band-filter caveat was not
implementable; traces add 0.65 s per step; pairs as a render-request payload;
`add_column` would double memory and break STORM's dtype check; "Connect traces
shows groups for free" was false; drift lookup by frame number was wrong for
1-based frames. *COMET and algorithms*: `drift[frame]` raises, wraps or
extrapolates; the drift is zero-mean; the outer-loop description was wrong;
the fallback should wrap module globals around the public call; a concrete 1.2
hook; missing per-localization and torch memory; quadratic segmentation;
MINFLUX defaults; exact pair counts; pair radius for 95 % capture; chains that
follow time; z anisotropy; fiducial bounds and raw positions; one-to-one
linking; Picasso `group` is a pick id. *Experience and scope*: fiducials
earlier; the tab drawn top to bottom; remedies on a CPU laptop; trace means;
the slider as a viewing aid; missing decisions (re-run, undo, saving, other
writers, multi-channel, QC, locking); σ vs FWHM; a failure table; the network
off by default; cut the duplicate dataset and the shipped fallback; fixture and
docs earlier.

**Round 2** (two fresh reviewers on v2).
*Consistency and feasibility*: after Undo the slider would apply the drift
twice → slider only while applied, Discard = Undo + drop; Apply would be
refused by its own guard, and the guard missed `locs_all`, `set_records`, the
pixel-size setter and the time column; which callers see the display offset,
re-plans that would snap back to 100 %, and a positions-only refresh; the
second overlay's appearance keys, removal callback and `Changed.TRACES`
short-circuit; exclusion reasons; side columns moved to phase 5 with
`field_names` records-only; envelope widening moved the crop; fallback vs
"cut" contradicted; fiducials also need py-comet; windows for rows outside
COMET's input; drift groups; no scene entry; small `DatasetState`; no
persistent KD-tree; one drift format; `psutil`; inconsistent numbers.
*Asks and science*: the pair-network default had silently reinterpreted the
request → D13; the anchor algebra stated once, through `evaluate`; measured
pair counts at the defaults (173 M at 300 nm) → a pair budget up front; the
CPU story credited to 1.1's numba kernel and checked at the defaults (→ the
window measurement above); MPS and ROCm stated plainly; the window as a
duration; the batch script described accurately (D12); bead radius ≈ half the
peak-to-peak drift, and bead tracks as an independent check; the network's σ
from localization precision, anisotropic, capture shown; circularity of the
network stated; colour by Δt; chains need Δt > dark frames; anchor on several
windows, explicit clock confirmation, scale ratio; the plot shows what is
applied; D14.

**Round 3** (one fresh reviewer on v3, consistency only): memory figures mixed
py-comet 1.1 and 1.2 → a per-version table, the mockup on 1.2; memory
constants given sources (§A.1); "the window is a duration" contradicted COMET's
mode 2 (populated frames only) → the tab computes true-duration windows and
1.2 fixes the pre-segmented path; window counts corrected (205, not 228);
Load drift must work without py-comet; K = 11 at 120 nm, and expected steps
shown; the pair budget defined; the time model given its overhead factor;
claims narrowed to what was measured (cancel on cpu only; 7 steps measured);
definitions added (frame of a group mean, MINFLUX window unit, chain rule,
windows by nearest centre, the square's size, z without σ columns, Run on a
group member, caps applied by the tab); the radius basis and the scope of
grouping added to the decisions; small numeric mismatches.

**Round 4** (an independent review the maintainer passed on, on v3) and the
maintainer's decisions. Undo left the stash able to overwrite later edits →
writes stay refused until Discard, with a lifecycle table. Preserving the
absolute crop re-introduced the clipping the widening was meant to remove →
a full field widens, an intentional crop clips. The guard budgeted the job,
not the process → stage increments over current resident memory plus a
reserve, a GPU check, constants measured on 1.2, described as risk reduction.
Clearing an exclusion bit could not restore the selection → the user
selection is kept apart. The NaN fix missed the bounds → finite-only bounds
everywhere, all-invalid datasets refused. A duration limit does not stop a
group straddling a window → split at the boundaries. Decided: D1, D2, D9–D12,
the radius half of D13, D14; MINFLUX per trace only (D12 had proposed the
opposite deviation from the batch script); the time view (§6.4).

**Round 5** (the same independent reviewer, on v4): `selection & ~excluded`
lets single-bit rows through on a byte array → `excluded == 0`, each bit
tested; MINFLUX allowed evaluation "at mean time or rank", which differ when
trace durations vary → the rank domain is canonical and time is used only to
map other datasets into it, with a round-trip test; trace identity became a
correction dependency but was writable → the trace-id column is guarded and
fingerprinted; when the pair set is rebuilt was unspecified → an invalidation
list, worker search with a generation counter; the failure table still
disabled data without a time column → MINFLUX v1-base is corrected by trace
order.

**Round 6** (the same reviewer, final): float32 times can make two traces'
mean times equal → the rank → time table is validated, and transfer is
disabled when it fails; obsolete pair searches could run side by side → one
active search, one pending request, checkpoints, counted by the COMET guard.
Reviewer's verdict: ready to implement.
