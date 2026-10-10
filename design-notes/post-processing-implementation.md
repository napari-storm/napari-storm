# Post-processing implementation checkpoint

Baseline: napari-storm `526c42c`; branch `codex/postprocessing-comet`.

## Phase 0 sanity check

COMET checkout: the COMET repository's 1.2 release branch, at
`17383e8`, version 1.2.0. Release notes say not yet tagged/published.

Focused progress/cancellation, segmentation, pair search, interpolation and
fiducial tests: **154 passed, 2 skipped** (CUDA unavailable; torch not installed).

A dense-x memory probe is checked in as `scripts/benchmark_comet_memory.py`.
At 2,000 points / 1,999,000 pairs: baseline RSS 194,494,464 bytes, peak
321,273,856 bytes, incremental **63.42 bytes/pair**. COMET's final pair arrays
are 8 bytes/pair, but the slab halo is not a bounded transient for this geometry.
The integration therefore uses a conservative 64 bytes/pair CPU estimate plus
per-row work, fixed allocation allowance and a reserve. This intentionally
supersedes the optimistic 8 bytes/pair figure in the draft.

## Integration delivered

- Finite-position exclusions survive filters and reset; all-invalid loads are
  rejected before the existing session is cleared; bounds ignore invalid rows.
- Separate selection/exclusion masks, side columns, exact position stash and
  write guard (including Undone), drift state/events, host-free model and I/O.
- Post-processing tab, bounded cancellable jobs, exact CPU pair budgeting,
  COMET adapter, plot, correction lifecycle, shared drift groups, drift and `.ns`
  I/O; optional dependency.
- Fiducial review/exclusion, STORM grouping and group-mean input.
- 10% preview and playback, time-height view, shared trace/pair coordinate path,
  bounded regional pair networks, independent pair overlays and invalidation.
- Scientific, lifecycle, dock and real-COMET integration tests; user docs.

## Release acceptance still required

The COMET package must be published before the extra can resolve from PyPI.
The initial integration intentionally exposes CPU only; GPU backend selection and
GPU budget validation need their own measured acceptance on available hardware.
The dense slab issue should be fixed upstream if the lower memory estimates are
to be advertised. Full-size representative STORM/MINFLUX accuracy and interaction
benchmarks remain release checks; the synthetic recovery test is not a substitute.

Automatic cap alternatives, bead-track overlays, raw/corrected sharpness counts
and raw-xy fiducial review circles are included in the integration. CPU is the
only selectable compute backend in this first integration.

## Review of this checkpoint and fixes

Three independent reviews (core/render path, algorithms, dock) of the
integration above; findings reproduced with probes before fixing. Regression
tests for each are in `_tests/test_postprocessing_review.py` and
`_tests/test_postprocessing.py`.

**COMET 1.2 (upstream).** The dense-x memory finding above was right. A slab
search fixing it was tried (`c5cdf29`) and then withdrawn (`7176f63`): 1.2 keeps
1.1's well-tested `query_pairs` search, ~24 B/pair at its peak, and the slab idea
is for another time. The budget therefore uses 24 B/pair plus per-localization
copies and a fixed allowance.

**Algorithms.**
- The drift was clamped at the window centres, not at the data's frame range
  (up to 3 nm wrong in the first and last half window): `DriftModel.frame_range`,
  saved and loaded, used for evaluation, File Infos counts, pair windows, the plot
  and transfer checks.
- "Apply to other datasets" refused every simultaneously recorded channel (same
  root cause).
- Knots sat at geometric window centres; now at the mean time of each window's
  localizations (0.99 → 0.12 nm RMS on gappy data).
- A window without cross-window pairs became a zero knot: windows below 200
  localizations were initially merged; this was removed at the user’s request
  (see final integration below). NaN knots remain handled as COMET handles them.
- MINFLUX traces were ranked by id; now by mean time (D12).
- Pair network lost the extreme point by an ulp and ran a Python query per row:
  whole-region search without an edge test, vectorised queries, bisection to 1 %.
- Grouping is anisotropic in 3D; Exclude replaces the previous bead choice.
- Load drift reads COMET's own files with the dataset's clock and checks the
  time range; fingerprints are hashes (trace membership, rank table), not the
  full id list.
- Over budget, Run now asks; it refuses only when the run alone exceeds RAM.

**Core.**
- Render ranges are world space, but the crop compared data-space columns: a
  shifted channel vanished on Reset render range or Apply. The crop is in world
  space now, and a transform change widens the range to the moved data.
- `set_records` crashed on text fields when extending, and dropped side columns.
- Writing a side column silently did nothing; a second Apply with another axis
  raised KeyError; the resolved trace/time columns are now protected by name.
- Pair arrival rebuilt splats and the trace layer: pairs-only refresh.
- The pair cache key used `id()` of freed arrays: a selection version counter.
- Z colour flipped on mirrored datasets with a drift.
- New dataclass fields moved to the end (hosts' positional construction).
- The all-NaN refusal was lost in a Qt slot on the threaded load path.

**Dock.**
- Without py-comet the install note overwrote every result: it has its own label.
- A cancelled pair search was never retried.
- Post-proc. moved left on the second load (the tab bar is an attribute here).
- Time view: NaN extent, not left on dataset open/close, label not reset.
- Quitting during a run waited for an uninterruptible search: daemon threads.
- Undone enabled Play/Transfer/Export; plot and status were not per dataset;
  errors were swallowed without a log; the lock could leak if submit raised.

## Still open against the plan

- §5.4 progress: done after the review (bar, "step k of ~E (≤ K) · σ · ETA",
  stage sentences); the ETA is per step, not per evaluation.
- §5.2/§5.3: no Backend choice (CPU only, by decision of this checkpoint), no
  "Before running" line with a time range, no sampled count while editing.
- §5.6: messages for one-frame data, too few windows (suggested window),
  "fiducials never checked"; ranked over-budget alternatives.
- §5.5 plot: dashed clamped ends with counts, parameters and the citation.
- §6.1 Play paced by a fixed 150 ms timer rather than by the refresh.
- §11 dock tests: progress display, Play end state and sweep invariants, unload
  of one dataset mid-run, every §4.2 enable state.
- GPU backends and their memory check need a CUDA machine; so does COMET 1.2's
  CUDA path before it is tagged.

## Final integration decisions (2026-10-04)

These supersede earlier checkpoints and the corresponding draft-plan controls.

- COMET uses its original mode 1 segmentation: the user selects localization
  count per window and maximum drift; target sigma is fixed at 10 nm.
- No custom 200-localization threshold, merging, cap, automatic sampling,
  modality-specific smoothing, initial-sigma override or frame-span cutoff.
  Original COMET whole-frame and final-window semantics remain intact.
- CPU execution is supported; GPU selection is outside this integration.
- Group input averages each group once before COMET forms count-based windows;
  MINFLUX continues to use measured-time-ordered trace means.
- The translation anchor is evaluated at the first acquisition frame in the
  estimation input, replacing the arbitrary first-2%-of-windows average.
- Memory estimation includes COMET's dense frame interpolation, so long clocks
  are guarded by their actual allocation cost rather than a frame-count cutoff.
  The upstream pair search remains the tested 1.x implementation (24 B/pair peak).
- py-comet 1.2 publication remains an external packaging prerequisite. This
  branch does not publish, tag or modify the upstream repository.

## Final validation

Validated on macOS with Python 3.11, napari 0.7.1 and Qt 6. COMET tests and
integration runs used the 1.2 release checkout at `7176f63`.

- Full napari-storm suite, including slow rendering tests: **984 passed,
  1 skipped** (629.93 seconds).
- Focused integration, export and host-free tests: **50 passed**.
- Upstream CPU kernel, reference segmentation, interpolation, pair search and
  progress/cancellation tests: **205 passed, 2 skipped** (GPU unavailable).
- All configured pre-commit hooks passed across the repository.
- Source archive and wheel built; required modules and the optional COMET
  dependency inspected. The extracted wheel imports independently of the source
  checkout and leaves COMET unloaded until requested.
- Plugin manifest validation and contribution resolution passed.
- Dense 1,999,000-pair probe: 0.022 seconds, 72 MB incremental peak RSS, within
  the pair estimate plus fixed working allowance.

Public PyPI returned 404 for py-comet at final validation. The optional extra
therefore requires upstream publication; local development uses the release
checkout. No package has been published and no branch has been pushed here.
