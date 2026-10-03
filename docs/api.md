# API Reference

napari-storm has two layers, and which one you want depends on what you are
building.

* **The plugin.** A napari dock widget with controls for import, filtering,
  rendering and export. Most users never go below this.
* **The core.** Value objects and a renderer contract that a *host application*
  can drive directly, with no dock widget and — for everything except the
  renderer itself — no napari, Qt or VisPy at all.

If you are embedding napari-storm in another application, start with
[`embedding.md`](embedding.md); this page is the reference behind it.

---

## The core

Everything under `napari_storm.core` imports and runs without a host.
`_tests/test_core_is_host_free.py` enforces that in a subprocess where napari,
Qt and VisPy are made unimportable, so it is a tested property rather than a
convention.

### `LocalizationTable`

The canonical data, and the single source of truth.

* `records` is written once and afterwards only through `set_column` /
  `adjust_column`. It is never reordered, never shortened, never filtered in
  place — so **the row index is the stable localization id**.
* Selection is a **boolean mask**, so filtering costs one boolean array rather
  than a copy of the data.
* There are **two** masks and the distinction is load-bearing.
  `filter_mask` is what the user asked for; the *display limit* on top of it is
  what the renderer can afford. `active_mask` is their intersection.
  **Anything that leaves the process — an export, a save, a reported count —
  must read the filter set.** Only the GPU sees the display set.
* `selection(FILTERED)` / `selection(ACTIVE)` hand either one to the planner
  behind a uniform accessor.
* Coordinates come back in **nanometres** from `coordinate_nm(axis)`, whatever
  unit the underlying column uses.
* **So do the other measured columns.** `sigma_nm(axis)` and `photons()` read
  through names the format declares (`sigma_columns`, `photon_column`), with
  `sigma_scale_nm` defaulting to follow `position_scale_nm` so a dataset carries
  one unit decision rather than two that can drift. Before this, only positions
  could be declared and the planner read widths by hardcoded pixel-native names
  — so a nanometre-native format rendered in fixed-width mode and raised in
  variable-width mode.

Axis *order* is deliberately not this class's business: positions are keyed by
axis name, and the renderer's `(z, y, x)` ordering is applied at that boundary.

### `RenderPlanner`

Decides *what* to draw; a backend decides *how*. This is where the Gaussian
sizing and intensity weighting live, which makes it science rather than
presentation — and why it is testable without a viewer.

`plan(table, settings, traits, *, name, transform, colormap, selection, …)`
returns a `RenderRequest`. Pass `selection=FILTERED` for an export.

`plan(..., trace_column="trace_id", time_column=AUTO, trace_properties=())`
also arranges the same rows as trajectories into `request.traces`; see
[`core.traces`](#coretraces) below.

### `GaussianSettings`, `DatasetTraits`

`GaussianSettings` is the subset of render configuration that changes *what* is
drawn. `DatasetTraits` is what a format actually recorded — variable-size mode
needs some measure of uncertainty, and which one exists is a property of the
file, not a guess from a column full of ones.

### `RenderRequest`, `LayerAppearance`, `Changed`

`RenderRequest` is everything a backend needs for one dataset. Its `colormap`
defaults to `"gray"` rather than `None`, because "no colormap" is not neutral —
see the warning in [`embedding.md`](embedding.md).

`LayerAppearance` fields default to `None` meaning *leave this as it is*, so a
control owning one slider can send only what it changed. Its `footprint`
picks what every localization is drawn as from `core.footprints.PALETTE` --
the scientific Gaussian or an alternative visualisation -- with `min_size_px`
as the smallest on-screen size; see
[alternative visualisations](embedding.md#alternative-visualisations).
`summed_contrast` says whether an additive footprint's `contrast_limits` window
the summed image (the default) or each localization; a backend reports what is
in effect through `contrast_is_summed(dataset_id)`. `traces`,
`trace_color_by` and `trace_width_px` draw a dataset's traces over it; see
[traces](embedding.md#traces).

`RenderRequest.traces` is the `TraceVertices` a plan with a `trace_column`
made, or `None`.

`Changed` is a flag set saying which parts of a request differ from the last
one, so a backend can update only those buffers. `Changed.TRACES` on its own
says only the trace overlay differs; `EVERYTHING` includes it.

### `LocalizationRenderer`

The contract between deciding and drawing:

`open` · `update` · `set_visible` · `set_appearance` · `appearance` ·
`value_range` · `close` · `close_all` · `is_open` · `host_bytes`

and, not abstract, `contrast_is_summed` and `draws_traces`, which answer False
on a backend that has no summed image or no trace overlay. The three napari
backends also offer `layer(dataset_id)` and `trace_layer(dataset_id)` for
inspection, and two callbacks a host may set:
`on_layer_removed_by_host(dataset_id)`, when the user deletes a localization
layer in napari, and `on_traces_removed_by_host(dataset_id)`, when they delete
a trace overlay -- which turns that dataset's traces off and leaves the dataset
open.

Three drawing backends satisfy it, and every contract test runs against all
three. The measured comparison that chose the default:

| Backend | Bytes/localization | 5M update | Notes |
|---|---:|---:|---|
| `InstancedRenderer` | 28 | 0.16 s | **Default.** Needs VisPy `gl+`. |
| `NapariParticlesRenderer` | 352 | 2.47 s | The original; the fallback. |
| `NapariPointsRenderer` | 52 | 25.57 s | No Gaussian. Kept as a reference point. |

`napari_particles.selection.select_renderer(viewer)` picks between the first two
based on what the GL session can actually do, and warns through napari when it
has to fall back.

A fourth implementation, **`NullRenderer`**, satisfies the same contract and
draws nothing. It is exported from `napari_storm.core`, needs no GL context,
and is what an embedding host should test against in CI.

### `DatasetStore`

Owns datasets, assigns each a never-reused id, and announces changes
(`DatasetOpened`, `DatasetClosed`, `AppearanceChanged`, `MaskChanged`,
`TransformChanged`, `StoreCleared`). Consumers subscribe rather than being
told, so unloading a dataset does not require remembering every dependant.

### `WorldTransform`

Where a dataset sits in world space, as a value rather than an assumption:
anisotropic scale plus translation, per axis, in nanometres. Identity returns
its input untouched. Rotation and landmark registration are not in it yet.

### `core.traces`

Which localizations make up one molecule's trajectory, and in which order --
host-free, like the rest of the core.

* `plan_traces(table, traits, *, trace_column=None, time_column=AUTO,
  transform=IDENTITY, selection=ACTIVE, properties=(), coords=None)` returns a
  `TraceVertices`: the selected rows grouped by trace and sorted by time within
  each, with exactly the coordinates the splats get. `time_column=AUTO` uses
  the table's time (`time_s`, then `frame_number`, then `frame`) and otherwise
  the row order within each trace; `None` asks for the row order. Negative ids
  join no trace, and a trace of one selected localization is left out.
* `TraceVertices` holds `coords` (`(N, 3)` float32, `(z, y, x)`),
  `trace_index` (dense, into `trace_ids`), `trace_ids`, `time` (what each
  trace was sorted by), `localization_ids` (stable row ids), `trace_column`,
  `time_column` and any planned `columns`; `n_vertices`, `n_traces`,
  `n_segments`, `starts`, `lengths`; and `color_values(name)` for `"trace"`,
  `"time"`, `"progress"` or a planned column.
* `find_trace_column(table)` returns `"trace_id"` when the table has a usable
  one, `is_trace_column(table, name)` checks any column: present, and whole
  numbers with no NaN.
* `trace_spread_is_meaningful(is_tracking)`: whether a per-trace spread
  estimates localization precision -- only for a fixed sample. Nothing in
  napari-storm estimates a width from it; whether an acquisition is tracking
  cannot be read off the file.

The napari side is `napari_particles.trace_overlay.TraceOverlay`, which every
napari backend composes: one napari Tracks layer per dataset that asks for one,
every vertex at a single time point so no slider hides any of it.

### `raster` and `ome_export`

CPU rasterization of the same Gaussian model, and the calibrated OME-TIFF
writer. Both host-free. `plan_export` describes the output — shape, bytes,
warnings — before a byte is written; `write_ome_tiff` streams tiles so peak
memory is one tile whatever the file size. The rasterizer is pinned against a
closed-form Gaussian sum, not against a screen capture. It writes the
reconstruction only: alternative visualisations and trace overlays are for
viewing.

---

## The plugin

### `napari_storm`

The dock widget. Owns the store, the interfaces and the tabs (Data Controls,
File Infos, Decorators, Data Filter, Data adjustment).

### `DataToLayerInterface`

The adapter between the Qt-configured settings and the host-free planner. Holds
per-dataset render state keyed by dataset id, applies the memory budget and the
screen-space cap, and subscribes to the store. `trace_column_of(dataset)` says
which column identifies a dataset's traces -- a declared `trace_column`, or
`trace_id` -- and `apply_trace_style()` gives every dataset that has one the
Decorators tab's trace settings.

### `ChannelControls`

Per-dataset UI: colormap, contrast, opacity, show/hide.

### `FileToLocalizationDataInterface`

Import for Picasso, ThunderSTORM, MINFLUX and custom formats, producing
`LocalizationDataBaseClass` / `StormDataClass` / `MinfluxDataBaseClass`.

### `DataFilterInterface`, `DataAdjustmentInterface`

Histogram property filters, and offset/rescale transformations plus `.ns`
export.

### `image_export`

Turns widget state into an `ExportPlan`. Coordinates and sigmas are both
`(z, y, x)`, so nothing is reordered here any more — see the note under
`RenderPlanner.coordinates` for why they once differed and what it cost.

### `_napari_compat`

Every private napari lookup lives here, with a documented fallback each — one
file to update when napari moves something.

---

## Dataset classes

In `localization_dataset_types/`:

* **`LocalizationDataBaseClass`** — x/y(/z) positions in nm.
* **`StormDataClass`** — pixel units plus photon counts and uncertainty.
* **`MinfluxDataBaseClass` / `MinfluxDataAIClass`** — MINFLUX with trace ids or
  AI JSON.
* **`MinfluxDataV2Class`** — the flat layout Imspector writes from 24.10
  onwards, in every container it and pyMINFLUX produce (`.npy`, `.json`,
  `.mat`, `.zarr`, `.pmx`). Which layout a file uses is determined from the
  file, so nothing has to be selected.
