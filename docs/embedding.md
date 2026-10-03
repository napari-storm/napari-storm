# Embedding napari-storm in another application

napari-storm can render localizations inside any modern napari session
**without its dock widget**. A host application — ImSwitch, an acquisition GUI,
a notebook — supplies the data and drives the lifecycle; the plugin supplies
the Gaussian model and the renderer.

Every example below is executed by the test suite, so the code you copy is the
code CI runs and it cannot quietly stop working. The lifecycle and trace
examples come from `_tests/test_embedding.py`; the column-declaration example
from `_tests/test_declared_columns.py`, and the export example from
`_tests/test_ome_export.py`.

## The whole of it

```python
import numpy as np
from napari_storm.core import (DatasetTraits, GaussianSettings,
                               LocalizationTable, RenderPlanner)
from napari_storm.napari_particles.selection import select_renderer

# 1. Your data, as a numpy record array.
table = LocalizationTable(records)

# 2. Decide what to draw.  Host-free: no napari, no Qt.
request = RenderPlanner().plan(
    table,
    GaussianSettings(fixed_sigma_xy_nm=30.0),
    DatasetTraits(zdim_present=True),
    name="from-the-host",
)

# 3. Draw it.
renderer = select_renderer(viewer)
renderer.open(1, request)
viewer.dims.ndisplay = 3        # 3-D data needs napari's 3-D canvas
```

`1` is a dataset id. It is yours to choose and yours to keep — every later call
refers to the dataset by it.

## If your columns are not named ours

A format declares its own column names and units, for every measured column —
not just positions. Nothing downstream needs to know which convention a file
used:

```python
table = LocalizationTable(
    records,
    position_columns={"x": "x_nm", "y": "y_nm", "z": "z_nm"},
    position_scale_nm=1.0,
    sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
    photon_column="photons",
    copy=False,
)
```

`sigma_scale_nm` defaults to following `position_scale_nm`, so a format storing
both in camera pixels needs only the pixel size, stated once. Pass it explicitly
only when widths and positions genuinely use different units.

## The three pieces

| Piece | What it is | Needs napari? |
|---|---|---|
| `LocalizationTable` | The canonical data. Never reordered, never filtered in place; selection is a boolean mask. | No |
| `RenderPlanner` | Decides *what* to draw: Gaussian widths, intensities, coordinates. This is the science. | No |
| A `LocalizationRenderer` | Owns GPU and host layer resources. `select_renderer` picks the best one this session supports. | Yes |

The first two import and run with napari, Qt and VisPy made unimportable —
`_tests/test_core_is_host_free.py` enforces it in a subprocess. A host can
compute a render plan on a worker, in a subprocess, or on a machine with no
display.

## Choosing a backend

`select_renderer(viewer)` returns the instanced backend — 28 bytes per
localization, 0.16 s to update 5 million — when the GL session can instance,
and falls back to the original billboard renderer with a napari warning when it
cannot. Instancing needs VisPy's `gl+`, which napari itself selects for its own
Points layer, so in practice the fast path is the one you get.

To pin a backend, construct it directly:

```python
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
renderer = InstancedRenderer(viewer)
```

All backends satisfy the same contract, and every contract test in
`_tests/test_renderer_backend.py` runs against all of them.

## The lifecycle

**Filtering** replaces the mask and replans. It costs one boolean array, not a
copy of the data:

```python
mask = np.zeros(len(table), dtype=bool)
mask[selected_rows] = True
table.set_filter_mask(mask)
renderer.update(1, RenderPlanner().plan(table, settings, traits, name="ch"))
```

**Placement** is a value you supply, not a hidden global:

```python
from napari_storm.core import WorldTransform
request = RenderPlanner().plan(
    table, settings, traits, name="ch",
    transform=WorldTransform(translation_nm=(5000.0, 0.0, 0.0)),
)
```

**Appearance** is separate from what is drawn, so changing it costs nothing:

```python
from napari_storm.core import LayerAppearance
renderer.set_appearance(1, LayerAppearance(colormap="green", opacity=0.5))
```

Fields left as `None` mean "leave this as it is", so a control that owns one
slider can send only what it changed.

**Contrast acts on the summed image.** For the Gaussian and every other
additive footprint, `contrast_limits` window the sum of the splats, not each
localization: the layer is drawn into a 32-bit float target first, and the
window and colormap are applied once, to the total. The limits are in units of
summed weight -- each localization's value times its footprint -- so with values
of 1 they count overlapping localizations, and the lower limit hides sparse
regions while keeping dense ones. `renderer.contrast_is_summed(dataset_id)`
says which model is in effect. Set `LayerAppearance(summed_contrast=False)` when
the values are not weights -- the dock does for Z colour coding, where they are
depths -- to window each localization on its own instead.

Below the lower limit nothing is drawn, so a channel never tints the canvas.
A colormap that does not start at black -- viridis, turbo, hsv -- has its lowest
colour faded in from black over the first 0.2 of summed weight above the limit;
otherwise every splat would show that colour out to the edge of the square it
is drawn on. Colormaps that start at black are unaffected.

**Closing** releases the layer and everything behind it:

```python
renderer.close(1)      # one dataset
renderer.close_all()   # all of them
```

## Several channels

One renderer holds many datasets, keyed by whatever ids you use:

```python
for dataset_id, colour in ((7, "red"), (9, "green")):
    renderer.open(dataset_id, RenderPlanner().plan(
        tables[dataset_id], settings, traits,
        name=f"channel-{dataset_id}", colormap=colour,
    ))
```

## Alternative visualisations

The Gaussian is the reconstruction: summed, it is the super-resolution image,
and it is all an export writes. Every localization can instead be drawn as
anything in the footprint palette -- points, spheres, uncertainty ellipses and
the sprites napari-particles shipped. The palette is plain data, so a host can
offer it in its own menus before any viewer exists:

```python
from napari_storm.core import PALETTE, LayerAppearance

for footprint in PALETTE:
    print(footprint.name, footprint.label, footprint.group)

renderer.open(1, RenderPlanner().plan(
    table, GaussianSettings(fixed_sigma_xy_nm=10.0, fixed_sigma_z_nm=10.0),
    traits, name="points",
))
renderer.set_appearance(1, LayerAppearance(footprint="disc", min_size_px=3.0))
```

| Attribute | Meaning |
|---|---|
| `name` | what `LayerAppearance.footprint` takes and a saved scene records |
| `label`, `description` | for a menu entry and its tooltip |
| `group` | `"reconstruction"` for the Gaussian, `"visualisation"` for the rest |
| `blend` | `"opaque"` markers occlude by depth; `"additive"` sprites add up |
| `uncertainty` | draws each localization's own one-sigma ellipse |

A few things follow:

* **Markers are drawn at the one-sigma outline.** Their radius is the width the
  planner computed, so `fixed_sigma_xy_nm` is the point size in nanometres.
  Footprints marked `uncertainty` draw each localization's own ellipse: plan
  with `GaussianSettings(mode=1)` and a ring becomes an uncertainty ellipse.
  The dock's **Use uncertainty** box is that same switch.
* **Opaque markers are depth-tested.** A nearer one hides a farther one
  whatever order they were drawn in, across datasets as well. Opacity between
  0 and 1 has no effect on them: translucent markers have to be depth-sorted to
  be drawn correctly. Opacity 0 still hides the dataset.
* **`min_size_px` keeps visualisations visible zoomed out.** It is the smallest
  diameter of the one-sigma outline, in screen pixels, 2 unless you say
  otherwise. The scientific Gaussian is never enlarged: its summed intensity is
  the measurement.
* **It is appearance, not data.** Switching costs no upload and survives
  `update`; `open` starts every dataset as the Gaussian, so set it after
  opening.

Colour is per marker: each localization's own value through the colormap and
contrast limits, since an opaque footprint has no sum to window. `NapariPointsRenderer`, the comparison backend, draws each
opaque footprint as napari's nearest marker and approximates the size floor
through napari's own marker limits.

## Traces

A trace is one molecule localized repeatedly -- MINFLUX's `tid`, which the
readers store as `trace_id`. Drawn as points it is a smear; connected through
its own localizations in the order they were measured, it is a path, which is
what a tracking acquisition measures. Any integer column can name the traces:
plan with it, then ask for the overlay through the appearance:

```python
from napari_storm.core import LayerAppearance, RenderPlanner, find_trace_column

request = RenderPlanner().plan(
    table, settings, traits, name="tracking",
    trace_column=find_trace_column(table),      # "trace_id", or None
)
renderer.open(1, request)
renderer.set_appearance(1, LayerAppearance(
    traces=True, trace_color_by="time", trace_width_px=2.0,
))
assert renderer.draws_traces(1)
```

`find_trace_column` tries `trace_id`; for any other column -- Picasso's
`group`, a tracker's id, a column your format declares -- pass its name. It
has to hold whole numbers: a fraction or a NaN is refused rather than guessed
at, and negative ids, which tracking and clustering tools write for
"unlinked", join no trace.

| Field | Meaning |
|---|---|
| `traces` | draw the overlay; off until set, and `open` starts every dataset without it |
| `trace_color_by` | `"trace"` (default): one hue per trace, kept when a filter removes its neighbours; `"time"`; `"progress"` along each trace, first to last; or a column planned with `trace_properties=("efo",)` |
| `trace_width_px` | line width in screen pixels |

What follows from planning it with the request:

* **The vertices are the splat centres, exactly.** `request.traces` is built
  from the coordinate array planned for the splats, so a trajectory passes
  through its own localizations after a `WorldTransform`, a filter mask, a
  render-range crop or the display limit -- under the budget it connects the
  localizations that are drawn. Flat data stays on the splats' plane.
* **Order is time, else acquisition.** Each trace is sorted by the table's time
  (`time_s`, then a frame number) and, where there is none, keeps its row order
  -- which is acquisition order for every reader here. Ties keep the row order
  too. `time_column=` names another column, or `None` asks for row order.
* **It follows the dataset.** `update` replans it, `set_visible` and the
  appearance's `opacity` hide and fade it, `close` removes it. An update whose
  `changed` lacks `TRACES`, `SELECTION` and `POSITIONS` leaves the overlay as
  it is -- that is what keeps a width change cheap -- and one that is only
  `Changed.TRACES` touches nothing but the overlay.
* **A trace of one selected localization** has nothing to connect and is left
  out; with nothing left to connect there is no overlay layer at all.
* **It is a visualisation.** No export writes it, and nothing derives a width
  from a trace's spread: that spread is a precision only for a fixed sample,
  and is the motion itself in a tracking run.
  `core.trace_spread_is_meaningful(is_tracking)` says which, and whether an
  acquisition is tracking cannot be read off the file -- the caller says so.

On the napari backends the overlay is a napari **Tracks** layer named
`"<dataset> traces"` (`renderer.trace_layer(1)` hands it over for inspection).
Tracks are time-resolved, and two things a host might otherwise trip over are
handled for it:

* every vertex sits at **one time point**, so the time axis napari adds has a
  single step: there is no slider to move before anything shows, and nothing
  is faded. Time is shown by `trace_color_by="time"` instead;
* napari **refits the camera** whenever the viewer gains or loses an axis; the
  overlay restores the camera, and the layer selection, after adding or
  removing its layer.

Lines are drawn by the GL driver, which caps their width: at 8 logical pixels
on macOS (measured), and a driver that offers only thin lines -- which OpenGL
core profiles are allowed to -- draws every width as one pixel.
Deleting an overlay layer in napari turns that dataset's traces off rather
than closing the dataset; set `renderer.on_traces_removed_by_host` to hear
about it.

**Checking for it.** As for summed contrast, check for the capability rather
than a version:

```python
from dataclasses import fields
has_traces = "traces" in {f.name for f in fields(LayerAppearance)}
```

`LocalizationRenderer.draws_traces` is part of the contract; a backend of your
own that does not draw traces inherits an answer of False.

**What it costs.** 10^4 traces of 100 localizations -- 10^6 vertices -- plan in
0.06 s, and the overlay takes 0.7 s to build, 0.4-0.5 s to follow a filter
change (most of it napari re-indexing the tracks) and adds about
2 ms to a 2560x1600 frame; a width or colour change is immediate, and a
filter change with traces off costs what it did. At 10^5 vertices those are
0.12 s and 0.04 s. Memory is about 90 bytes a vertex for the vertices and the
arrays handed to napari, and as much again inside napari.
`scripts/benchmark_traces.py` measures it.

## Things that will catch you

**The renderer is main-thread and same-process.** Planning runs anywhere —
`LocalizationTable`, `RenderPlanner`, filtering, coordinate conversion and
export are all host-free. Everything from `renderer.open` onward must be on the
Qt GUI thread, and building the geometry is the larger cost on big datasets, so
it cannot be moved off it.

**3-D data needs `ndisplay = 3`.** napari's canvas defaults to 2-D, where it
shows a single slice. The dock widget sets this for you; a host must do it
itself.

**Do not pass `colormap=None`.** It defaults to `"gray"` precisely so that this
is hard to get wrong — but if you pass `None` explicitly, napari assigns an
arbitrary unnamed colormap and the instanced backend, which samples the
colormap in its own shader, can resolve it to black. Every piece of state then
reports itself healthy while the canvas stays empty.

**The memory budget is yours to apply.** The dock widget applies it; a host
driving the core directly does not get it for free:

```python
from napari_storm.memory_budget import (default_render_budget_mb,
                                        max_localizations_for_budget)
table.limit_active_to(max_localizations_for_budget(default_render_budget_mb()))
```

!!! danger "Two masks, and the distinction is load-bearing"

    `filter_mask` is what the user selected; the display limit above it is what
    the GPU can afford; `active_mask` is the intersection.

    **Anything leaving the process — an export, a saved file, a reported count
    — must read the filter set**, via `plan(..., selection=FILTERED)`. Only the
    renderer sees the display set. Collapsing the two writes a subsample of
    someone's data to disk because their graphics card was busy.

## Axis order

`RenderPlanner.coordinates` and `RenderPlanner.sigmas` both return `(z, y, x)`,
which is napari's order, and a test pins the result against a real napari Points
layer rather than against a convention of ours.

This matters to a host because a reconstruction shares its viewer with ordinary
napari layers — ROI shapes, points overlays, a widefield image. If the orders
disagree the reconstruction is *misregistered* against all of them, and the
error is self-consistent inside the plugin, so only a host ever sees it. A
transposition check against a plain napari layer is worth keeping in your own
test suite.

`WorldTransform` is keyed by axis *name*, so placement code is unaffected by
array column order.

## Testing without a GL context

`NullRenderer` satisfies the same contract and draws nothing:

```python
from napari_storm.core import NullRenderer
renderer = NullRenderer()
```

It needs no viewer and no GL session, so an integration can be covered in CI on
a headless runner.

## Exporting

The exporter is host-free too, and takes the same plan-then-act shape:

```python
from napari_storm.core.ome_export import ExportChannel, plan_export, write_ome_tiff

plan = plan_export([channel], bounds_nm, pixel_size_nm=10.0)
print(plan.shape, plan.nbytes)     # knowable before a byte is written
write_ome_tiff("out.ome.tif", plan)
```

It never downsamples: the requested pixel size is honoured exactly and tiles are
streamed, so peak memory is one 1024² tile whatever the size of the file.

## What is not settled yet

* **`viewer` is still required** for the renderer. There is no transport-neutral
  command layer — Level 5 of the modernization plan — so a host embeds in the
  same process as napari.
* **The reader hook still uses `napari.current_viewer()`**, so file
  drag-and-drop is tied to a global. A host driving the API directly does not
  touch that path.
* **There is no incremental append.** Adding localizations to an open dataset
  means replacing the records and replanning; `set_records` resets the
  selection, so a host that filters must re-apply its mask afterwards.

## Worked examples

### ImSwitch2

[ImSwitch2](https://github.com/openUC2/ImSwitch)'s ImProcess is a desktop Qt
application that owns its own napari viewer, so it embeds in-process: it hands
the viewer over rather than having one created for it. Its localizations are
already a nanometre recarray (`frame`, `x/y/z_nm`, `sigma_x/y/z_nm`, `photons`),
so the whole data contract is one `LocalizationTable` call with
`sigma_columns` and `photon_column` declared.

The part worth copying is the lifecycle mapping. ImProcess has an explicit
results list, so it announces openings and closings rather than being watched:

| Host event | napari-storm call |
|---|---|
| result created | nothing (lazy) |
| result selected, first time | `open(next_id, request)` |
| result selected, already open | set layer visible |
| result deselected | set layer hidden — **not** `close` |
| result changed in place | `update(id, request)` |
| result removed | `close(id)`, id retired forever |
| viewer torn down | `close_all()` |

Hide-rather-than-close is where a host's architecture is most likely to collide
with this one. A display path that rebuilds every managed layer from a spec on
each selection change would open and close on every click — the churn `update`
exists to prevent. The fix is to keep localization layers in a **parallel
retained channel** keyed by result identity, where only explicit removal closes
a dataset.

Two things fell out of that integration that are worth knowing. Its live
acquisition grows a table and re-updates, which is the `set_records` pattern
above; and because its own export reads its own records rather than the
renderer, the display-limit hazard cannot reach a file it writes — a shape worth
copying deliberately.

The full five-round interface negotiation, including the two bugs it found in
napari-storm, is kept in `design-notes/imswitch-integration-log.md` in the
repository.
