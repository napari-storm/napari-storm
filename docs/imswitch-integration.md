# Integrating napari-storm into ImSwitch2

A worked integration against a specific host, for a developer who has not seen
this codebase. [`embedding.md`](embedding.md) covers the API in general; this
page covers the decisions a real host had to make, and what they cost.

The interface described here is settled and ImSwitch2 is building against it.
The negotiation that produced it — five rounds, including two bugs it found in
napari-storm — is kept in `design-notes/imswitch-integration-log.md` in the
repository, for anyone who needs to know *why* rather than *what*.

---

## The question that decides the shape

**Is the host a desktop Qt process, or a browser/headless one?**

The renderer draws through napari and VisPy onto a GL canvas, so it must live in
a Qt process with a real viewer. There is no transport-neutral command layer —
no RPC, no socket protocol, no way to drive the renderer from another process.
A browser frontend talking to a headless backend needs a **sidecar
visualization process** instead: the host hands over localizations and a
separate napari process renders them. The data contract below is reusable
either way; only the transport changes.

ImSwitch2's ImProcess is a Qt application, so this page assumes embedding.

---

## The whole integration

Three steps. This is copied from `_tests/test_embedding.py`, which CI runs, so
it cannot quietly stop working.

```python
import numpy as np
from napari_storm.core import (DatasetTraits, GaussianSettings,
                               LocalizationTable, RenderPlanner)
from napari_storm.napari_particles.selection import select_renderer

# 1. Your localizations, as a numpy record array.
table = LocalizationTable(records)

# 2. Decide what to draw.  No napari, no Qt — this runs on a worker.
request = RenderPlanner().plan(
    table,
    GaussianSettings(fixed_sigma_xy_nm=30.0),
    DatasetTraits(zdim_present=True),
    name="imswitch-channel-0",
)

# 3. Draw it.  This needs the viewer and the GUI thread.
renderer = select_renderer(viewer)
renderer.open(1, request)
viewer.dims.ndisplay = 3        # 3-D data needs napari's 3-D canvas
```

`1` is a dataset id. The host chooses it and keeps it; every later call refers
to the dataset by it. **Do not reuse an id after closing it** — the renderer
keys its resources by id, and our own store guarantees non-reuse for exactly
this reason. A recycled id is how a stale handle gets mistaken for a live one.

`select_renderer` picks the instanced backend — 28 bytes per localization,
0.16 s to update 5 million — when the GL session supports instancing, and falls
back to a billboard renderer with a warning when it does not. Same image,
roughly 12× the memory.

### Versions

napari-storm needs `napari>=0.4,<0.8` and Python ≥3.10. The base package
deliberately does not pin a Qt binding: the host's own binding is used, or take
the `[pyqt6]` / `[pyside6]` extra.

ImSwitch2 pins `napari>=0.7.0` through `qtpy`, so the versions satisfying both
sides are **0.7.x** — a one-minor-version intersection. It is worth checking
that whichever napari version a host depends on is one that both projects
actually exercise in CI, rather than one that merely satisfies two ranges.

---

## The data contract

A **numpy record array**, one row per localization. By default the position
columns are `x_pos_nm`, `y_pos_nm`, `z_pos_nm`. A format declares its own column
names and units rather than converting to ours — for every measured column, not
just positions:

```python
table = LocalizationTable(
    records,
    position_columns={"x": "x_nm", "y": "y_nm", "z": "z_nm"},
    position_scale_nm=1.0,
    sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
    photon_column="photons",
    copy=False,          # skip the defensive copy when you just built the array
)
```

Everything downstream reads nanometres regardless. A missing `z` column simply
means the table has no z axis.

`sigma_scale_nm` defaults to *following* `position_scale_nm`, so a format
storing both in camera pixels needs only the pixel size, stated once. Pass it
explicitly only when widths and positions genuinely use different units.

### Declaring what the format recorded

`DatasetTraits` states what a format actually measured, as opposed to what it
could have:

| Field | Meaning |
|---|---|
| `zdim_present` | there is a real z coordinate |
| `sigma_present` | localization uncertainty was fitted |
| `photon_count_present` | a photon count was recorded |
| `pixel_size_nm` | camera pixel size |

This is declared rather than sniffed on purpose. Variable-width Gaussian mode
(`GaussianSettings(mode=1)`) needs a real uncertainty measure, and a column full
of ones is indistinguishable from a real one by inspection. With `mode=0` every
localization gets one fixed width and none of this matters.

!!! warning "Declare only what you fitted"

    A dataset with `zdim_present=True` whose axial width was never fitted will
    raise, and deliberately so: declaring a width you did not measure is a wrong
    declaration, and the error names the column. An *absent* axial sigma column
    is fine — it falls back to the declared floor — so a 2-D format may drop the
    column rather than zero-filling it.

### Axis order

`RenderPlanner.coordinates` and `RenderPlanner.sigmas` both return `(z, y, x)`,
which is napari's order. They agree, and a test pins the result against a real
napari Points layer rather than against a convention of ours.

This matters to a host because a reconstruction shares its viewer with ordinary
napari layers — ROI shapes, points overlays, a widefield image. If the axis
orders disagree the reconstruction is *misregistered* against all of them, and
the error is self-consistent inside the plugin, so only a host sees it. A
transposition check against a plain napari layer is a worthwhile thing for a
host to keep in its own test suite.

`WorldTransform` is keyed by axis *name*, so placement code is unaffected by
array column order.

---

## Lifecycle

**Filtering** replaces a boolean mask and replans. It costs one boolean array,
never a copy of the data:

```python
table.set_filter_mask(mask)
renderer.update(1, RenderPlanner().plan(table, settings, traits, name="ch"))
```

**Placement** is a value passed in, not a hidden global — this is how a channel
gets registered against another:

```python
from napari_storm.core import WorldTransform
request = RenderPlanner().plan(
    table, settings, traits, name="ch",
    transform=WorldTransform(translation_nm=(5000.0, 0.0, 0.0)),
)
```

**Appearance** is separate from what is drawn, so changing it rebuilds nothing:

```python
from napari_storm.core import LayerAppearance
renderer.set_appearance(1, LayerAppearance(colormap="green", opacity=0.5))
```

`None` fields mean "leave this alone", so a control owning one slider sends only
what it changed.

**Several channels** are just several ids on one renderer: `renderer.open(7, …)`,
`renderer.open(9, …)`.

**Closing** releases the layer and its GPU resources: `renderer.close(1)`, or
`renderer.close_all()`.

`update` keeps the dataset's resources and `open` replaces them. The distinction
is load-bearing — recreating layers on every change is the leak this
architecture was built to fix, so an acquisition loop must call `update`.

### Mapping a host's events onto it

The mapping ImSwitch2 settled on, as a worked example:

| Host event | napari-storm call |
|---|---|
| localization result created | nothing (lazy) |
| result selected, first time | `open(next_id, request)` |
| result selected, already open | set layer visible |
| result deselected | set layer hidden — **not** `close` |
| result data changed in place | `update(id, request)` |
| result removed from list | `close(id)`, id retired forever |
| viewer torn down | `close_all()` |

Hide-rather-than-close is the one place a host's architecture is likely to
collide with this one. A display path that rebuilds every managed layer from a
spec on each selection change would open and close on every click — precisely
the churn `update` exists to prevent. The answer is to keep localization layers
in a **parallel retained channel**, keyed by result identity, where only
explicit removal closes a dataset.

---

## Constraints worth knowing before you design around them

**The renderer is main-thread and same-process.** Planning
(`LocalizationTable`, `RenderPlanner`, filtering, coordinate conversion, export)
runs anywhere — importing `napari_storm.core` with napari, Qt and VisPy made
unimportable is enforced by a test in a subprocess. Everything from
`renderer.open` onward must be on the Qt GUI thread. Building the geometry and
handing it to napari is the larger cost on big datasets, and it cannot be moved
off that thread.

**3-D data needs `viewer.dims.ndisplay = 3`.** napari's canvas defaults to 2-D,
where it shows a single slice. The dock widget sets this; a host must do it.

**Never pass `colormap=None`.** It defaults to `"gray"` precisely because "no
colormap" is not neutral: napari assigns an arbitrary unnamed colormap, and the
instanced backend resolves that to black. Every piece of state then reports
itself healthy while the canvas stays empty. It is the worst thing to debug in
this codebase.

**The memory budget is the host's job when driving the core directly.** The dock
widget applies it; a host bypassing the widget does not get it for free:

```python
from napari_storm.memory_budget import (default_render_budget_mb,
                                        max_localizations_for_budget)
table.limit_active_to(max_localizations_for_budget(default_render_budget_mb()))
```

!!! danger "Two masks, and the distinction is load-bearing"

    `filter_mask` is what the user selected; the display limit above it is what
    the GPU can afford; `active_mask` is the intersection. **Anything leaving
    the process — an export, a saved file, a reported count — must read the
    filter set** (`plan(..., selection=FILTERED)`). Only the renderer sees the
    display set.

    Collapsing the two writes a subsample of someone's data to disk because
    their graphics card was busy. A host whose own export path reads its own
    localization records, never the renderer, is immune to this by
    construction — which is the shape to aim for.

---

## Live acquisition

There is **no incremental append API**. The pattern that works today is to
rebuild the table over the grown array and update:

```python
table.set_records(all_records_so_far, copy=False)
renderer.update(dataset_id, planner.plan(table, settings, traits, name="live"))
```

Two things to know about that:

* `set_records` **resets the selection to all rows** and drops the derived
  caches. A host that is also filtering must re-apply its mask afterwards.
* Each update replans the whole dataset — coordinates, sigmas, values — and
  re-uploads. At 5M localizations that measured 0.16 s on the instanced backend,
  which bounds a realistic refresh rate rather than making it free.

For a live reconstruction at a few Hz this is adequate. For a high-rate stream
an append path taking a delta rather than a replacement is the obvious thing to
improve, and a host that needs one should come with a measured
localizations-per-second and batch size rather than an estimate.

---

## What else is available

* **Scene persistence.** `save_scene` / `load_scene` write a small JSON file
  recording the *decisions* — per-dataset transform, appearance, Gaussian
  settings, reference-image placement, camera — and never the localizations.
  Reloading re-reads the source files and re-applies the decisions.
* **Calibrated export.** `plan_export` describes the output — shape, bytes,
  warnings — before a byte is written; `write_ome_tiff` streams tiles, so peak
  memory is one 1024² tile whatever the file size. It never downsamples. Both
  are host-free, so a host can export without a viewer at all.
* **Reference images.** Widefield/confocal images can be placed alongside a
  reconstruction with a pixel size and an offset.
* **`NullRenderer`.** Exported from `napari_storm.core` and satisfying the same
  contract as the drawing backends, so an integration can be covered in CI
  without a GL context.

---

## Questions to answer before drafting an adapter

Reusable for any host, in roughly the order the answers unblock work:

1. **Desktop Qt or browser/headless?** The single blocking question.
2. If Qt: **does the host already own a napari viewer** to hand over, or should
   one be created and docked?
3. **Which Qt binding and version**, and which napari version, if any is
   already pinned.
4. **What do the localizations look like** at the point of handover — array,
   dtype, column names, units, and whether uncertainty or photon counts are
   genuinely fitted?
5. **Live or post-hoc?** If live: expected localizations per second, batch size,
   and total per acquisition.
6. **Who owns the lifecycle** — does the host announce when a channel opens and
   closes, or is it expected to be watched?
7. **Is napari-storm an optional dependency or a hard one?** Optional is the
   recommendation, with the adapter importing lazily behind a feature flag and
   falling back to the host's existing preview when the import fails.

Answers to 1, 4 and 5 are enough to draft the adapter.
