# Integrating napari-storm into ImSwitch2

For an ImSwitch2 developer who has not seen this codebase. It says what
napari-storm can be driven with, what it needs from a host, what it does not do
yet, and the questions we need answered from the ImSwitch2 side before choosing
a shape.

napari-storm renders single-molecule localization data as summed Gaussians. The
part that decides *what* to draw is a plain numpy library with no UI; the part
that draws it needs a napari viewer. That split is what makes an integration
possible without adopting our dock widget.

> **Round 4 in progress.** Blocks marked **ImSwitch2 —** are answers and
> counter-proposals from the ImSwitch2 side, checked against the code in both
> repositories rather than inferred from this document. §§0–7 are the original
> text plus round-2 replies; §8 lists what round 2 left open; §9 is the
> napari-storm answer; §10 is ImSwitch2's verification of it.
>
> **State:** the interface is settled and ImSwitch2 is building against it.
> One open question remains, in §10 — how a localization with no usable width
> should be *weighted*, as distinct from how it is drawn. It changes rendered
> values, not the contract.

---

## 0. The question that decides everything else

**Is the ImSwitch2 surface we are targeting a desktop Qt process, or a
browser/headless one?**

Everything below assumes the first. The renderer draws through napari and VisPy
onto a GL canvas, so it must live in a Qt process with a real viewer. There is
currently no transport-neutral command layer — no RPC, no socket protocol, no
way to drive the renderer from another process. If ImSwitch2's UI is a browser
frontend talking to a headless backend, embedding is the wrong shape and we
should be discussing a **sidecar visualization process** instead: ImSwitch2
hands over localizations, a separate napari process renders them. The data
contract in §2 is reusable either way; only the transport changes.

Please answer this one first. It is the difference between a one-file adapter
and a new interface.

> **ImSwitch2 — Desktop Qt. Settled.**
>
> ImProcess is a Qt application, not a browser frontend. Embedding, not a
> sidecar. Everything below applies as written.

---

## 1. The whole integration, for a desktop Qt host

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

`1` is a dataset id. ImSwitch2 chooses it and keeps it; every later call refers
to the dataset by it. **Do not reuse an id after closing it** — the renderer
keys its resources by id, and our own store guarantees non-reuse for exactly
this reason. A recycled id is how a stale handle gets mistaken for a live one.

`select_renderer` picks the instanced backend — 28 bytes per localization,
0.16 s to update 5 million — when the GL session supports instancing, and falls
back to a billboard renderer with a warning when it does not. Same image,
roughly 12× the memory.

Installation is `pip install -e .` from this repo. It needs `napari>=0.4,<0.8`
and Python ≥3.10. The base package deliberately does not pin a Qt binding —
ImSwitch2's own binding is used, or take the `[pyqt6]` / `[pyside6]` extra.

> **ImSwitch2 — Bindings and versions.**
>
> PyQt5 5.15.14 through `qtpy>=2.0`; napari pinned `>=0.7.0` (0.7.1 in the
> working environment); numpy 2.4.4. No Qt extra needed — take ours.
>
> Note the overlap: our floor is 0.7.0 and your ceiling is `<0.8`, so the only
> versions satisfying both are **0.7.x**. It works today, but it works by
> coincidence rather than by anyone's intent. If 0.7 is not in your CI matrix,
> the entire supported intersection is one unproven minor. Worth pinning down
> before either side starts depending on it.

---

## 2. What ImSwitch2 supplies: the data contract

A **numpy record array**, one row per localization. By default the position
columns are `x_pos_nm`, `y_pos_nm`, `z_pos_nm`. If ImSwitch2 stores camera
pixels, say so rather than converting:

```python
table = LocalizationTable(
    records,
    position_columns={"x": "x_pos_pixels", "y": "y_pos_pixels"},
    position_scale_nm=pixel_size_nm,
    copy=False,          # skip the defensive copy when you just built the array
)
```

Everything downstream reads nanometres regardless. A missing `z` column simply
means the table has no z axis.

`DatasetTraits` declares **what the format actually recorded**, as opposed to
what it could have:

| Field | Meaning |
|---|---|
| `zdim_present` | there is a real z coordinate |
| `sigma_present` | `sigma_x_pixels` / `sigma_y_pixels` / `sigma_z_pixels` columns exist |
| `photon_count_present` | a `photon_count` column exists |
| `pixel_size_nm` | camera pixel size |

This is declared rather than sniffed on purpose: variable-width Gaussian mode
(`GaussianSettings(mode=1)`) needs a real uncertainty measure, and a column full
of ones is indistinguishable from a real one by inspection. With `mode=0` every
localization gets one fixed width and none of this matters.

### ImSwitch2 — what our localizations look like

A numpy record array, canonical schema, **nanometres throughout**:

| Column | dtype | Meaning |
|---|---|---|
| `frame` | `i4` | index into the source stack |
| `x_nm`, `y_nm`, `z_nm` | `f4` | position in nm |
| `sigma_x_nm`, `sigma_y_nm`, `sigma_z_nm` | `f4` | fitted width in nm |
| `photons` | `f4` | fitted integrated intensity |

The field order deliberately mirrors your `LOCS_DTYPE`, and we already ship a
`to_napari_storm_recarray()` that renames onto your pixel schema and divides by
the pixel size. **So a working integration is available today** — the request
below is about not needing it.

`sigma_x_nm`, `sigma_y_nm` and `photons` are genuinely fitted, so
`sigma_present` and `photon_count_present` would be honest declarations.
`sigma_z_nm` is **not** fitted — see §2.2, which is a trap rather than a
preference.

### ImSwitch2 — the one contract change we would like

Your offer above — *"If ImSwitch2 stores camera pixels, say so rather than
converting"* — is the right design, and it is exactly why our nm-native table
can meet you without a conversion. But the offer currently only covers
**positions**. The planner reads uncertainty and photons by hardcoded,
pixel-native field names, reaching past the table into the raw records:

```python
# render_planner.py, sigmas()
scale = traits.pixel_size_nm
x = _clipped(np.asarray(rows.records.sigma_x_pixels) * scale, ...)

# render_planner.py, _variable_values()
photons = self._usable(rows, "photon_count", traits)
```

So `LocalizationTable(records, position_columns={"x": "x_nm", ...},
position_scale_nm=1.0)` works for `GaussianSettings(mode=0)` and breaks for
`mode=1`: the fixed-width path never reads the sigma columns, and the variable
path reads names we do not have. The consequence is that variable-width mode —
the mode where our real fitted per-localization widths are worth anything —
forces a full array copy and a nm → px → nm round trip in `f4`, to arrive back
at the numbers we started with.

**What we would like: the same treatment positions already get.**

```python
table = LocalizationTable(
    records,
    position_columns={"x": "x_nm", "y": "y_nm", "z": "z_nm"},
    position_scale_nm=1.0,
    sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
    sigma_scale_nm=1.0,
    photon_column="photons",
    copy=False,
)
```

with accessors mirroring the one that already exists —
`TableSelection.coordinate_nm(axis)` gains `sigma_nm(axis)` and `photons()`,
cached the same way — so the planner stops touching `rows.records` entirely:

```python
x = _clipped(rows.sigma_nm("x"), settings.var_sigma_min_xy_nm)
```

**Why this is worth doing for napari-storm, not only for us.** The position
path already established that unit conversion and column naming belong to the
table. `sigmas()` and `_variable_values()` are the two places that still reach
around it, and closing them is the refactor you already did once, finished. It
also decouples `DatasetTraits.sigma_present` and the `has_field` checks from
one vendor's column names, which is the same argument that motivated
`position_columns` in the first place.

**Three things that make it smaller than it looks:**

* `_variable_values()` is already unit-invariant. Both branches end in
  `_normalized(...)`, so px² and nm² produce identical output. Only `sigmas()`
  actually depends on the unit, and there the change *removes* the
  `* traits.pixel_size_nm` multiply rather than relocating it.
* `MIN_USABLE_UNCERTAINTY = 1e-3` is a sentinel, not a physical floor, and the
  real floor (`var_sigma_min_xy_nm`) is applied afterwards in nm and does not
  move. Repairing post-conversion rather than pre-conversion is therefore
  behaviour-preserving.
* Defaults keep every existing reader working:
  `{"x": "sigma_x_pixels", ...}` and `"photon_count"`.

**The one wrinkle, stated plainly.** `sigma_scale_nm` has no honest default.
Positions could default to 1.0 because the readers pass the pixel size
explicitly; sigmas currently get theirs from `traits.pixel_size_nm` at plan
time. Two ways out:

1. **Explicit** — each existing reader passes `pixel_size_nm` at construction,
   the way it already does for positions. Cleanest end state, costs one line
   per reader.
2. **Deferred** — `sigma_scale_nm=None` means "use `traits.pixel_size_nm` when
   planning". Zero reader changes, but it keeps one unit decision split across
   two objects, which is the thing worth removing.

We would take (1), but it is your reader surface and your migration cost, so
(2) is fine by us. Either way we pass 1.0.

**If you would rather not do this at all,** say so and we will ship v1 on
`to_napari_storm_recarray()`. It is a copy and a round trip, not a blocker. We
are raising it because the asymmetry reads like an oversight rather than a
decision, and because we are the first external caller to hit it.

### 2.2 ImSwitch2 — `sigma_present` is not expressive enough, and the failure is hard

Our 2D localizer writes `sigma_x_nm` and `sigma_y_nm` and leaves `sigma_z_nm`
at its zero fill; 3D astigmatism is not implemented on our side yet. Meanwhile
`sanitize_positive` raises `InvalidLocalizationData` when *every* value in a
column is unusable, and `_variable_values()` reads the z sigma whenever
`zdim_present and sigma_present`.

So `mode=1` + `zdim_present=True` + `sigma_present=True` over a table with an
all-zero z sigma is a **hard error, not a degraded render**. Today that
combination cannot arise for us, because nothing produces 3D localizations yet.
It becomes reachable the moment either (a) we implement 3D astigmatism, or (b) a
user loads an external 3D file with fitted z positions but no fitted z width —
which is an entirely ordinary file.

Declaring this correctly is ours to do, and we will: `sigma_present` will mean
*every declared sigma axis is real*, not *some sigma columns exist*. But it
argues that `DatasetTraits` may want **per-axis sigma presence** rather than a
single flag, because "x and y are fitted, z is not" is a real and common state
that the current boolean cannot express — and the way it currently fails is an
exception rather than a fallback. Your call; flagging it, not asking for it.

---

## 3. Lifecycle

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
    transform=WorldTransform(scale=(1, 1, 1), translation_nm=(5000.0, 0.0, 0.0)),
)
```

**Appearance** is separate from what is drawn, so changing it rebuilds nothing:

```python
from napari_storm.core import LayerAppearance
renderer.set_appearance(1, LayerAppearance(colormap="green", opacity=0.5))
```

`None` fields mean "leave this alone", so a control owning one slider sends only
what it changed.

**Several channels** are just several ids on one renderer:
`renderer.open(7, ...)`, `renderer.open(9, ...)`.

**Closing** releases the layer and its GPU resources: `renderer.close(1)`, or
`renderer.close_all()`.

`update` keeps the dataset's resources and `open` replaces them. The
distinction is load-bearing — recreating layers on every change is the leak this
architecture was built to fix, so an acquisition loop must call `update`.

### ImSwitch2 — who owns the lifecycle, and how we intend to map it

**We do.** ImProcess has an explicit results list with add, select,
`removeRecon` and `removeAllRecon`. We will tell you when a dataset opens and
closes; you do not need to watch anything.

The mapping we intend, so you can tell us if it fights your model:

| ImProcess event | napari-storm call |
|---|---|
| localization result created | nothing (lazy) |
| result selected, first time | `open(next_id, request)` |
| result selected, already open | set layer visible |
| result deselected | set layer hidden — **not** `close` |
| result data changed in place | `update(id, request)` |
| result removed from list | `close(id)`, id retired forever |
| viewer torn down | `close_all()` |

Ids come from a monotonic counter and are never reused, per §1.

The reason for hide-rather-than-close is worth stating, because it is the one
place our architecture and yours genuinely collide. **Our display path is
stateless.** Selecting a result clears every managed layer and rebuilds it from
a spec carrying a plain array. Driving your renderer from that mechanism would
open and close on every click in the results list — precisely the churn §3 says
this architecture exists to prevent — and our spec objects coerce their payload
with `np.asarray()`, which a localization table would not survive.

So napari-storm layers will live in a **parallel retained channel** outside our
normal display mechanism, keyed by result identity, and only explicit removal
closes a dataset. Nothing is required from you for this; we are describing it
so that §3's assumptions and ours are on the record together.

**One open question back to you (see §8):** our `smlm-filter` produces a *new*
result rather than mutating a mask, so each filter would spend a fresh dataset
id and a full replan, and your cheap `set_filter_mask` path would go unused. We
can live with that for v1. If you would rather we drive masks, that is a change
on our side, not yours — we would like your view on whether it matters.

---

## 4. Constraints worth knowing before you design around them

**The renderer is main-thread and same-process.** Planning (`LocalizationTable`,
`RenderPlanner`, filtering, coordinate conversion, export) runs anywhere —
importing `napari_storm.core` with napari, Qt and VisPy made unimportable is
enforced by a test in a subprocess. Everything from `renderer.open` onward must
be on the Qt GUI thread. Building the geometry and handing it to napari is the
larger cost on big datasets, and it cannot be moved off that thread.

**3-D data needs `viewer.dims.ndisplay = 3`.** napari's canvas defaults to 2-D,
where it shows a single slice. Our dock widget sets this; a host must do it.

**Never pass `colormap=None`.** It defaults to `"gray"` precisely because "no
colormap" is not neutral: napari assigns an arbitrary unnamed colormap, and the
instanced backend resolves that to black. Every piece of state then reports
itself healthy while the canvas stays empty. It is the worst thing to debug in
this codebase.

**The memory budget is the host's job when driving the core directly.** The
dock widget applies it; a host bypassing the widget does not get it for free:

```python
from napari_storm.memory_budget import (default_render_budget_mb,
                                        max_localizations_for_budget)
table.limit_active_to(max_localizations_for_budget(default_render_budget_mb()))
```

Note the two masks, because the distinction matters for anything ImSwitch2
saves. `filter_mask` is what the user selected; the display limit above it is
what the GPU can afford; `active_mask` is the intersection. **Anything leaving
the process — an export, a saved file, a reported count — must read the
filter set** (`plan(..., selection=FILTERED)`). Only the renderer sees the
display set. Collapsing them writes a subsample of someone's data to disk
because their graphics card was busy.

> **ImSwitch2 — the export hazard does not apply to us, by construction.**
>
> Everything we export or save reads our own `LocalizationResult.locs`, never
> the renderer. There is no path by which a GPU display budget can reach a file
> we write. So we will apply `limit_active_to` without the usual care about it
> leaking into saved data — the budget is purely a display concern on our side.
>
> Threading is understood and matches what we already do: planning goes on our
> existing off-thread reconstruction worker, and every
> `open`/`update`/`set_appearance` call marshals to the GUI thread.
>
> `ndisplay = 3` we will set. We expect some friction there against a
> permanently-present 2-D image layer in our viewer, but that one is ours to
> solve and we are not asking you to accommodate it.

---

## 5. Live acquisition — what exists, and what does not

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
  re-uploads. At 5M localizations that measured 0.16 s on the instanced
  backend, which bounds a realistic refresh rate rather than making it free.

For a live reconstruction at a few Hz this is adequate. For a high-rate stream
it is the obvious thing to improve, and an append path (`update_localizations`
with a delta rather than a replacement) is on the roadmap as part of the same
adapter work. **How fast does ImSwitch2 need to push, and in what batch size?**
That number decides whether we need the delta path before or after a first
working integration.

> **ImSwitch2 — both live and post-hoc; the delta path is not a v1 prerequisite.**
>
> Post-hoc is the common case. Live runs through `SmlmLiveSession`, which
> localizes each incoming chunk and accumulates into a growing table — exactly
> the rebuild-and-update pattern described above.
>
> **We do not have a measured rate yet, and would rather owe you a real number
> than invent one.** We will instrument the live session and come back with
> localizations/s and batch size. Our expectation is a few Hz of table growth,
> which §5 already calls adequate — so please do **not** schedule the delta
> append path ahead of v1 on our account. If instrumentation contradicts that,
> we will say so before you have spent anything on it.
>
> Noted on `set_records` resetting the selection: we re-apply after every
> growth step.

---

## 6. What else is available

* **Scene persistence.** `save_scene` / `load_scene` write a small JSON file
  recording the *decisions* — per-dataset transform, appearance, Gaussian
  settings, reference-image placement, camera — and never the localizations.
  Reloading re-reads the source files and re-applies the decisions.
* **Calibrated export.** `plan_export` describes the output — shape, bytes,
  warnings — before a byte is written; `write_ome_tiff` streams tiles, so peak
  memory is one 1024² tile whatever the file size. It never downsamples. Both
  are host-free, so ImSwitch2 can export without a viewer at all.
* **Reference images.** Widefield/confocal images can be placed alongside a
  reconstruction with a pixel size and an offset.

> **ImSwitch2 —** all three are interesting, none are v1. Reference images map
> cleanly onto our widefield results and are the most likely second step.
> `NullRenderer` is what we will test against headlessly, so please keep it
> exported; it is the reason we can cover this integration in CI without a GL
> context.

---

## 7. What we need from the ImSwitch2 side

1. **Desktop Qt or browser/headless?** (§0.) The single blocking question.
2. If Qt: **does ImSwitch2 already own a napari viewer** we can be handed, or
   would we create and dock one?
3. **Which Qt binding and version**, and which napari version, if any is
   already pinned. We support `napari>=0.4,<0.8` and no fixed binding.
4. **What do ImSwitch2's localizations look like** at the point they would be
   handed over — array/dtype, column names, units, and whether uncertainty or
   photon counts are present?
5. **Live or post-hoc?** If live: expected localizations per second, batch
   size, and total per acquisition. (§5.)
6. **Who owns the lifecycle** — does ImSwitch2 tell us when a channel opens and
   closes, or does it expect us to watch something?
7. **Is napari-storm an optional dependency** of ImSwitch2, or a hard one? We
   would recommend optional, with the adapter importing lazily.

Answers to 1, 4 and 5 are enough to draft the adapter.

### ImSwitch2 — answers

| # | Answer |
|---|---|
| 1 | **Desktop Qt.** Embedding, not a sidecar. (§0) |
| 2 | **Yes, we own one** and would hand it over — `naparitools.EmbeddedNapari` inside `ReconstructionView`. Do not create or dock a viewer. |
| 3 | **PyQt5 5.15.14** via `qtpy>=2.0`; **napari `>=0.7.0`** (0.7.1 installed); numpy 2.4.4. Overlap with your pin is 0.7.x only. (§1) |
| 4 | Canonical **nm** recarray: `frame`, `x/y/z_nm`, `sigma_x/y/z_nm`, `photons`. Sigma (x, y) and photons genuinely fitted; **z sigma is not**. (§2, §2.2) |
| 5 | **Both.** Live via `SmlmLiveSession`. Rate unmeasured — number owed, delta path not a prerequisite. (§5) |
| 6 | **We do.** Explicit results list; mapping table in §3. |
| 7 | **Optional**, lazily imported, behind a feature flag, falling back to our existing preview when the import fails. |

---

## 8. Open items after round 2

For the napari-storm side to answer:

1. **Symmetric `sigma_columns` / `photon_column` / `sigma_scale_nm`?** (§2.)
   The one actual request. If yes, variant (1) explicit or (2) deferred. If no,
   we ship v1 on the pixel round trip and nothing is blocked.
2. **Per-axis sigma presence in `DatasetTraits`?** (§2.2.) Not a request — but
   the current single flag turns an ordinary file into an exception, and you
   may want to know that before someone hits it.
3. **Is 0.7 in your CI matrix?** (§1.) The supported intersection between our
   pins is one minor version wide.
4. **Should ImProcess drive `set_filter_mask` instead of producing a new
   result per filter?** (§3.) Change on our side; we want your view on whether
   the id churn and full replans are worth avoiding.

Owed by ImSwitch2:

5. **Measured live rate** — localizations/s and batch size, from an
   instrumented `SmlmLiveSession`. (§5.)

Answers to 1 are enough for us to start; 2–4 can trail.

---

## 9. Round 3 — napari-storm replies

Your two code claims both reproduce exactly. A third — the one your request
rests on — does not, and correcting it made the change larger than either of us
scoped. **It is done and merged**, so item 1 is closed rather than answered.

### Item 1 — `sigma_columns` / `photon_column` / `sigma_scale_nm`: done

The asymmetry was an oversight, as you guessed. Positions could always be
declared; widths and photons could not, and the argument that motivated
`position_columns` never stopped applying at the position columns. Landed:

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

`TableSelection` gained `sigma_nm(axis)` and `photons()`, cached in the same
three stages as `coordinate_nm`. The planner no longer touches `rows.records`
at all. **Drop `to_napari_storm_recarray()` — pass your canonical recarray
directly.**

**Neither of your two variants, and you can pass nothing.** `sigma_scale_nm`
defaults to *following* `position_scale_nm`. No reader among ours fits widths in
a different unit from the one it stores positions in, so the honest default is
not 1.0 and not a second explicit argument — it is "the same unit as everything
else in this dataset". That gets variant (1)'s end state, one unit decision per
dataset, at zero reader migration cost, and a reader that learns its pixel size
late updates both together instead of leaving widths converted with the old
value. You still pass `sigma_scale_nm=1.0`; it is now belt-and-braces rather
than required.

### The claim that does not hold, because it changes what you get

> *"`_variable_values()` is already unit-invariant. Both branches end in
> `_normalized(...)`, so px² and nm² produce identical output."*

True only when no row needs repair:

```
NO repaired rows  -> unit-invariant: True
WITH 5 zero rows  -> unit-invariant: False
```

`MIN_USABLE_UNCERTAINTY = 1e-3` was an **absolute** sentinel substituted
*before* unit conversion. Under your schema a repaired row meant 1e-3 nm rather
than 1e-3 px — a hundredfold narrower width, and since the weight is
`1 / product`, ~10⁶× the intensity. Measured, 5 dead rows in 5000 at 100 nm/px:

| sigmas stored as | value range fed to contrast | real data occupies |
|---|---|---|
| pixels, before | `[0, 158]` | bottom 1.68% |
| **nm, before** | **`[0, 15800]`** | **bottom 0.017%** |

Note the first row — this was already wrong for our own readers; your schema
would have multiplied the error by the pixel size. And the median was correct
throughout, because the 99th-percentile clip restores the scale, so nothing in
any summary statistic would have shown it. You would have seen contrast limits
that refused to come right and had no reason to suspect the unit.

Photons keep the old sentinel; a count carries no length unit, so your
reasoning was right for that branch.

> **Round 3 was wrong about the sigma repair, and §10 corrects it.** The row
> that stood here — *"either, after: `[0, 3.79]`, 71.9%"* — was measured at a
> fixture width sitting just above the floor, which is the one place the
> remaining half of the bug is invisible. Repairing to the floor fixed the unit
> dependence and left the blowout. Do not use these "after" figures.

### Item 2 — per-axis sigma presence: your instinct was right, and it is worse

You called §2.2 "flagging, not asking". The adjacent case is sharper. A 2-D
dataset with `sigma_present=True` and **no z sigma column at all**:

```
AttributeError: recarray has no attribute sigma_z_pixels
```

Not even our own error type. `sigmas()` read the axial column unconditionally
whenever `sigma_present`, while `_variable_values()` guarded on `zdim_present` —
the two disagreed, on a file neither had reason to refuse. Fixed: an absent
axial width now falls back to the declared floor, which is exactly what a
present-but-zero-filled column already clipped to.

**So your zero-filled `sigma_z_nm` was the only thing preventing that crash.**
If you had "cleaned up" the schema by dropping a column you do not fit, you
would have hit it. That is now safe either way — keep the column or drop it.

The all-dead-column case you actually described (§2.2, `zdim_present=True` with
an unfitted z) still raises, and we think it should: declaring an axial width
you never fitted is a wrong declaration, and the error names the column. Per-axis
flags on `DatasetTraits` remain open — the crash that made them urgent is gone,
so this is now a modelling question rather than a bug, and we would rather see
your 3-D astigmatism land first and design against something real.

### Item 3 — CI

Nothing in CI pins napari: it installs `.[dev,pyqt6]` and pip resolves the
newest under `<0.8`, which is 0.7.1 today. So 0.7 is what CI actually tests, but
incidentally — and the declared 0.4 floor is tested by nothing. You are right
that the intersection works by coincidence. We are raising the floor to what we
genuinely support rather than leaving a range we do not exercise; `<0.8` stays.
Python is now 3.10–3.12, which matches your environment.

### Item 4 — filters: keep what you have

Do not change your side for v1. A new result per filter carries provenance a
mask cannot, it matches an architecture you have already committed to, and the
cost is one replan at a size where a replan is cheap. `set_filter_mask` exists
for a slider dragged at interactive rates; that is not what your filter is.
Revisit only if you add a continuous control.

### Confirmations

* `NullRenderer` stays exported and stays in the contract tests. Worth knowing:
  **it will catch all three bugs above**, because every one of them is
  planner-side and fires before a renderer is touched. Headless CI that calls
  `plan()` is a real gate, not a smoke test.
* Your §3 lifecycle mapping is right, including hide-rather-than-close. That is
  what the `open`/`update`/`close` split is for, and the parallel retained
  channel is the correct place to put it.
* Reference images as the second step suits us; they need no core change.
* Take your time on the live rate. §5 stands: do not treat the delta path as a
  prerequisite, and we will not build it until your number says otherwise.

### Still owed by us

Nothing blocking. `DatasetTraits` per-axis sigma flags, if your 3-D work wants
them.

### Still owed by you

The measured live rate (§5, item 5). Unchanged and unhurried.

---

## 11. Round 5 — the transposition

**Confirmed, and fixed at the root.** Your test reproduces here exactly.
`coordinates()` now returns `(z, y, x)`.

You were also right that it could not be worked around host-side, and right
about where the fix belonged. What neither of us saw is that it was costing
more than the misregistration you hit.

### A second bug, from the same root

`sigmas()` has always returned `(z, y, x)`, and the backends broadcast sigmas
against coords and upload them column-wise. So while `coordinates()` returned
`(z, x, y)`, **an anisotropic PSF had its widths applied to the wrong lateral
axis**:

```
true: sigma_x=300 nm (wide), sigma_y=50 nm (narrow)
width applied along x = 0.167   <- the narrow one
width applied along y = 1.000   <- the wide one
```

That is wrong science rather than wrong framing, and it survived because
sigma_x and sigma_y are nearly equal in most real data. It is fixed by the same
swap, for free, because the two orders now agree.

### The export was already right, which settled it

`image_export.channel_for` reordered coordinates to `(z, y, x)` and deliberately
left sigmas alone, and the rasterizer reads them that way. **So the image on
screen was transposed relative to the OME-TIFF written from it.** Once we
looked at that, there was no question which side was wrong.

There were three compensations for one root cause, and the third is our
favourite: `pyqt/image_layer_controls.py` transposed every reference image on
the way in, with a docstring reading *"The lateral swap is a fix, not a
convention... the 90-degree rotation controls existed largely to undo it by
hand."* Someone hit this before you and patched the image instead of the
coordinates. Both compensations are now deleted rather than adjusted.

### What changed

| | before | after |
|---|---|---|
| `RenderPlanner.coordinates` | `(z, x, y)` | `(z, y, x)` |
| `RenderPlanner.sigmas` | `(z, y, x)` | unchanged — now agrees |
| `channel_for` reorder | present | deleted |
| reference-image lateral swap | present | deleted |
| `ReferenceImageEntry.offset_nm` | `(z, x, y)` | `(z, y, x)` |

Also moved with it: the render-range preview box, the grid plane, the rotation
controls' axis permutation, and the XZ/YZ reference orientations, whose
remaining lateral axis travels with everything else. The scalebar needed
nothing — it is a symmetric marker with no axis labels.

Nothing in the interface you build against changes shape. `WorldTransform` is
still keyed by axis *name*, so your placement code is unaffected; only the
column order of the arrays moved.

### What we would like from you

**Re-run your transposition check.** It should now report `False`, and your ROI
shapes and points overlays should land on the reconstruction without anything
host-side. If it does not, we would rather hear it before this is anywhere near
a release.

Your framing fix reads correct to us and needs nothing from our side.

### Two notes

Your `verify.py` / `contrast.py` offer stands and we would still like them
landed — and this round is the argument for a third. A check that a
reconstruction is not transposed against an ordinary napari layer is exactly
the kind of thing our own tests could not see, because inside our widget the
error was self-consistent. There is now one here
(`test_coordinates_are_in_napari_axis_order`, pinned against a real napari
Points layer rather than against a convention of ours), but yours would be the
one that catches it from the outside.

No migration was needed for `offset_nm`: nothing is deployed yet, so the field
simply changes meaning rather than gaining a version.

---

## 10. Round 4 — the sigma repair, corrected

**Your finding is right and our §9 numbers were not.** Both halves reproduce
here. The correction is landed.

### What we got wrong

The `[0, 3.79]` / 71.9% figures came from a fixture whose widths were 10–60 nm
— two to fourteen times the floor. You identified the missing variable exactly:
how far the real widths sit above it. Sweeping, with a *fully* failed fit
(zeros on every axis, which is what a failed fit actually writes — our sweep
initially zeroed only x and understated the effect further):

| real sigma_xy | × floor | range, floor-repair | real occupies |
|---|---|---|---|
| 4.5 nm | 1.1× | `[0, 2.16]` | 100% |
| 30 nm | 7.1× | `[0, 2.16]` | 58.5% |
| 100 nm | 23.5× | `[0, 1.22e4]` | 0.017% |
| 140 nm | 33.0× | `[0, 3.35e4]` | 0.006% |

Your diagnosis is the part worth keeping, because it is structural rather than
numerical: **`sigmas()` and `_variable_values()` repair the same dead row for
different questions, and we gave them one answer.** How wide to draw an unknown
row — the floor is right. How bright it should be — the floor is the single
worst answer available, because the weight is `1 / product` and the narrowest
width is the weight-maximising one. We closed the unit dependence and left the
blowout, then measured at the one width where those are indistinguishable.

### The correction

The weight path now repairs to the **median of the usable rows** for that
column. A median is unit-covariant, so the units still agree; and it is the
honest reading of a failed fit — no information about brightness, so take the
typical value rather than an extreme one. The width path is unchanged: the
floor is still the right answer to its question.

Measured across the same span, with the units checked at each point:

| real sigma_xy | × floor | range, median-repair | real occupies | units agree |
|---|---|---|---|---|
| 4.5 nm | 1.1× | `[0, 2.16]` | 100% | yes |
| 30 nm | 7.1× | `[0, 2.16]` | 100% | yes |
| 100 nm | 23.5× | `[0, 2.16]` | 100% | yes |
| 200 nm | 47.1× | `[0, 2.16]` | 100% | yes |

The range is now independent of the fitted width, which is what a scale-free
value model should have produced all along. Dead rows sit at the median and
cannot set the scale that every other row is normalized against.

`test_dead_rows_do_not_set_the_scale_at_any_fitted_width` is parametrized over
4.5 / 30 / 100 / 200 nm, because a single-width test has already been passed by
a broken implementation once.

### The percentile divide-by-zero

Also real, also fixed. `_normalized` puts the majority at zero when ≥99% of
rows share the minimum, the 99th percentile is then zero, and dividing by it
produced NaN — surfacing as *"render values has no positive finite maximum"*,
which names neither the column nor the cause. There is nothing to clip against
in that case, since `_normalized` has already produced `[0, 1]`, so the clip is
now skipped. Your reachability argument was right: a fitter writing one nominal
width for most rows and fitting the rest gets there. Pinned by
`test_a_degenerate_width_distribution_plans_rather_than_raising`.

### On the process

You were right that this should have been run rather than reasoned about — and
that cuts both ways. Your unit-invariance claim and our floor-repair fix failed
the same way: each was argued from the shape of the code and each was wrong in
a régime neither of us had measured. The difference in outcome was only that
yours was checked. Publishing the sweep rather than a single number is the
actual lesson, and §10 does that.

**Yes, please land `verify.py` and `contrast.py`** on
`feat/napari-storm-viewer` as skip-if-missing contract tests. A check that
fails on our side when we regress your schema is worth more to us than anything
in this document. If you would rather they lived here instead, we will take
them — say which.

### Net effect on you

None. Interface unchanged, so nothing you have built moves. Rendered values
change for datasets containing failed fits, in the direction of your report.
The live rate remains the one open item, and it is still not urgent.

---

## 10. Round 4 — ImSwitch2 replies

We ran our canonical recarray through the merged API rather than reading it.
Item 1 verified end to end. Your correction of our unit-invariance claim is
right and we were wrong. One thing in the same table does not reproduce, and it
is the part that decides whether the bug is actually closed.

### Item 1 — verified, and `to_napari_storm_recarray()` is dropped

Our nm-native recarray passed straight in against the same data through the old
pixel round trip, both planned, outputs compared. 5000 rows, 100 nm/px:

| case | max Δ coords | max Δ sigmas | max Δ values |
|---|---|---|---|
| `mode=0` fixed, clean | 1.95e-03 nm | 0 | 0 |
| `mode=1` variable, clean | 1.95e-03 nm | 5.96e-08 | 2.98e-07 |
| `mode=1` variable, 5 dead rows | 1.95e-03 nm | 5.96e-08 | 2.98e-07 |

The two paths agree. Note the third row in particular — that is the case that
diverged before your fix, so the fix does what it says. The residual coordinate
difference is the `f4` round trip through pixels at 51 µm field, which is
exactly the loss the nm-native path removes; it is now gone from our side.

`sigma_scale_nm` following `position_scale_nm` is better than either variant we
proposed, and the reasoning — one unit decision per dataset, updated together —
is right. We pass nothing.

**Our claim was wrong.** `_variable_values()` is not unit-invariant once a row
is repaired, for the reason you give: an absolute sentinel substituted before
conversion does not scale with the column. We asserted it as one of three
reasons the change was small, and it was load-bearing for your decision. It
should have been checked rather than reasoned about.

### The number that does not reproduce: repairing to the floor does not restore contrast

Your table's "after" row reads `[0, 3.79]`, real data occupying 71.9%, in both
units. The unit agreement reproduces. The 71.9% does not, and it appears to
depend on a property of the fixture that the table does not state: **how far the
real widths sit above the floor.**

Sweeping exactly that, 5000 rows with 5 zero-width rows, realistic fit-to-fit
scatter, `var_sigma_min_xy_nm = 4.248 nm`:

| real sigma_xy | × floor | 2-D range | real occupies | 3-D range | real occupies |
|---|---|---|---|---|---|
| 4.5 nm | 1.1× | `[0, 1.13]` | 100.0% | `[0, 1.73]` | 70.0% |
| 6.0 nm | 1.4× | `[0, 1.70]` | 65.9% | `[0, 4.82]` | 25.5% |
| 10 nm | 2.4× | `[0, 6.39]` | 17.5% | `[0, 24.9]` | 4.9% |
| 20 nm | 4.7× | `[0, 28.5]` | 3.9% | `[0, 200]` | 0.6% |
| 50 nm | 11.8× | `[0, 179]` | 0.6% | `[0, 3242]` | 0.04% |
| **100 nm** | **23.5×** | `[0, 730]` | **0.15%** | `[0, 2.49e4]` | **0.01%** |
| **140 nm** | **33.0×** | `[0, 1420]` | **0.08%** | `[0, 6.80e4]` | **0.00%** |

Your published figures land in the first row: at 4.5 nm widths and 3-D we get
`[0, 1.74]` and 73.9%, which is your `[0, 3.79]` / 71.9% to within fixture
noise. But a fitted PSF width is 100–200 nm — 25–50× the floor — and there the
real data occupies **0.01–0.15%** of the range. That is the same order as the
1.68% you were fixing.

**Why.** The floor is now shared by `sigmas()` and `_usable_sigma()`, which is
what made the units agree. But a repaired row is thereby drawn at the *smallest*
width the renderer permits, and the weight is `1 / product`, so the tightest row
is the brightest. A dead row is still the brightest thing in the image; it is
just now equally so in both units.

The two call sites are answering different questions with one substituted value:

* `sigmas()` asks *how wide do we draw a row whose width we do not know?* — the
  declared floor is a good answer.
* `_variable_values()` asks *how bright is a row whose width we do not know?* —
  and the floor is the worst available answer, because it is the value that
  maximises the weight.

A row with no usable width is not a confident detection, so it should be
weighted like the dimmest data rather than the brightest. Weighting repaired
rows at the *maximum* observed width, or excluding them from `_normalized` and
the percentile clip and assigning them the resulting minimum, would both do it.
The floor stays right for `sigmas()` either way.

We are not asking for a specific fix — it is your value model. We are saying the
bug you found is half closed: the unit dependence is gone, the contrast blowout
is not, and the fixture that measured it happens to sit at the one width where
the two are indistinguishable.

**Smaller, adjacent:** `_normalized` followed by `/ np.percentile(values, 99)`
divides by zero when ≥99% of rows share the minimum value, yielding `nan` and
then `InvalidLocalizationData: largest is nan`. We hit it with a constant fitted
width — plausible from a fitter that writes a nominal sigma it does not vary,
which is the "column full of ones" case `DatasetTraits` already warns about
declaring. Arguably a wrong declaration rather than your bug, but the error
names neither the column nor the cause.

### Items 2, 3, 4 — agreed

**2.** Confirmed against the merged code: a 2-D table with `sigma_x`/`sigma_y`
and no z sigma column at all now plans cleanly (5000 rows, size 992.51 nm)
where it previously raised `AttributeError`. Our §2.2 case — `zdim_present=True`
with an all-zero z sigma — still raises, and names the column:
`every sigma_z_nm value is zero, negative or non-finite`. We agree that is
correct: it is a wrong declaration and the message says so. We will keep the
zero-filled column regardless, since it costs 4 bytes a row and nothing now
depends on it either way. Per-axis flags parked until our 3-D astigmatism is
real — agreed, designing them against nothing would be guesswork.

**3.** Python 3.10–3.12 matches us exactly: we pin `>=3.10` and run 3.12.2.
Raising the napari floor to what you actually exercise is the right call.

**4.** Keeping a new result per filter. Your reasoning matches ours — the
provenance is worth more than the replan, and a filter that is not a dragged
slider does not need `set_filter_mask`. We will revisit only if we add a
continuous control, and we will say so before we do.

### Still owed by us

The measured live rate, unchanged. Nothing else — we consider the contract
settled enough to start building against, independently of how you land the
weighting question above, since it changes values and not the interface.
