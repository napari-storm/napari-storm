# Embedding napari-storm in another application

napari-storm can render localizations inside any modern napari session
**without its dock widget**. A host application — ImSwitch, an acquisition GUI,
a notebook — supplies the data and drives the lifecycle; the plugin supplies
the Gaussian model and the renderer.

Every example below is executed by `_tests/test_embedding.py`. The code you copy
is the code CI runs, so it cannot quietly stop working.

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

## Two things that will catch you

**3-D data needs `ndisplay = 3`.** napari's canvas defaults to 2-D, where it
shows a single slice. The dock widget sets this for you; a host must do it
itself.

**Do not pass `colormap=None`.** It defaults to `"gray"` precisely so that this
is hard to get wrong — but if you pass `None` explicitly, napari assigns an
arbitrary unnamed colormap and the instanced backend, which samples the
colormap in its own shader, can resolve it to black. Every piece of state then
reports itself healthy while the canvas stays empty.

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

For a worked integration against a specific host, including the questions a
host has to answer before the shape can be chosen, see
[`imswitch-integration.md`](imswitch-integration.md).
