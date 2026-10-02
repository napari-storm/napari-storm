# Step-by-step Guide

## napari-storm visualization & analysis

### 0) Launch napari-storm and import a dataset:

    napari-storm

opens napari with the dock already docked. Alternatively start `napari` and open
**Plugins → Napari STORM**.

Import your dataset by drag & drop, or with **Import Localization File…** in the
Data Controls tab. Closing the file picker without
choosing a file leaves the current session unchanged.

Reading the file happens in the background. The window stays usable, a progress dialog appears for
anything slower than a moment, and **Cancel** abandons the import without touching what is already
loaded. A cancelled import cannot interrupt a read that is already in flight — the file finishes
being read and the result is then discarded — so cancelling never leaves a half-loaded dataset.
Importers that need to ask you something (a missing pixel size, for example) still ask, and the
question appears as a normal dialog.

Building the layer itself still happens on the interface thread, so the window pauses briefly at the
end of a large import.

### Optional: overlay and orient a reference image

Import a reference image and enter its physical pixel size and X/Y/Z position. Adding or removing a
reference does not change the camera projection. Grayscale references also provide a two-handle
contrast slider and a colormap selector; napari does not apply those scalar controls to RGB images,
so they are omitted for RGB. RGBA references instead provide a uniform opacity slider that
multiplies their existing per-pixel alpha.

XY, XZ, and YZ references use napari's embedded-plane renderer with depth testing disabled rather
than ray-marching a one-voxel volume. This keeps planar overlays stable while the camera moves;
genuine 3D reference stacks retain normal volume rendering.

Each reference-image control has paired **↶/↷** buttons for the physical X, Y, and Z axes. Every
pair is placed directly beside its matching position field. Each click rotates the image by exactly
90° while keeping its centre fixed in world coordinates. A rotation switches napari to the 3D
display because an X- or Y-axis turn cannot be represented as a 2D slice.

### 1) Choose a colormap (per channel)

In the Channel Controls panel (one per dataset/channel), use the Colormap dropdown to pick a palette.
Use **Unload** to remove only that dataset, its layer, and its associated filter/adjustment state;
the other datasets remain loaded. The Data Controls tab becomes vertically scrollable when many
channels are open.

### 2) Adjust contrast

Still in Channel Controls, use the two-handle contrast slider. It acts on the
reconstruction itself -- the Gaussians summed where they overlap -- so with
fixed-size Gaussians its numbers count overlapping localizations:

Left handle = **Cutoff**: everything where fewer localizations overlap than this
is hidden. Raise it to 1.5 and lone localizations disappear while clusters stay.

Right handle = the top of the range: where at least this many overlap is drawn
at full brightness. The default, 1, saturates the peak of a single Gaussian.

Both handles sit on the same logarithmic scale, from 0.01 to 100, and you can
type exact values in the spin boxes next to the slider. With variable-size
Gaussians each localization counts by its precision instead of as one.

Opaque point-cloud styles from the Decorators tab have no sum to act on; with
them the slider maps each marker's own value onto the colormap, and keeps its
own handle positions for when you switch back.

Tip: Each channel remembers its own settings; toggling Show/Hide is instant.

### 3) Pick render mode & adjust Gaussian width

In Data Controls, under **Rendering options:**, choose **Fixed-size gaussian** or
**Variable-size gaussian**. Four entry fields sit beneath it:

| Field | Fixed-size gaussian | Variable-size gaussian |
|---|---|---|
| **FWHM in XY [nm]:** | the width every localization is drawn at | the PSF width the fitted uncertainty is scaled against |
| **FWHM in Z [nm]:** | as above, axially | as above, axially |
| **Min. FWHM in XY [nm]:** | unused | floor, so a tiny fitted width cannot vanish |
| **Min. FWHM in Z [nm]:** | unused | floor, axially |

!!! note "These fields are FWHM, not σ"

    Every one of these four entries is a **full width at half maximum** in
    nanometres, as the labels say. The renderer works in σ and converts on your
    behalf (σ = FWHM / 2.354). Typing the σ you have in mind draws the point
    about 2.35× wider than you intended, so convert first: a 30 nm σ is a
    70.6 nm FWHM.

**Variable-size gaussian** needs a fitted uncertainty or photon count in the
data to scale each localization by; on a dataset that has neither, every point
falls back to the floor. Selecting it also hides the Z colour-encoding button
and switches the encoding off, because depth is already spoken for. Fixed-size
allows it.

Under the hood, particles are rendered with Gaussian shading through the
renderer described in [How it works](how-it-works.md).

**Size safety cap.** A Gaussian is drawn as a camera-facing quad, and the cost of drawing it grows
with the area it covers — independently of how many localizations you have. A single splat spanning
many times the field of view can therefore stall the GPU on a dataset of a thousand points. The quad
is clamped to half the current field of view, and a notification tells you when that happened. The
clamp crops the outer tail of a Gaussian that is already a flat wash at that size; it does not
change the shape of anything you can actually resolve.

### 4) (Optional) Enable Z color encoding

For 3D datasets in **Fixed-size gaussian** mode, press **Activate Rainbow
colorcoding in Z** to map depth to colour. The button is hidden in
Variable-size mode and the encoding is turned off with it.

### 5) Add a decorator layer (Grid plane) & scalebar

**Rendering Style**, at the top of the Decorators tab, chooses what every
localization is drawn as. **Gaussian (scientific)** is the reconstruction and
the default. The **Alternative visualisations** are for viewing:

* **Markers** -- points, outlined points, rings, spheres, crosses, squares and
  diamonds -- are opaque, one sigma in radius, and a nearer one hides a farther
  one.
* **napari-particles' sprites** -- domes, glow, bubbles, Airy-like, Fresnel, a
  Julia fractal, a polynomial Gaussian and tiles -- glow and add up.
* **Use uncertainty** sizes and shapes each marker by its own localization
  uncertainty, so rings become uncertainty ellipses. It is the same switch as
  **Variable-size gaussian**, and needs data that records an uncertainty.
* **Smallest on screen [px]** keeps a visualisation visible when zoomed out.

Exports always write the Gaussian reconstruction.

Tick **Grid plane activated?** in the Decorators tab, then adjust:

* **Grid line distance [µm]:**
* **Grid beyond data [%]:** — how far the plane runs past the data, as a share
  of each axis' span added at both ends. 0 stops it at the render range, which
  is where it always stopped before.
* **Grid line thickness:**
* **Z Pos:**
* **Grid line color:** and **Grid plane opacity:**

The grid is created as a vectors layer and updates with your render ranges and
view. **Render Range Box** draws the render-range bounds in the same tab, and
the **Scalebar** checkbox and its **Size of Scalebar [nm]:** field are in Data
Controls.

### 6) Adjust the render range & view

Use the **Render range** sliders — **X-range**, **Y-range**, **Z-range** — to
restrict what part of the dataset is drawn, and **Reset Render Range** to put
them back. **Reset view:** offers **XY**, **YZ** and **XZ**, each of which
looks at the plane its label names, keeping the current centre and zoom.

The interface computes global min/max in true nanometre world coordinates so localization channels
and calibrated reference images share one stable frame.

### 7) Filter the data

Open the Data Filter tab:

Choose a property (e.g., x/y/z, photons, sigma).

Use the histogram range and pass-band sliders.

Apply to the current dataset (or all).

Filtering marks localizations inactive in a mask over one canonical table rather than copying the
surviving records into a new array, so a slider drag costs one pass over the data instead of a full
copy of it per gesture. Your loaded data is never modified by filtering — **Reset all filtering**
always restores exactly what was imported.

### Memory budget

Rendered localizations cost about 352 bytes each in host memory before napari's own buffers and the
GPU copy. To keep a large import from taking the process down, the plugin renders at most a **2 GB**
budget's worth — roughly 5.8 million localizations, shared between all loaded datasets. Beyond that
it draws an evenly spaced subsample and tells you how many of your localizations are on screen; the
full dataset stays loaded and filtering still applies to all of it.

Set `NAPARI_STORM_RENDER_BUDGET_MB` before starting napari to change the ceiling, or to `0` to
remove it:

```bash
NAPARI_STORM_RENDER_BUDGET_MB=8192 napari
```

### 8) Adjust values (offset / rescale)

In the **Data adjustment** tab, select a parameter and apply an offset or a
rescale; the view refreshes automatically. The adjusted dataset can be written
back out as a `.ns` file from the same tab.

### 9) Export an image

**Export OME-TIFF…** in Data Controls rasterizes the reconstruction at a pixel
size you choose and writes that calibration into the file. Two things are worth
knowing:

* **It never downsamples to fit.** The pixel size you ask for is the pixel size
  you get, so a small one on a large field of view makes a large file. The
  dialog shows the resulting dimensions, the file size and any warnings, and
  updates them as you change the pixel size — all before anything is written.
* **It exports what your filters left active**, not what the renderer happens
  to be showing. If a large dataset is being displayed as a subsample to stay
  within the memory budget, the export still contains every localization that
  passed your filters.

### 10) Save the scene

**Save Scene…** writes this session's *decisions* — alignment, colours, render
settings, reference-image placement, camera — as a small JSON file. The
localizations are not copied into it; the scene points at the files it came
from, so it stays small and does not duplicate your data.

**Load Scene…** re-applies a saved scene to the datasets that are loaded now. It
does not open files: which data is loaded stays your choice.

Tips

Detach tabs by dragging them out to see multiple panels at once.

For STORM/PALM datasets, try Variable Gaussian mode to incorporate uncertainty or photon counts into rendering.

Hold Shift + drag to pan the canvas smoothly.
