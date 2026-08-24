# Step-by-step Guide

## napari-storm visualization & analysis

### 0) Launch napari-storm and import a dataset:

    napari

→ Open Plugins → napari-storm.

Import your dataset by drag & drop or via the Import File button. Closing the file picker without
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

Still in Channel Controls, use the two-handle contrast slider:

Left handle = absolute cutoff (min intensity shown)

Right handle = log-scaled max (expands/compresses the top range)
You can also type exact values in the numeric spin boxes next to the slider.

Tip: Each channel remembers its own settings; toggling Show/Hide is instant.

### 3) Pick render mode & adjust Gaussian width

In the main controls, select Fixed-size Gaussian or Variable-size Gaussian (PSF). Then tune the sigma parameters:

Fixed-size: render_fixed_gauss_sigma_xy_nm, render_fixed_gauss_sigma_z_nm

Variable-size (PSF): render_var_gauss_PSF_sigma_xy_nm, render_var_gauss_PSF_sigma_z_nm (with min clamps)

Variable-size mode disables Z color encoding (by design), while Fixed-size allows it.

Under the hood, particles are rendered with Gaussian shading via the napari-particles pipeline.

**Size safety cap.** A Gaussian is drawn as a camera-facing quad, and the cost of drawing it grows
with the area it covers — independently of how many localizations you have. A single splat spanning
many times the field of view can therefore stall the GPU on a dataset of a thousand points. The quad
is clamped to half the current field of view, and a notification tells you when that happened. The
clamp crops the outer tail of a Gaussian that is already a flat wash at that size; it does not
change the shape of anything you can actually resolve.

### 4) (Optional) Enable Z color encoding

For 3D datasets in Fixed-size Gaussian mode, toggle Z color encoding to map depth to color. If you switch to Variable-size, Z color encoding is automatically turned off.

### 5) Add a decorator layer (Grid plane) & scalebar

Activate the Grid plane in the Decorators tab, then adjust:

Line distance (µm)

Line thickness

Z position

Color & opacity

The grid is created as a vectors layer and updates with your render ranges and view. You can also toggle the Scalebar from the same area.

### 6) Adjust the render range & view

Use the Render Range sliders (X/Y/Z) to restrict what part of the dataset is drawn. The camera can be centered to the selected range; switching views (XY/YZ/XZ) is available in the controls.

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

### 8) Adjust values (offset / rescale) & export

In Data Adjustment:

Select a parameter and apply add offset or rescale; the view refreshes automatically.

Use Export current dataset as .ns to save your adjusted data.

### 9) Save or export processed datasets

After filtering/adjustment, save your results from the respective panels (e.g., the .ns export).

Tips

Detach tabs by dragging them out to see multiple panels at once.

For STORM/PALM datasets, try Variable Gaussian mode to incorporate uncertainty or photon counts into rendering.

Hold Shift + drag to pan the canvas smoothly.


