# Filtering and adjusting

## Filtering

![The Data Filter tab with a histogram of frame numbers](images/tab-data-filter.png){ .dock align=right }

The **Data Filter** tab hides localizations by the value of any property the
data has -- photons, precision, frame, sigma, `efo`, `cfr`, x, y or z.

1. Choose the dataset, then the property. The histogram shows its
   distribution; **Number of bins:** sets its resolution, and
   **Adjust graph range:** zooms the histogram onto part of it.
2. Choose a **Filter mode:** -- **Bandpass** keeps what lies inside the range,
   **Bandstop** what lies outside -- and set the range with the slider below
   the histogram.
3. Press **Apply filter to current dataset**, or
   **Apply filter to all datasets** to apply the same range to every loaded
   dataset.

Filters accumulate: apply one on photons and another on precision, and only
localizations passing both are drawn. **Reset all filtering** brings back
everything that was imported.

Filtering never changes or copies your data. It marks localizations as
inactive, which is why applying a filter is fast even on millions of
localizations and why a reset is exact. Exports and the counts in
**File Infos** follow the filters; the render budget does not hide anything
from them.

<div style="clear: both"></div>

## Adjusting values

![The Data adjustment tab](images/tab-data-adjustment.png){ .dock align=right }

The **Data adjustment** tab changes a property of the loaded data: choose the
dataset and the property, a **Math mode:** -- **add offset** or
**rescale** -- and a **Value:**, then press
**Apply adjustment to current dataset**. The view updates.

Unlike a filter, an adjustment changes the values held in memory; the file on
disk is not touched. To undo one, apply the inverse (an offset of minus the
value, or a rescale by its reciprocal) or open the file again.

**Export current dataset as .ns** writes the dataset, adjustments included, to
napari-storm's own HDF5 format, which [opens](importing.md) like any other
file.

<div style="clear: both"></div>
