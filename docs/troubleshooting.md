# Troubleshooting

## Starting

**`napari-storm` opens no window, or napari complains about Qt.**
No Qt binding is installed. Install one with the package:
`pip install "napari-storm[pyqt6]"` (or `[pyside6]`).

**"Instanced rendering is unavailable … Continuing with the slower billboard
renderer."**
napari-storm draws fastest through VisPy's `gl+` backend, which needs
PyOpenGL. Without it everything still works, with about eleven times the
memory per localization. `pip install PyOpenGL` usually fixes it. See
[How it works](how-it-works.md#two-backends-one-contract).

## Opening files

**"Unknown data file extension"** -- the extension is not one napari-storm
reads. Try **Auto-detect Format**, or write a **Custom** import; see
[Formats it does not know](importing.md#formats-it-does-not-know).

**A TIFF is refused.** napari-storm draws localizations; it does not fit raw
movies. Localize the movie first, or open the TIFF as a
[reference image](importing.md#reference-images).

**"opening MINFLUX Zarr datasets needs zarr<3"** -- install the extra:
`pip install "napari-storm[minflux]"`. See [Zarr stores](minflux.md#zarr-stores).

**"MINFLUX Zarr datasets are structured arrays, which zarr 3 does not
support"** -- zarr 3 is installed. Install `zarr<3` in this environment, for
example through the `[minflux]` extra; if something else in the environment
needs zarr 3, give napari-storm an environment of its own.

**The file dialog cannot select a `.zarr` folder.** Select any file inside it.

**"A single import cannot mix 2D and 3D datasets"**, or a merge is refused --
merged datasets must all be 2-D or all 3-D.

## Display

**Everything is one saturated blob.** The default contrast saturates at a
single localization. Raise **×Range** in the dataset's card; see
[Contrast](rendering.md#contrast).

**Isolated specks everywhere.** Raise **Cutoff**, or widen the Gaussians.

**"Variable-size gaussian" jumps back to fixed.** It needs STORM/PALM data
with precisions; it is not offered while MINFLUX or plain localizations are
loaded.

**Rainbow depth colouring is missing.** It needs 3-D data and fixed-size
Gaussians.

**"Connect traces" is greyed out.** None of the loaded datasets records which
localizations form a trace -- only MINFLUX data does.

**Only part of my data is drawn, and a message says so.** The render budget
caps what is drawn at about 6.7 million localizations; see
[How much is drawn](rendering.md#how-much-is-drawn). Filters and exports
still use everything.

**Very dense regions look clipped on screen.** The canvas is 8-bit; the
exported OME-TIFF is not. See [Screen and export](rendering.md#screen-and-export).

## Environment variables

| Variable | Effect |
|---|---|
| `NAPARI_STORM_RENDER_BUDGET_MB` | The render budget in MB (default 2048); `0` removes it |

## Still stuck

[File an issue](https://github.com/napari-storm/napari-storm/issues) with what
you did, what happened, and the file format -- a small file that shows the
problem helps most.
