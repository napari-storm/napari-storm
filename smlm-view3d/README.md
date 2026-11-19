# smlm-view3d

napari-storm's drawing core without napari: the 3D view of SMLM localizations.

- `smlm_view3d.core`: the localization table, the render planner, the
  renderer contract and the scene. It needs numpy only: no Qt, no napari, no
  VisPy.
- `smlm_view3d.particles`: the instanced VisPy billboards and the plain-VisPy
  backend, `VispyInstancedRenderer`. It needs VisPy, PyOpenGL and numpy, and no
  napari. It reaches Qt only through VisPy's app.

napari-storm depends on this package and keeps its napari layers, its dock
widget and its readers. Its old import names (`napari_storm.core`,
`napari_storm.core.<module>` and `napari_storm.napari_particles.<module>` for
the modules moved here) are the same module objects as this package's.

## Install

From the repository root, before napari-storm:

```
pip install -e ./smlm-view3d
pip install -e ".[dev,pyqt6]"
```

Extras: `density` (scipy, for `core.density.local_density`) and `ome-tiff`
(tifffile, for `core.ome_export`'s writer).

## Licence

BSD 3-Clause; see `LICENSE`. The modules in `smlm_view3d/particles` are
derived from napari-particles by Martin Weigert; see the `NOTICE` there.
