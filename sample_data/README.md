# Sample data

Files to open while checking that an importer still works, and to hand to
someone reproducing a report.

## Real measurement

| File | Format |
| --- | --- |
| `beta_ii_spectrin.storm_4pi.h5` | daxview molecule set: a `molecule_set_data` group, 350,949 localizations, 4Pi STORM of beta-II spectrin |

## Synthetic MINFLUX

**These are not instrument output.** They are written to the layout pyMINFLUX
documents, by the fixtures in
`src/napari_storm/_tests/test_minflux_v2.py` (`_v2_array`, `_v2_json_records`,
`_v1_json_records`), and they carry 64 localizations of ramp data rather than
anything measured. They exercise container handling, layout detection and
routing. They cannot tell you whether a real Abberior export matches that
layout in some detail nobody wrote down — only a real export can.

| File | What it is for |
| --- | --- |
| `imspector_v2.npy` | Imspector >= 24.10 layout, NumPy format 1.0 |
| `imspector_v2_format2.npy` | The same array in NumPy format 2.0, whose header is sized with four bytes rather than two — the case that used to be read as corrupt |
| `imspector_v2.json` | The same data as JSON records, flat, `itr` a scalar |
| `imspector_legacy_v1.json` | The **older** layout: one record per trace, iterations nested under `itr`. Routes to the v1 reader, and is the file that says whether legacy support still works |
| `imspector_v2.mat` | The Matlab container |
| `pyminflux_v2.pmx` | pyMINFLUX's own save format |

No Zarr store is checked in. Building one needs `zarr<3` — which is what
`setup.cfg` pins, because zarr 3 refuses the structured arrays Imspector
writes — so a store cannot be created in an environment that has zarr 3
installed. `test_minflux_v2.py` builds one in a temporary directory when the
right zarr is present, and skips otherwise.
