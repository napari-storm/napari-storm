"""The scientific core imports and works without Qt, napari or VisPy.

This is Level 2's exit gate from docs/modernization-review.md, expressed as a
test rather than as a convention. Adding a Qt or napari import anywhere under
``napari_storm.core`` fails here, in a subprocess where those modules are made
unimportable — checking ``sys.modules`` in-process would not work, since pytest
has already imported all of them for the widget tests.
"""
import os
import subprocess
import sys
import textwrap

FORBIDDEN_ROOTS = (
    "qtpy",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "napari",
    "vispy",
)

_BLOCKER = """
import sys

FORBIDDEN = {forbidden!r}


class _Blocker:
    \"\"\"Refuse to import anything a headless core must not need.\"\"\"

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        # napari_storm itself is obviously allowed; only the hosts are not.
        if root in FORBIDDEN:
            raise ImportError(
                f"napari_storm.core must not import {{fullname}}"
            )
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run_headless(body):
    """Run *body* in a subprocess where Qt, napari and VisPy cannot be imported."""
    script = _BLOCKER.format(forbidden=FORBIDDEN_ROOTS) + textwrap.dedent(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )


def test_core_imports_without_a_host():
    result = _run_headless(
        """
        import napari_storm.core as core

        assert core.LocalizationTable is not None
        assert core.DatasetStore is not None

        # The renderer contract is part of the host-free core: a backend
        # implements it, but describing one must not require a host.
        import numpy as np

        planned = core.NullRenderer()
        planned.open(
            1,
            core.RenderRequest(
                coords=np.zeros((2, 3), dtype=np.float32),
                sigmas=np.ones((2, 3), dtype=np.float32),
                size=1.0,
                values=np.ones(2, dtype=np.float32),
                name="headless",
            ),
        )
        assert planned.is_open(1)

        # And the planner -- where the Gaussian sizing and intensity weighting
        # live -- decides what to draw with no host present either.
        table = core.LocalizationTable(
            np.rec.array(
                np.zeros(4, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
            )
        )
        request = core.RenderPlanner().plan(
            table,
            core.GaussianSettings(),
            core.DatasetTraits(),
            name="headless",
            transform=core.WorldTransform(translation_nm=(10.0, 0.0, 0.0)),
        )
        assert request.coords.shape == (4, 3)
        # Coordinates are (z, y, x), so the translated x is the last column.
        assert request.coords[0, 2] == 10.0
        assert core.Bounds.of(request.coords[:, 2]) == core.Bounds(10.0, 10.0)

        # The store is only useful if its events reach a listener, so exercise
        # that too rather than merely importing the names.
        store = core.DatasetStore()
        seen = []
        store.subscribe(seen.append)

        class _Dataset:
            pass

        dataset = _Dataset()
        store.add(dataset)
        store.remove(dataset)
        assert seen == [
            core.DatasetOpened(dataset.dataset_id),
            core.DatasetClosed(dataset.dataset_id),
        ]
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_core_filters_and_converts_without_a_host():
    """Not just importable — the actual science runs with no host present."""
    result = _run_headless(
        """
        import numpy as np
        from napari_storm.core import LocalizationTable

        records = np.rec.array(
            np.zeros(10, dtype=[("x_pos_pixels", "f4"), ("y_pos_pixels", "f4")])
        )
        records.x_pos_pixels = np.arange(10, dtype="f4")
        records.y_pos_pixels = np.arange(10, dtype="f4") * 2

        table = LocalizationTable(
            records,
            position_columns={"x": "x_pos_pixels", "y": "y_pos_pixels"},
            position_scale_nm=100.0,
        )

        assert table.coordinate_nm("x").tolist() == [i * 100.0 for i in range(10)]
        table.bandpass("x_pos_nm", 200, 400)
        assert table.active_ids.tolist() == [2, 3, 4]
        assert table.limit_active_to(2) == 1
        assert table.n_active == 2
        table.reset()
        assert table.n_active == 10
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_dataset_classes_import_without_a_host():
    """P1-04: a reader may not construct a widget, so it may not need one.

    This is the boundary moving outward from ``core`` to the whole data layer.
    The application still opens dialogs — it just does so on the reader's behalf,
    through a MetadataProvider it injects.
    """
    result = _run_headless(
        """
        from napari_storm.localization_dataset_types import (
            LocalizationDataBaseClass, MinfluxDataBaseClass, StormDataClass,
        )

        assert StormDataClass is not None
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_a_reader_can_be_answered_without_a_dialog():
    """The questions a format cannot answer are supplied, not prompted for."""
    result = _run_headless(
        """
        import numpy as np
        from napari_storm.core import (PIXEL_SIZE_NM, ZDIM_PRESENT,
                                       MetadataProvider,
                                       StaticMetadataProvider)
        from napari_storm.localization_dataset_types import StormDataClass
        from napari_storm.localization_dataset_types.data_formats import (
            storm_data_dtype,
        )

        raw = np.zeros(4, dtype=storm_data_dtype)
        metadata = {
            "dataset_class_dtype": storm_data_dtype,
            "name": "headless",
            "sigma_present": False,
            "photon_count_present": False,
        }
        provider = StaticMetadataProvider(
            {ZDIM_PRESENT: False, PIXEL_SIZE_NM: "100"}
        )
        dataset = StormDataClass().import_recognized_data(
            raw, dict(metadata), metadata_provider=provider
        )
        assert dataset.pixelsize_nm == 100
        assert dataset.zdim_present is False

        # And with nothing to answer it, the reader fails rather than guessing.
        try:
            StormDataClass().import_recognized_data(
                raw, dict(metadata), metadata_provider=MetadataProvider()
            )
        except KeyError:
            pass
        else:
            raise AssertionError("an unanswerable import should not succeed")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_blocker_itself_works():
    """A guard that never fires proves nothing; check it can fail."""
    result = _run_headless("import qtpy\n")
    assert result.returncode != 0
    assert "must not import qtpy" in result.stderr
