"""Load Scene reopens the files a scene names and shows what was saved.

A scene used to re-apply settings only to data the user had already opened by
hand -- and could not have done more, because no importer ever recorded where a
dataset came from: every saved scene said ``"source_path": null``. These tests
pin the whole loop: save in one viewer, load into an empty one, and get the
same data, the same edits to it, and the same view.
"""

import json
import os
import shutil

import h5py
import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm._tests.test_hdf5_layout_dispatch import (
    _write_molecule_set,
    _write_picasso,
)
from napari_storm.core import (
    PIXEL_SIZE_NM,
    MetadataProvider,
    RecordingMetadataProvider,
    StaticMetadataProvider,
    load_scene,
)
from napari_storm.ns_constants import FWHM_TO_SIGMA


def _write_ns(path, n=50, x0=10_000.0, name="cells"):
    locs = np.rec.array(np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")]))
    locs.x_pos_nm = np.linspace(x0, x0 + 10_000.0, n)
    locs.y_pos_nm = np.linspace(40_000.0, 50_000.0, n)
    with h5py.File(path, "w") as file:
        stored = file.create_dataset("dataset", data=locs)
        stored.attrs["name"] = name
        stored.attrs["zdim_present"] = False
        stored.attrs["dataset_class"] = "LocalizationDataBaseClass"
    return str(path)


def _widget(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.reported = []
    widget._warn_user = widget.reported.append
    return widget


def _open(widget, path, **options):
    assert widget.open_localization_data_file_and_get_dataset(
        file_path=str(path), **options
    )
    return widget.localization_datasets[-1]


# ----------------------------------------------------------- the providers


def test_a_recording_provider_keeps_real_answers_only():
    recorder = RecordingMetadataProvider(StaticMetadataProvider({PIXEL_SIZE_NM: 130}))

    assert recorder.ask_text(PIXEL_SIZE_NM, "pixel size?") == "130"
    assert recorder.ask_yes_no("zdim_present", "3-D?") is None

    # A dismissed question is not an answer: replaying it would refuse.
    assert recorder.answers == {PIXEL_SIZE_NM: "130"}


def test_canned_answers_fall_back_to_asking():
    asked = []

    class Asker(MetadataProvider):
        def ask_text(self, key, question):
            asked.append(key)
            return "7"

    provider = StaticMetadataProvider({PIXEL_SIZE_NM: "130"}, fallback=Asker())

    assert provider.ask_text(PIXEL_SIZE_NM, "?") == "130"
    assert provider.ask_text("something_new", "?") == "7"
    assert asked == ["something_new"]


def test_a_dataset_is_named_after_its_file_whichever_separator_the_path_uses():
    """Windows paths named datasets after the whole path, separators and all."""
    from napari_storm.FileToLocalizationDataInterface import _file_name

    assert _file_name(r"C:\data\run 1\cells.h5") == "cells.h5"
    assert _file_name("/data/run 1/cells.h5") == "cells.h5"
    assert _file_name("C:/data/store.zarr/") == "store.zarr"


# ----------------------------------------------------------- saving sources


def test_the_scene_dialogs_start_beside_the_data_named_after_it(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    assert widget.default_scene_path() == ("", "session.napari_storm_scene_001.json")

    _open(widget, _write_ns(tmp_path / "cells.ns"))

    folder, name = widget.default_scene_path()
    assert os.path.samefile(folder, tmp_path)
    assert name == "cells.napari_storm_scene_001.json"
    assert widget.default_scene_path(latest=True)[1] is None


def test_each_new_scene_gets_the_next_number_and_load_offers_the_latest(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    _open(widget, _write_ns(tmp_path / "cells.ns"))
    for number in (1, 2, 7):  # a gap is not refilled
        widget.save_scene_to(tmp_path / f"cells.napari_storm_scene_{number:03d}.json")

    assert widget.default_scene_path()[1] == "cells.napari_storm_scene_008.json"
    assert widget.default_scene_path(latest=True)[1] == (
        "cells.napari_storm_scene_007.json"
    )


def test_a_saved_scene_says_which_file_each_dataset_came_from(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    data = _write_ns(tmp_path / "cells.ns")
    _open(widget, data)
    path = tmp_path / "view.napari-storm.json"

    widget.save_scene_to(path)

    entry = json.loads(path.read_text())["datasets"][0]
    assert os.path.samefile(entry["source_path"], data)
    assert entry["source_path_relative"] == "cells.ns"


# ----------------------------------------------------------- reopening


def test_loading_a_scene_into_an_empty_viewer_opens_its_file_and_view(
    make_napari_viewer, tmp_path
):
    """The point of the feature: come back to exactly what you were looking at."""
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    before.channel[0]._shift_spins["x"].setValue(2.0)
    camera = before.viewer.camera
    camera.center = (0.0, 45_000.0, 14_000.0)
    camera.zoom = 3.5
    camera.perspective = 30.0
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    assert after.load_scene_from(path) is not None

    assert [d.name for d in after.localization_datasets] == ["cells"]
    dataset = after.localization_datasets[0]
    transform = after.dataset_store.state(dataset.dataset_id).transform
    assert transform.translation_nm[0] == pytest.approx(2000.0)
    # The Shift box shows it too, or its next edit would undo it.
    assert after.channel[0]._shift_spins["x"].value() == pytest.approx(2.0)
    assert after.viewer.camera.zoom == pytest.approx(3.5)
    assert after.viewer.camera.perspective == pytest.approx(30.0)
    assert tuple(after.viewer.camera.center)[-2:] == pytest.approx((45_000.0, 14_000.0))


def test_a_molecule_sets_channels_come_back_as_they_were(make_napari_viewer, tmp_path):
    """Two datasets from one path, told apart by their place in the file."""
    before = _widget(make_napari_viewer)
    _open(before, _write_molecule_set(tmp_path / "cells.hdf5"))
    assert len(before.localization_datasets) == 2
    second = before.channel[1].Colormap_selector
    second.setCurrentIndex((second.currentIndex() + 2) % second.count())
    chosen = second.currentText()
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    assert [d.source_index for d in after.localization_datasets] == [0, 1]
    assert after.channel[1].Colormap_selector.currentText() == chosen


def test_a_scene_moved_together_with_its_data_still_opens(make_napari_viewer, tmp_path):
    before = _widget(make_napari_viewer)
    original = tmp_path / "experiment"
    original.mkdir()
    _open(before, _write_ns(original / "cells.ns"))
    before.save_scene_to(original / "view.json")

    moved = tmp_path / "archive" / "experiment"
    moved.parent.mkdir()
    shutil.move(str(original), str(moved))

    after = _widget(make_napari_viewer)
    after.load_scene_from(moved / "view.json")

    assert len(after.localization_datasets) == 1
    assert os.path.samefile(
        after.localization_datasets[0].file_path, moved / "cells.ns"
    )


def test_what_was_typed_in_at_import_is_not_asked_again(make_napari_viewer, tmp_path):
    """A Picasso file without a pixel size: asked once, never again."""
    data = _write_picasso(tmp_path / "locs.hdf5")
    before = _widget(make_napari_viewer)
    before.file_to_data_itf.metadata_provider = StaticMetadataProvider(
        {PIXEL_SIZE_NM: "130"}
    )
    _open(before, data)
    path = tmp_path / "view.json"
    before.save_scene_to(path)
    assert load_scene(path).datasets[0].import_answers == {PIXEL_SIZE_NM: "130"}

    after = _widget(make_napari_viewer)
    # Answers nothing: had the scene not kept the answer, the import would fail.
    after.file_to_data_itf.metadata_provider = MetadataProvider()
    after.load_scene_from(path)

    assert len(after.localization_datasets) == 1
    assert after.localization_datasets[0].pixelsize_nm == 130


def test_filters_and_adjustments_are_replayed_on_the_reopened_file(
    make_napari_viewer, tmp_path
):
    before = _widget(make_napari_viewer)
    dataset = _open(before, _write_ns(tmp_path / "cells.ns"))

    filters = before.data_filter_itf
    filters.current_parameter_idx = filters.list_of_filterable_parameters.index(
        "x_pos_nm"
    )
    filters.filter_slider_values_decimal = [0.0, 0.5]
    filters.apply_filtering(0)

    adjust = before.data_adjustment_itf
    adjust.current_parameter_idx = adjust.list_of_adjustable_parameters.index(
        "y_pos_nm"
    )
    adjust.math_mode_active_idx = 0
    adjust.value = 250.0
    adjust.apply_filtering(0)

    removed = filters.indices_for(dataset).copy()
    y_values = np.asarray(dataset.locs_all.y_pos_nm).copy()
    assert removed.size and y_values[0] == pytest.approx(40_250.0)
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    reopened = after.localization_datasets[0]
    np.testing.assert_array_equal(after.data_filter_itf.indices_for(reopened), removed)
    np.testing.assert_allclose(reopened.locs_all.y_pos_nm, y_values)


def test_reapplying_a_scene_to_its_own_data_does_not_adjust_twice(
    make_napari_viewer, tmp_path
):
    widget = _widget(make_napari_viewer)
    dataset = _open(widget, _write_ns(tmp_path / "cells.ns"))
    adjust = widget.data_adjustment_itf
    adjust.current_parameter_idx = adjust.list_of_adjustable_parameters.index(
        "y_pos_nm"
    )
    adjust.value = 250.0
    adjust.apply_filtering(0)
    path = tmp_path / "view.json"
    widget.save_scene_to(path)

    widget.load_scene_from(path)

    assert dataset.locs_all.y_pos_nm[0] == pytest.approx(40_250.0)


def test_render_settings_come_back(make_napari_viewer, tmp_path):
    """Widths, crop, grid and scalebar: the rest of 'what we saw'."""
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    before.Esigma_xy.setText("50")
    before.update_sigma()
    before.update_render_range("x", (20.0, 60.0))
    before.data_to_layer_itf.update_layers()
    before.Cgrid_plane.setChecked(True)
    before.Sgrid_plane_opacity.setValue(35)
    before.Esbsize.setText("500")
    before.Cscalebar.setChecked(True)
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    assert after.render_config.fixed_sigma_xy_nm == pytest.approx(50 / FWHM_TO_SIGMA)
    assert float(after.Esigma_xy.text()) == pytest.approx(50.0)
    assert tuple(after.render_config.range_x_percent) == pytest.approx((20.0, 60.0))
    # The X slider is drawn inverted.
    assert tuple(after.Srender_rangex.value()) == pytest.approx((40.0, 80.0))
    assert after.Cgrid_plane.isChecked()
    assert after.Sgrid_plane_opacity.value() == 35
    assert after.Cscalebar.isChecked()
    assert after.render_config.scalebar_size_nm == 500


def test_a_hidden_channel_stays_hidden(make_napari_viewer, tmp_path):
    """Through a layer refresh, and through a scene."""
    before = _widget(make_napari_viewer)
    dataset = _open(before, _write_ns(tmp_path / "cells.ns"))
    before.channel[0].show_channel()
    before.data_to_layer_itf.update_layers()
    state = before.dataset_store.state(dataset.dataset_id)
    assert state.appearance.opacity == 0.0, "a refresh showed a hidden channel"
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    reopened = after.localization_datasets[0]
    assert not after.channel[0].show_channel_state
    assert after.dataset_store.state(reopened.dataset_id).appearance.opacity == 0.0


def test_a_reference_image_is_reopened_where_it_was(make_napari_viewer, tmp_path):
    import tifffile

    from napari_storm.pyqt.image_import_dialog import ImageImportResult

    image_path = tmp_path / "widefield.tif"
    tifffile.imwrite(image_path, np.arange(32 * 32, dtype=np.uint16).reshape(32, 32))
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    controls = before._add_image_layer_from_result(
        ImageImportResult(
            file_path=str(image_path),
            img=tifffile.imread(image_path),
            layer_name="widefield.tif",
            orientation="XY",
            px_xy_nm=106.0,
            px_z_nm=1.0,
            x_off_nm=0.0,
            y_off_nm=0.0,
            z_off_nm=0.0,
        )
    )
    controls._layer.translate = (0.0, 41_000.0, 12_000.0)
    controls._layer.opacity = 0.4
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path)

    assert len(after.image_layer_controls) == 1
    layer = after.image_layer_controls[0]._layer
    assert layer.name == "widefield.tif"
    assert tuple(layer.translate) == pytest.approx((0.0, 41_000.0, 12_000.0))
    assert tuple(layer.scale)[1:] == pytest.approx((106.0, 106.0))
    assert layer.opacity == pytest.approx(0.4)


# ----------------------------------------------------------- choices and refusals


def test_open_files_false_only_reapplies(make_napari_viewer, tmp_path):
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    after.load_scene_from(path, open_files=False)

    assert after.localization_datasets == []
    assert any("not loaded" in message for message in after.reported)


def test_declining_the_confirmation_leaves_the_session_alone(
    make_napari_viewer, tmp_path
):
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    other = _widget(make_napari_viewer)
    _open(other, _write_ns(tmp_path / "other.ns", name="other"))
    asked = []

    result = other.load_scene_from(
        path, confirm=lambda plan: asked.append(plan) or False
    )

    assert result is None
    assert [len(plan.files) for plan in asked] == [1]
    assert [d.name for d in other.localization_datasets] == ["other"]


def test_loading_a_scene_replaces_unrelated_data(make_napari_viewer, tmp_path):
    before = _widget(make_napari_viewer)
    _open(before, _write_ns(tmp_path / "cells.ns"))
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    other = _widget(make_napari_viewer)
    _open(other, _write_ns(tmp_path / "other.ns", name="other"))
    other.load_scene_from(path)

    assert [d.name for d in other.localization_datasets] == ["cells"]


def test_a_scene_loads_in_the_background_too(make_napari_viewer, tmp_path, qtbot):
    """The Load Scene button's path: one worker per file, then the view."""
    before = _widget(make_napari_viewer)
    _open(before, _write_molecule_set(tmp_path / "cells.hdf5"))
    before.viewer.camera.zoom = 2.25
    path = tmp_path / "view.json"
    before.save_scene_to(path)

    after = _widget(make_napari_viewer)
    done = []
    after.load_scene_from(path, background=True, on_done=done.append)

    qtbot.waitUntil(lambda: bool(done), timeout=15_000)
    assert done == [[]]
    assert len(after.localization_datasets) == 2
    assert after.viewer.camera.zoom == pytest.approx(2.25)
