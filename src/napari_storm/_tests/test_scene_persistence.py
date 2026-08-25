"""The scene format: what it records, what it refuses, and what it never stores.

Level 4 lists the persistence format as a deliverable and notes it "does not
exist today". §7.4 additionally requires that reference-image transforms
round-trip through it.

The format records *decisions*, never pixels and never localizations. That is
not only a size argument: napari serialises a layer's mesh, and the instanced
backend's mesh is a four-vertex quad standing in for every localization, so
saving through the host's own layer state would write a quad and call it a
reconstruction.
"""
import json

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core import (CameraState, DatasetEntry, GaussianSettings,
                               LayerAppearance, ReferenceImageEntry, Scene,
                               SceneFormatError, WorldTransform, load_scene,
                               save_scene)
from napari_storm.core.scene import SCENE_FORMAT, SCENE_VERSION
from napari_storm.localization_dataset_types import LocalizationDataBaseClass


def _dataset(name="ds", n=200):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(1_000, 3_000, n)
    locs["y_pos_nm"] = np.linspace(1_000, 5_000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _widget(make_napari_viewer, names=("a",)):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset(name) for name in names])
    return widget, viewer


# --------------------------------------------------------------- the format


def test_a_scene_round_trips_through_a_file(tmp_path):
    scene = Scene(
        datasets=(
            DatasetEntry(
                name="channel-1",
                source_path="/data/one.hdf5",
                transform=WorldTransform(
                    scale=(1.0, 2.0, 1.0), translation_nm=(500.0, -250.0, 10.0)
                ),
                appearance=LayerAppearance(
                    colormap="green", opacity=0.4, contrast_limits=(0.1, 0.9)
                ),
            ),
        ),
        gaussian=GaussianSettings(mode=1, fixed_sigma_xy_nm=12.5),
        camera=CameraState(center_nm=(1.0, 2.0, 3.0), zoom=4.0, ndisplay=2),
    )
    path = tmp_path / "session.json"

    save_scene(path, scene)
    restored = load_scene(path)

    assert restored.datasets[0].name == "channel-1"
    assert restored.datasets[0].source_path == "/data/one.hdf5"
    assert restored.datasets[0].transform.translation_nm == (500.0, -250.0, 10.0)
    assert restored.datasets[0].transform.scale == (1.0, 2.0, 1.0)
    assert restored.datasets[0].appearance.colormap == "green"
    assert restored.datasets[0].appearance.contrast_limits == (0.1, 0.9)
    assert restored.gaussian.mode == 1
    assert restored.gaussian.fixed_sigma_xy_nm == 12.5
    assert restored.camera.zoom == 4.0
    assert restored.camera.ndisplay == 2


def test_a_reference_image_placement_round_trips(tmp_path):
    """§7.4 asks for this one by name."""
    scene = Scene(
        reference_images=(
            ReferenceImageEntry(
                name="widefield",
                source_path="/data/ref.tif",
                orientation="XY",
                pixel_size_xy_nm=106.0,
                pixel_size_z_nm=250.0,
                offset_nm=(1.0, -3000.0, 4000.0),
            ),
        )
    )
    path = tmp_path / "session.json"

    save_scene(path, scene)
    image = load_scene(path).reference_images[0]

    assert image.pixel_size_xy_nm == 106.0
    assert image.pixel_size_z_nm == 250.0
    assert image.offset_nm == (1.0, -3000.0, 4000.0)
    assert image.orientation == "XY"


def test_a_scene_is_readable_and_diffable(tmp_path):
    """Small, indented, and units declared -- a file a human can check."""
    path = tmp_path / "session.json"
    save_scene(path, Scene(datasets=(DatasetEntry(name="a"),)))

    text = path.read_text()
    raw = json.loads(text)

    assert raw["format"] == SCENE_FORMAT
    assert raw["version"] == SCENE_VERSION
    assert raw["length_unit"] == "nm"
    assert "\n  " in text, "written on one line; not diffable"


def test_no_localizations_are_stored(tmp_path):
    """Decisions, not data. A scene must not become a stale copy."""
    scene = Scene(datasets=(DatasetEntry(name="a", source_path="/data/a.hdf5"),))
    path = tmp_path / "session.json"

    save_scene(path, scene)

    raw = json.loads(path.read_text())
    text = json.dumps(raw)
    assert "x_pos_nm" not in text
    assert len(text) < 4000, "a scene this small should not be kilobytes of data"


# --------------------------------------------------------------- refusals


def test_a_file_that_is_not_a_scene_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else", "version": 1}')

    with pytest.raises(SceneFormatError, match="not a napari-storm scene"):
        load_scene(path)


def test_invalid_json_is_refused_with_its_reason(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    with pytest.raises(SceneFormatError, match="not valid JSON"):
        load_scene(path)


def test_a_newer_scene_is_refused_rather_than_guessed_at(tmp_path):
    """Silently ignoring fields whose meaning changed would misplace data."""
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps({"format": SCENE_FORMAT, "version": SCENE_VERSION + 1})
    )

    with pytest.raises(SceneFormatError, match="reads up to"):
        load_scene(path)


def test_unknown_settings_within_a_version_are_dropped_not_fatal(tmp_path):
    """Forward compatibility: a later build may add settings this one lacks."""
    path = tmp_path / "extra.json"
    path.write_text(
        json.dumps(
            {
                "format": SCENE_FORMAT,
                "version": SCENE_VERSION,
                "gaussian": {"mode": 1, "a_setting_from_the_future": 42},
            }
        )
    )

    scene = load_scene(path)

    assert scene.gaussian.mode == 1


# ------------------------------------------------------- through the widget


def test_the_widget_saves_what_the_user_set(make_napari_viewer, tmp_path):
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    widget.channel[0]._shift_spins["x"].setValue(3.0)
    widget.data_to_layer_itf.set_appearance(dataset, colormap="green", opacity=0.25)
    path = tmp_path / "session.json"

    widget.save_scene_to(path)
    scene = load_scene(path)

    entry = scene.datasets[0]
    assert entry.name == dataset.name
    assert entry.transform.translation_nm[0] == pytest.approx(3000.0)
    assert entry.appearance.opacity == pytest.approx(0.25)


def test_loading_a_scene_restores_the_alignment(make_napari_viewer, tmp_path):
    """The point of the whole feature: come back to the scene you left."""
    widget, _ = _widget(make_napari_viewer)
    dataset = widget.localization_datasets[0]
    widget.channel[0]._shift_spins["x"].setValue(4.0)
    path = tmp_path / "session.json"
    widget.save_scene_to(path)

    widget.channel[0].reset_shift()
    assert widget.dataset_store.state(dataset.dataset_id).transform.translation_nm[0] == 0.0

    widget.load_scene_from(path)

    restored = widget.dataset_store.state(dataset.dataset_id).transform
    assert restored.translation_nm[0] == pytest.approx(4000.0)


def test_a_scene_naming_absent_datasets_reports_rather_than_guesses(
    make_napari_viewer, tmp_path
):
    """Matching by name can fail, and a wrong guess would misplace real data."""
    widget, _ = _widget(make_napari_viewer, names=("a",))
    reported = []
    widget._warn_user = reported.append
    path = tmp_path / "session.json"
    save_scene(path, Scene(datasets=(DatasetEntry(name="not-loaded"),)))

    unmatched = widget.apply_scene(load_scene(path))

    assert unmatched == ["not-loaded"]
    assert reported and "not loaded" in reported[0]


def test_loading_a_scene_does_not_open_files(make_napari_viewer, tmp_path):
    """A scene that silently loaded gigabytes on open would be a surprise."""
    widget, _ = _widget(make_napari_viewer, names=("a",))
    before = len(widget.localization_datasets)
    path = tmp_path / "session.json"
    save_scene(
        path,
        Scene(datasets=(DatasetEntry(name="b", source_path="/nonexistent.hdf5"),)),
    )
    widget._warn_user = lambda message: None

    widget.load_scene_from(path)

    assert len(widget.localization_datasets) == before


def test_an_unreadable_scene_is_reported_not_raised(make_napari_viewer, tmp_path):
    widget, _ = _widget(make_napari_viewer)
    reported = []
    widget._warn_user = reported.append

    result = widget.load_scene_from(tmp_path / "does-not-exist.json")

    assert result is None
    assert reported and "could not load scene" in reported[0]


def test_the_scene_buttons_do_not_overlap_other_controls(make_napari_viewer):
    """A QGridLayout silently stacks two widgets in one cell.

    The first version of these buttons landed on the row already holding a
    separator, which no functional test would have noticed.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    layout = widget.data_controls_tab_layout

    occupied = {}
    for index in range(layout.count()):
        item = layout.itemAt(index)
        row, column, row_span, column_span = layout.getItemPosition(index)
        for r in range(row, row + row_span):
            for c in range(column, column + column_span):
                clash = occupied.get((r, c))
                assert clash is None, (
                    f"grid cell ({r}, {c}) holds two widgets: "
                    f"{clash} and {item.widget()}"
                )
                occupied[(r, c)] = item.widget()


def test_saving_is_offered_only_once_something_is_loaded(make_napari_viewer):
    """Saving an empty scene is not useful; loading one into an empty session is."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    assert not widget.Bsave_scene.isVisibleTo(widget)
    assert widget.Bload_scene.isVisibleTo(widget)

    widget.get_dataset_from_test_mode([_dataset()])

    assert widget.Bsave_scene.isVisibleTo(widget)
