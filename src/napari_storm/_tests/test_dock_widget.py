"""
Smoke tests for the napari_storm dock widget.

These require a real napari viewer and Qt event loop.
"""

import numpy as np
from qtpy.QtWidgets import QApplication, QScrollArea

from napari_storm._dock_widget import napari_storm
from napari_storm.FileToLocalizationDataInterface import QFileDialog
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.render_config import RenderConfig


def test_legacy_top_level_widget_export_is_lazy_but_compatible():
    import napari_storm as package

    assert package.napari_storm is napari_storm


def _make_dataset_2d(n=20, name="smoke_2d"):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def test_widget_instantiation(make_napari_viewer):
    """napari_storm should construct without raising."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    assert widget is not None


def test_reader_registry_is_scoped_to_the_viewer(make_napari_viewer):
    first_viewer = make_napari_viewer()
    second_viewer = make_napari_viewer()
    first_widget = napari_storm(napari_viewer=first_viewer)
    second_widget = napari_storm(napari_viewer=second_viewer)

    assert napari_storm.get_instance(first_viewer) is first_widget
    assert napari_storm.get_instance(second_viewer) is second_widget


def test_render_config_present(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    assert isinstance(widget.render_config, RenderConfig)


def test_get_dataset_from_test_mode(make_napari_viewer):
    """get_dataset_from_test_mode() should load a dataset without raising."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    ds = _make_dataset_2d()
    widget.get_dataset_from_test_mode([ds])
    assert widget.n_datasets == 1
    assert len(widget.localization_datasets) == 1


def test_clear_datasets(make_napari_viewer):
    """clear_datasets() should reset to empty state."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    ds = _make_dataset_2d()
    widget.get_dataset_from_test_mode([ds])
    widget.clear_datasets()
    assert widget.n_datasets == 0
    assert widget.localization_datasets == []


def test_multiple_datasets(make_napari_viewer):
    """Loading two datasets should give n_datasets == 2."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    ds1 = _make_dataset_2d(20, name="ch1")
    ds2 = _make_dataset_2d(15, name="ch2")
    widget.get_dataset_from_test_mode([ds1, ds2])
    assert widget.n_datasets == 2


def test_cancelled_open_preserves_current_session(make_napari_viewer, monkeypatch):
    """Closing the file picker must not clear or hide the current dock."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    dataset = _make_dataset_2d(name="keep-me")
    widget.get_dataset_from_test_mode([dataset])

    layer = widget.data_to_layer_itf.layer_for(dataset)
    controls = widget.channel[0]
    hidden = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda: ("", ""))
    monkeypatch.setattr(widget, "hide", lambda: hidden.append(True))

    assert widget.open_localization_data_file_and_get_dataset() is False
    assert widget.localization_datasets == [dataset]
    assert widget.channel == [controls]
    assert widget.data_to_layer_itf.layer_for(dataset) is layer
    assert layer in viewer.layers
    assert widget.data_filter_itf.dfw.Cdatasets.count() == 1
    assert widget.file_to_data_itf.dataset_names == ["keep-me"]
    assert hidden == []


def test_data_controls_tab_is_scrollable(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    assert isinstance(widget.data_controls_scroll, QScrollArea)
    assert widget.data_controls_scroll.widget() is widget.data_controls_content

    # The base controls already exceed a compact dock; dataset controls only
    # increase this size.  Verify Qt can expose that overflow vertically.
    widget.setFixedSize(320, 140)
    widget.show()
    QApplication.processEvents()
    assert widget.data_controls_scroll.verticalScrollBar().maximum() > 0


def test_unload_dataset_keeps_remaining_state_aligned(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    datasets = [
        _make_dataset_2d(name="first"),
        _make_dataset_2d(name="middle"),
        _make_dataset_2d(name="last"),
    ]
    widget.get_dataset_from_test_mode(datasets)

    first_layer = widget.data_to_layer_itf.layer_for(datasets[0])
    last_layer = widget.data_to_layer_itf.layer_for(datasets[2])
    first_controls = widget.channel[0]
    last_controls = widget.channel[2]

    # Exercise the actual control presented to users.
    widget.channel[1].Bunload.click()

    assert widget.localization_datasets == [datasets[0], datasets[2]]
    assert widget.channel == [first_controls, last_controls]
    assert first_controls.dataset is datasets[0]
    assert last_controls.dataset is datasets[2]
    assert widget.data_to_layer_itf.layer_for(datasets[0]) is first_layer
    assert widget.data_to_layer_itf.layer_for(datasets[2]) is last_layer
    assert first_layer in viewer.layers
    assert last_layer in viewer.layers
    assert widget.data_to_layer_itf.layer_for(datasets[1]) is None
    assert widget.n_datasets == 2
    assert widget.data_to_layer_itf.n_layers == 2
    assert len(widget.data_to_layer_itf.render_state) == 2
    # Keyed by identity, so the survivors keep their own entries.
    assert sorted(widget.data_to_layer_itf.render_state) == [
        datasets[0].dataset_id,
        datasets[2].dataset_id,
    ]
    assert widget.data_filter_itf.dfw.Cdatasets.count() == 2
    assert widget.data_adjustment_itf.daw.Cdatasets.count() == 2
    assert len(widget.Lnumberoflocs._cards) == 2
    assert widget.file_to_data_itf.dataset_names == ["first", "last"]


def test_unloading_last_dataset_returns_to_clean_empty_state(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_make_dataset_2d(name="only")])

    assert widget.unload_dataset(0) is True
    assert widget.localization_datasets == []
    assert widget.channel == []
    assert widget.n_datasets == 0
    assert widget.render_config.zdim is None
    assert widget.data_to_layer_itf.n_layers == 0
    assert widget.data_to_layer_itf.render_state == {}
    assert widget.data_filter_itf.dfw.Cdatasets.count() == 0
    assert widget.data_adjustment_itf.daw.Cdatasets.count() == 0
    assert widget.file_to_data_itf.dataset_names == []
    assert len(viewer.layers) == 0


def test_the_renderer_releases_state_from_the_store_event(make_napari_viewer):
    """Unloading must not depend on the widget remembering every dependant.

    The renderer subscribes to the store, so removing a dataset from the store
    alone is enough to release its render arrays.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(
        [_make_dataset_2d(name="a"), _make_dataset_2d(name="b")]
    )
    kept, dropped = widget.localization_datasets

    widget.dataset_store.remove(dropped)

    assert sorted(widget.data_to_layer_itf.render_state) == [kept.dataset_id]


def test_render_state_follows_identity_not_position(make_napari_viewer):
    """The bug the id keying exists to prevent: state shifting onto a neighbour."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(
        [
            _make_dataset_2d(name="a"),
            _make_dataset_2d(name="b"),
            _make_dataset_2d(name="c"),
        ]
    )
    datasets = list(widget.localization_datasets)
    third_state = widget.data_to_layer_itf.render_state[datasets[2].dataset_id]

    widget.unload_dataset(0)

    # The third dataset moved from index 2 to index 1 and kept its own arrays.
    assert widget.data_to_layer_itf.render_state[datasets[2].dataset_id] is third_state
    assert datasets[0].dataset_id not in widget.data_to_layer_itf.render_state
