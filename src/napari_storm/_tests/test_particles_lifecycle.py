"""Regression tests for Particles resource ownership and render dtypes.

Covers register items P0-02 (undisconnected viewer callbacks), P1-07 (float64
positions and int64 indices) and P1-03 (private napari access must degrade,
not raise) from docs/modernization-review.md.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.napari_particles._napari_compat import _FORCED_BLENDING
from napari_storm.napari_particles.particles import Particles
from napari_storm.napari_particles.renderer import NapariParticlesRenderer
from napari_storm.napari_particles.utils import generate_billboards_2d


def _dataset_2d(n=32, name="lifecycle"):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=False)


def _dataset_3d(n=32, name="lifecycle-3d"):
    locs = np.zeros(
        n,
        dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")],
    )
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    locs["z_pos_nm"] = np.linspace(-500, 500, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=True)


def _billboard_widget(viewer):
    """A widget pinned to the billboard backend.

    The plugin now defaults to the instanced backend, which stores one row per
    localization and has no `_size`, `filter` or `shader` to inspect. Tests that
    reach into billboard internals have to name the backend they are testing
    rather than inherit whichever one happens to be default -- otherwise the
    next backend change turns them into failures that say nothing.
    """
    return napari_storm(napari_viewer=viewer, renderer=NapariParticlesRenderer(viewer))


def _forced_visual_count():
    """How many visuals currently have their blend function forced.

    This replaced a count of `_apply_blend_state` callbacks on the layer-list
    events. Those callbacks are gone -- forcing the blend function through the
    visual's own `set_gl_state` covers triggers the event list never did -- but
    the leak they were counted for is unchanged: repeated updates must not
    accumulate per-layer bookkeeping.
    """
    return len(_FORCED_BLENDING)


# --------------------------------------------------------------------- dtypes


def test_billboard_faces_are_uint32():
    """Index buffers must not silently widen to int64 (P1-07)."""
    coords = np.zeros((16, 3), dtype=np.float32)
    _verts, faces, _tex = generate_billboards_2d(coords, size=10.0)
    assert faces.dtype == np.uint32
    # winding and range must survive the dtype change
    assert faces.max() == 6 * len(coords) - 1
    assert faces.shape == (2 * len(coords), 3)


def test_particles_render_arrays_are_float32():
    """No float64 array may reach the renderer (P1-07)."""
    coords = np.zeros((16, 3), dtype=np.float64)  # deliberately float64 input
    layer = Particles(coords, size=10.0, sigmas=(1, 1, 1), values=1.0)
    for name in ("_coords", "_centercoords", "_sigmas", "_size", "_texcoords"):
        arr = getattr(layer, name)
        assert arr.dtype == np.float32, f"{name} is {arr.dtype}, expected float32"
    vertices, faces, values = layer.data
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32
    assert values.dtype == np.float32
    assert layer.shader == "gaussian"


def test_particles_host_arrays_use_352_bytes_per_localization():
    """Keep the measured Level 1 dtype improvement honest."""
    n = 16
    layer = Particles(np.zeros((n, 3), dtype=np.float64), size=10.0)
    arrays = [
        layer._coords,
        layer._centercoords,
        layer._sigmas,
        layer._size,
        layer._texcoords,
        layer._view_faces,
        *layer.data,
    ]
    assert sum(array.nbytes for array in arrays) / n == 352


def test_empty_particles_layer_fails_before_geometry_or_shader_work():
    """An empty import must report its cause rather than divide by zero."""
    with pytest.raises(ValueError, match="at least one localization"):
        Particles(np.empty((0, 3), dtype=np.float32), size=10.0)


def test_coords_from_locs_is_float32(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    ds = _dataset_2d()
    coords = widget.data_to_layer_itf.get_coords_from_locs(ds)
    assert coords.dtype == np.float32


# ------------------------------------------------------------------ lifecycle


def test_close_is_safe_before_add_to_viewer():
    """close() on a layer that was never added must not raise."""
    layer = Particles(np.zeros((8, 3), dtype=np.float32), size=10.0)
    layer.close()
    layer.close()  # idempotent


def test_close_restores_the_blend_setter(make_napari_viewer):
    """P0-02, against the mechanism that replaced the callbacks.

    This used to count the three layer-list callbacks `add_to_viewer` connected,
    because that was how additive blending was re-asserted. Chasing events that
    way missed triggers -- any *other* layer changing visibility resets our blend
    too -- so the state is now forced by wrapping the visual's `set_gl_state`
    instead. The leak this guards against is the same one: what `add_to_viewer`
    installs, `close` has to take back off.
    """
    viewer = make_napari_viewer()
    layer = Particles(np.zeros((8, 3), dtype=np.float32), size=10.0, name="p")
    layer.add_to_viewer(viewer)

    visual = layer._visual
    assert visual in _FORCED_BLENDING
    wrapped = visual.set_gl_state

    layer.close()

    assert visual not in _FORCED_BLENDING
    assert visual.set_gl_state is not wrapped
    assert layer._viewer is None
    assert layer._visual is None


def test_a_released_visual_takes_napari_blending_again(make_napari_viewer):
    """Releasing has to actually hand the GL state back, not just stop tracking."""
    viewer = make_napari_viewer()
    layer = Particles(np.zeros((8, 3), dtype=np.float32), size=10.0, name="p")
    layer.add_to_viewer(viewer)
    visual = layer._visual
    layer.close()

    visual.set_gl_state(blend=True, blend_func=("src_alpha", "zero"))

    assert visual._vshare.gl_state["blend_func"] == ("src_alpha", "zero")


def test_repeated_updates_do_not_accumulate_per_layer_state(make_napari_viewer):
    """The leak that made repeated setting changes progressively slower.

    Each update_layers() rebuild used to add three more layer-list callbacks
    without removing the previous set. Those callbacks no longer exist, so what
    is counted now is the per-visual blend bookkeeping that replaced them --
    same invariant, current mechanism.
    """
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset_2d()])

    after_first = _forced_visual_count()
    after_first_layers = len(viewer.layers)
    for _ in range(100):
        widget.data_to_layer_itf.update_layers()

    assert _forced_visual_count() == after_first
    assert len(viewer.layers) == after_first_layers


def test_sigma_update_preserves_layer_and_visual_identity(make_napari_viewer):
    """P0-01: an appearance-only change must update the existing layer."""
    viewer = make_napari_viewer()
    widget = _billboard_widget(viewer)
    dataset = _dataset_2d()
    widget.get_dataset_from_test_mode([dataset])

    layer = widget.data_to_layer_itf.layer_for(dataset)
    visual = layer._visual
    callbacks = _forced_visual_count()
    old_size = layer._size.copy()
    widget.Esigma_xy.setText("40")

    widget.update_sigma()

    assert widget.data_to_layer_itf.layer_for(dataset) is layer
    assert layer._visual is visual
    assert _forced_visual_count() == callbacks
    assert np.all(layer._size > old_size)


def test_rainbow_round_trip_preserves_billboard_vertex_attributes(
    make_napari_viewer,
):
    """A Z-color toggle must not split the two triangles of a Gaussian quad."""
    viewer = make_napari_viewer()
    widget = _billboard_widget(viewer)
    dataset = _dataset_3d()
    widget.get_dataset_from_test_mode([dataset])

    layer = widget.data_to_layer_itf.layer_for(dataset)
    visual = layer._visual
    gaussian_filter = layer.filter[0]
    original_colormap = layer.colormap.name

    widget.Bz_color_coding.setChecked(True)

    assert layer.colormap.name == "hsv"
    np.testing.assert_array_equal(layer._billboard_filter.texcoords, layer._texcoords)
    np.testing.assert_array_equal(
        layer._billboard_filter.centercoords, layer._centercoords[:, -3:]
    )
    np.testing.assert_array_equal(layer._billboard_filter.sigmas, layer._sigmas[:, -3:])
    rainbow_values = layer.data[2].reshape(-1, 6)
    np.testing.assert_array_equal(
        rainbow_values, np.repeat(rainbow_values[:, :1], 6, axis=1)
    )

    widget.Bz_color_coding.setChecked(False)

    assert widget.data_to_layer_itf.layer_for(dataset) is layer
    assert layer._visual is visual
    assert layer.filter[0] is gaussian_filter
    assert layer.colormap.name == original_colormap
    np.testing.assert_array_equal(layer._billboard_filter.texcoords, layer._texcoords)
    np.testing.assert_array_equal(layer.data[2], np.ones_like(layer.data[2]))


def test_empty_earlier_channel_does_not_shift_later_channel_index(make_napari_viewer):
    """A filtered-out channel must not make following channels use its settings."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    near = _dataset_2d(name="near")
    far = _dataset_2d(name="far")
    far.adjust_column("x_pos_nm", offset=10_000)
    far.reset_filters()
    widget.get_dataset_from_test_mode([near, far])

    updated_channels = []
    widget.data_to_layer_itf.on_layer_updated = updated_channels.append
    widget.render_config.range_x_percent = np.array([75, 100])
    widget.data_to_layer_itf.update_layers()

    assert near.x_pos_nm.size == 0
    assert far.x_pos_nm.size > 0
    assert updated_channels == [1]
    assert widget.data_to_layer_itf.layer_for(far).visible


def test_clear_datasets_releases_callbacks(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    before = _forced_visual_count()

    widget.get_dataset_from_test_mode([_dataset_2d()])
    widget.clear_datasets()

    assert _forced_visual_count() == before


def test_repeated_load_unload_cycles_release_all_dataset_state(
    make_napari_viewer,
):
    """Reopen cycles must not retain callbacks or parallel render entries."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    before = _forced_visual_count()

    for cycle in range(5):
        widget.get_dataset_from_test_mode([_dataset_2d(name=f"lifecycle-{cycle}")])
        assert widget.data_to_layer_itf.n_layers == 1
        assert len(widget.data_to_layer_itf.render_state) == 1

        widget.unload_dataset(0)
        assert _forced_visual_count() == before
        assert widget.data_to_layer_itf.n_layers == 0
        assert widget.data_to_layer_itf.render_state == {}
        assert len(viewer.layers) == 0


# --------------------------------------------------------------- napari compat


def test_missing_shading_combo_does_not_break_layer_creation(make_napari_viewer):
    """P1-03: napari 0.7 dropped shadingComboBox; that must not be fatal."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset_2d()])
    assert widget.n_datasets == 1
    assert "lifecycle" in viewer.layers


def test_gaussian_shader_does_not_hijack_napari_shading(make_napari_viewer):
    """napari forwards layer.shading straight to VisPy, which rejects 'gaussian'.

    Overriding the property made napari raise AssertionError as soon as it
    re-sliced the layer, which it does to every layer whenever the scene extent
    changes -- i.e. on loading a second dataset covering a different area.
    """
    viewer = make_napari_viewer()
    widget = _billboard_widget(viewer)
    widget.get_dataset_from_test_mode([_dataset_2d()])

    layer = widget.data_to_layer_itf.layer_for(widget.localization_datasets[0])
    assert layer.shader == "gaussian"
    assert (
        layer.shading == "none"
    ), "napari's shading property must stay a value VisPy accepts"


def test_second_dataset_with_a_different_extent_loads(make_napari_viewer):
    """Regression for the AssertionError above.

    Two fields of view covering different areas force napari to re-slice, which
    is what surfaced the shading hijack.
    """
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    far = _dataset_2d(name="far")
    far.adjust_column("x_pos_nm", offset=10_000)
    far.adjust_column("y_pos_nm", offset=10_000)

    widget.get_dataset_from_test_mode([_dataset_2d(name="near"), far])
    assert widget.n_datasets == 2
    assert {"near", "far"} <= {layer.name for layer in viewer.layers}


def test_compat_reports_missing_internals_clearly():
    from napari_storm.napari_particles._napari_compat import (
        NapariInternalsChanged,
        get_layer_controls,
        get_layer_visual,
        set_builtin_layer_docks_visible,
    )

    class NotAViewer:
        pass

    with pytest.raises(NapariInternalsChanged):
        get_layer_visual(NotAViewer(), object())

    # Cosmetic lookups degrade to None rather than raising.
    assert get_layer_controls(NotAViewer(), object()) is None
    assert set_builtin_layer_docks_visible(NotAViewer(), False) is False


def test_napari_shading_control_is_left_usable(make_napari_viewer):
    """We must not repurpose napari's shading combo box.

    QtSurfaceControls.changeShading assigns the combo's data straight to
    napari's `shading` property, so clearing that combo made the next signal
    assign None and selecting one of our shader names assigned 'gaussian' --
    both rejected by the Shading enum with ValueError.
    """
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset_2d()])
    layer = widget.data_to_layer_itf.layer_for(widget.localization_datasets[0])

    # Every value napari's own control can produce must be assignable.
    for mode in ("none", "flat", "smooth"):
        layer.shading = mode
        assert layer.shading == mode

    from napari_storm.napari_particles._napari_compat import get_layer_controls

    controls = get_layer_controls(viewer, layer)
    combo = getattr(controls, "shadingComboBox", None)
    if combo is not None:
        entries = {combo.itemData(i) for i in range(combo.count())}
        assert entries <= {
            "none",
            "flat",
            "smooth",
        }, f"napari's shading combo was repurposed: {entries}"


def test_filter_update_preserves_layer_and_visual_identity(make_napari_viewer):
    """P0-01: a filter/range change must update the layer, not replace it."""
    viewer = make_napari_viewer()
    widget = _billboard_widget(viewer)
    dataset = _dataset_2d()
    widget.get_dataset_from_test_mode([dataset])

    layer = widget.data_to_layer_itf.layer_for(dataset)
    visual = layer._visual
    gaussian_filter = layer.filter[0]
    callbacks = _forced_visual_count()
    n_layers = len(viewer.layers)

    widget.render_config.range_x_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_layers()

    assert widget.data_to_layer_itf.layer_for(dataset) is layer
    assert layer._visual is visual
    assert layer.filter[0] is gaussian_filter
    assert _forced_visual_count() == callbacks
    assert len(viewer.layers) == n_layers
    # And it really did re-filter: fewer localizations are drawn than were loaded.
    assert 0 < len(layer._coords) < dataset.number_of_entries()


def test_a_filter_that_empties_a_dataset_hides_it_reversibly(make_napari_viewer):
    """An empty selection must not hand the renderer a zero-vertex mesh."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    dataset = _dataset_2d()
    widget.get_dataset_from_test_mode([dataset])
    layer = widget.data_to_layer_itf.layer_for(dataset)

    widget.render_config.range_x_percent = np.array([0, 0])
    widget.render_config.range_y_percent = np.array([100, 100])
    widget.data_to_layer_itf.update_layers()
    assert layer.visible is False
    assert widget.data_to_layer_itf.layer_for(dataset) is layer

    widget.render_config.range_x_percent = np.array([0, 100])
    widget.render_config.range_y_percent = np.array([0, 100])
    widget.data_to_layer_itf.update_layers()
    assert layer.visible is True
    assert layer.n_localizations == dataset.number_of_entries()
