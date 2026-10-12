"""Colour coding by any property, with a choice of colormap and editable limits.

Colour coding used to mean one thing: rainbow (hsv) by z, each dataset scaled
to its own depth range, with a bar whose two numbers were only true of the scene
while one dataset was loaded. It now takes any numeric property the datasets
share -- or a position axis -- and any of the offered colormaps, and its limits
are one pair for the whole scene: following the data by default, typed by the
user otherwise.

The bar's ends used to be the literal words "min" and "max" pushed apart by
spaces (issue #37); they are anchored under the colours they describe.
"""

import numpy as np
import pytest

from napari_storm.core import DatasetTraits, GaussianSettings, RenderPlanner
from napari_storm.core.localization_table import LocalizationTable
from napari_storm.core.validation import InvalidLocalizationData
from napari_storm.GUI import ColorCodingWidget
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.ns_constants import (
    COLOR_CODING_COLORMAPS,
    DEFAULT_COLOR_CODING_COLORMAP,
)


def _records(n=11, zdim=True):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    fields += [("frame", "i4"), ("photons", "f4")]
    locs = np.zeros(n, dtype=fields)
    locs["x_pos_nm"] = np.linspace(0, 5000, n)
    locs["y_pos_nm"] = np.linspace(0, 5000, n)
    if zdim:
        locs["z_pos_nm"] = np.linspace(-500, 500, n)
    locs["frame"] = np.arange(n) * 100
    locs["photons"] = np.linspace(100, 1100, n)
    return np.rec.array(locs)


def _dataset(name="d", zdim=True, n=11):
    return LocalizationDataBaseClass(_records(n, zdim), name=name, zdim_present=zdim)


def _values(color_by, limits=None, table=None, zdim=True):
    table = table if table is not None else LocalizationTable(_records(zdim=zdim))
    return RenderPlanner().values(
        table.selection(),
        GaussianSettings(color_by=color_by, color_limits=limits),
        DatasetTraits(zdim_present=zdim),
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_any_column_maps_onto_the_colormap():
    np.testing.assert_allclose(_values("frame"), np.linspace(0, 1, 11), atol=1e-6)
    np.testing.assert_allclose(_values("photons"), np.linspace(0, 1, 11), atol=1e-6)


def test_an_axis_is_read_in_nanometres():
    np.testing.assert_allclose(_values("z"), np.linspace(0, 1, 11), atol=1e-6)
    np.testing.assert_allclose(_values("x", zdim=False), np.linspace(0, 1, 11))


def test_limits_put_their_values_at_the_ends_and_clip_beyond():
    values = _values("frame", limits=(200, 600))
    assert values[:3].tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(values[2:7], np.linspace(0, 1, 5))
    assert values[6:].tolist() == [1.0] * 5


def test_limits_wider_than_the_data_leave_the_ends_unused():
    values = _values("frame", limits=(-1000, 2000))
    assert values.min() == pytest.approx(1 / 3)
    assert values.max() == pytest.approx(2 / 3)


def test_a_degenerate_range_draws_the_middle_colour_rather_than_failing():
    np.testing.assert_allclose(_values("frame", limits=(5, 5)), 0.5)
    table = LocalizationTable(_records())
    table.set_column("photons", np.full(len(table), 7.0, dtype=np.float32))
    np.testing.assert_allclose(_values("photons", table=table), 0.5)


def test_a_missing_value_takes_the_low_colour():
    table = LocalizationTable(_records())
    photons = np.asarray(table.column("photons")).copy()
    photons[3] = np.nan
    table.set_column("photons", photons)
    values = _values("photons", table=table)
    assert values[3] == 0.0
    assert np.isfinite(values).all()


def test_colour_follows_the_drawn_rows_only():
    table = LocalizationTable(_records())
    mask = np.zeros(len(table), dtype=bool)
    mask[5:] = True
    table.set_filter_mask(mask)
    np.testing.assert_allclose(_values("frame", table=table), np.linspace(0, 1, 6))


def test_an_unknown_property_or_a_flat_z_says_so():
    with pytest.raises(InvalidLocalizationData, match="nonsense"):
        _values("nonsense")
    with pytest.raises(InvalidLocalizationData, match="z"):
        _values("z", zdim=False)


def test_a_rewritten_side_column_is_not_served_stale():
    table = LocalizationTable(_records())
    table.set_side_column("group_id", np.arange(len(table)))
    first = _values("group_id", table=table)
    table.set_side_column("group_id", np.arange(len(table))[::-1])
    np.testing.assert_allclose(_values("group_id", table=table), first[::-1])


# ---------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------


def test_hsv_is_the_default_and_the_choices_include_grey_and_diverging(qtbot):
    bar = ColorCodingWidget()
    qtbot.addWidget(bar)
    assert bar.colormap == DEFAULT_COLOR_CODING_COLORMAP == "hsv"
    offered = {name for _label, name in COLOR_CODING_COLORMAPS}
    assert {"gray", "viridis", "coolwarm", "PiYG"} <= offered


def test_every_offered_colormap_exists_in_napari():
    from napari.utils.colormaps import ensure_colormap

    for _label, name in COLOR_CODING_COLORMAPS:
        assert ensure_colormap(name).name == name


def test_the_ends_of_the_bar_carry_the_range(qtbot):
    bar = ColorCodingWidget()
    qtbot.addWidget(bar)
    bar.set_property("z (depth)", "nm")
    bar.set_range(-300.0, 1500.0)
    assert (bar.low_edit.text(), bar.high_edit.text()) == ("-300", "1500")
    assert bar.title.text() == "Colour code: z (depth)"
    assert bar.unit_label.text() == "nm"


def test_without_a_range_it_claims_nothing(qtbot):
    bar = ColorCodingWidget()
    qtbot.addWidget(bar)
    bar.set_range(0.0, 100.0)
    bar.set_range(np.inf, -np.inf)
    assert (bar.low_edit.text(), bar.high_edit.text()) == ("", "")
    assert (bar.low_edit.placeholderText(), bar.high_edit.placeholderText()) == (
        "min",
        "max",
    )


def test_typing_a_limit_turns_auto_off_and_announces_both(qtbot):
    bar = ColorCodingWidget()
    qtbot.addWidget(bar)
    bar.set_range(0.0, 100.0, auto=True)
    edited = []
    bar.limits_edited.connect(lambda low, high: edited.append((low, high)))

    bar.high_edit.setText("50")
    bar.high_edit.editingFinished.emit()
    assert edited == [(0.0, 50.0)]
    assert not bar.Cauto.isChecked()


def test_leaving_a_field_unchanged_is_not_an_edit(qtbot):
    bar = ColorCodingWidget()
    qtbot.addWidget(bar)
    bar.set_range(0.0, 100.0, auto=True)
    edited = []
    bar.limits_edited.connect(lambda *limits: edited.append(limits))
    bar.low_edit.editingFinished.emit()
    assert edited == []
    assert bar.Cauto.isChecked()


# ---------------------------------------------------------------------------
# The dock widget
# ---------------------------------------------------------------------------


def _dock(make_napari_viewer, *datasets):
    from napari_storm._dock_widget import napari_storm

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode(list(datasets))
    return widget


def _appearance(widget, index=0):
    renderer = widget.data_to_layer_itf.renderer
    return renderer.appearance(widget.localization_datasets[index].dataset_id)


def _offered(widget):
    combo = widget.Bcolor_by
    return [combo.itemData(i) for i in range(combo.count())]


def test_every_shared_numeric_property_is_offered(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    assert _offered(widget) == [None, "x", "y", "z", "frame", "photons"]


def test_flat_data_can_be_colour_coded_too_but_not_by_z(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset(zdim=False))
    assert _offered(widget) == [None, "x", "y", "frame", "photons"]
    assert widget.Wcolor_by.isVisibleTo(widget)


def test_choosing_a_property_colours_by_it_with_hsv(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.Bcolor_by.setCurrentIndex(widget.Bcolor_by.findData("frame"))

    assert widget.render_config.color_by == "frame"
    appearance = _appearance(widget)
    assert getattr(appearance.colormap, "name", appearance.colormap) == "hsv"
    # Overlaps blend to their mean colour; the window sets brightness.
    assert appearance.summed_contrast and appearance.color_coded
    values = widget.data_to_layer_itf.render_state[
        widget.localization_datasets[0].dataset_id
    ].values
    # Auto limits are the 1st and 99th percentile.
    assert values.min() == 0.0 and values.max() == 1.0
    assert widget.Lcolor_encoding_bar.title.text() == "Colour code: frame"


def test_another_colormap_reaches_every_channel(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset("a"), _dataset("b"))
    widget.set_color_coding(color_by="photons")
    bar = widget.Lcolor_encoding_bar
    bar.colormap_selector.setCurrentIndex(bar.colormap_selector.findData("coolwarm"))

    assert widget.render_config.color_colormap == "coolwarm"
    for index in (0, 1):
        colormap = _appearance(widget, index).colormap
        assert getattr(colormap, "name", colormap) == "coolwarm"


def test_the_limits_are_shared_by_every_dataset(make_napari_viewer):
    low = _dataset("low")
    high_records = _records()
    high_records["frame"] += 1000
    high = LocalizationDataBaseClass(high_records, name="high", zdim_present=True)
    widget = _dock(make_napari_viewer, low, high)
    widget.set_color_coding(color_by="frame", limits=(0, 2000))

    state = widget.data_to_layer_itf.render_state
    first = state[widget.localization_datasets[0].dataset_id].values
    second = state[widget.localization_datasets[1].dataset_id].values
    np.testing.assert_allclose(first, np.arange(11) * 100 / 2000)
    np.testing.assert_allclose(second, (np.arange(11) * 100 + 1000) / 2000)
    assert widget.Lcolor_encoding_bar.high_edit.text() == "2000"
    assert not widget.Lcolor_encoding_bar.Cauto.isChecked()


def test_auto_hands_the_limits_back_to_the_data(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="frame", limits=(0, 5000))
    widget.Lcolor_encoding_bar.Cauto.setChecked(True)
    assert widget.render_config.color_limits is None
    low, high = widget.data_to_layer_itf.color_limits()
    assert low == pytest.approx(10.0) and high == pytest.approx(990.0)


def test_unticking_auto_keeps_the_colours_where_they_are(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="frame")
    auto = widget.data_to_layer_itf.color_limits()
    widget.Lcolor_encoding_bar.Cauto.setChecked(False)
    assert widget.render_config.color_limits == pytest.approx(auto)


def test_another_property_drops_limits_in_the_old_unit(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="frame", limits=(0, 5000))
    widget.set_color_coding(color_by="photons")
    assert widget.render_config.color_limits is None


def test_z_limits_follow_the_render_range_window(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="z")
    assert widget.data_to_layer_itf.color_limits() == pytest.approx((-500.0, 500.0))


def test_off_gives_each_channel_its_own_colormap_back(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    own = _appearance(widget).colormap
    widget.set_color_coding(color_by="z")
    widget.set_color_coding(color_by=None)
    assert _appearance(widget).colormap == own
    assert _appearance(widget).summed_contrast
    assert not _appearance(widget).color_coded
    assert not widget.Lcolor_encoding_bar.isVisibleTo(widget)


def test_variable_mode_switches_colour_coding_off(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="frame")
    widget.render_gaussian_mode = 1
    assert widget.render_config.color_by is None
    assert not widget.Wcolor_by.isVisibleTo(widget)
    with pytest.raises(ValueError, match="fixed"):
        widget.set_color_coding(color_by="frame")


def test_the_old_z_switch_still_works_for_embedders(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    widget.z_color_encoding_mode = 1
    assert widget.render_config.color_by == "z"
    assert widget.render_config.z_color_encoding == 1
    widget.z_color_encoding_mode = 0
    assert widget.render_config.color_by is None


# ---------------------------------------------------------------------------
# Blending: overlaps take their mean colour, density sets brightness
# ---------------------------------------------------------------------------

SIGMA_NM = 200.0
NM_PER_PX = 10.0


def _render_request(positions_nm, values, colormap):
    from napari_storm.core import RenderRequest
    from napari_storm.core.render_planner import SIGMA_TO_SIZE_FACTOR

    positions_nm = np.asarray(positions_nm, dtype=np.float32)
    coords = np.zeros((len(positions_nm), 3), dtype=np.float32)
    coords[:, 0] = 1.0
    coords[:, 2] = positions_nm
    return RenderRequest(
        coords=coords,
        sigmas=np.full((len(positions_nm), 3), SIGMA_NM, dtype=np.float32),
        size=SIGMA_TO_SIZE_FACTOR * SIGMA_NM,
        values=np.asarray(values, dtype=np.float32),
        name="coded",
        colormap=colormap,
    )


def _draw(make_napari_viewer, positions_nm, values, colormap, **appearance):
    """The centre pixel's colour, in [0, 1], of colour-coded localizations."""
    from napari_storm.core import LayerAppearance
    from napari_storm.napari_particles.instanced_renderer import InstancedRenderer

    viewer = make_napari_viewer()
    renderer = InstancedRenderer(viewer)
    renderer.open(1, _render_request(positions_nm, values, colormap))
    renderer.set_appearance(
        1,
        LayerAppearance(
            **{"color_coded": True, "contrast_limits": (0.0, 1.0), **appearance}
        ),
    )
    viewer.dims.ndisplay = 2
    viewer.camera.center = (0.0, 0.0)
    viewer.camera.zoom = 1.0 / NM_PER_PX
    canvas = viewer.window._qt_viewer.canvas._scene_canvas
    image = np.asarray(canvas.render())[..., :3].astype(float) / 255.0
    # The middle of what was drawn: an unshown canvas does not put the camera
    # centre at the middle of the image.
    rows, cols = np.nonzero(image.max(axis=2) > 0.02)
    if rows.size == 0:
        return np.zeros(3), renderer
    return image[int(round(rows.mean())), int(round(cols.mean()))], renderer


def test_overlapping_localizations_keep_their_colour(make_napari_viewer):
    """Ten mid-grey splats on one spot stay mid-grey, at full brightness.

    Added straight onto the canvas they summed to ten times grey: white.
    """
    centre, renderer = _draw(make_napari_viewer, [0.0] * 10, [0.5] * 10, "gray")
    assert renderer.contrast_is_summed(1)
    np.testing.assert_allclose(centre, 0.5, atol=0.03)


def test_different_colours_on_one_spot_blend_to_their_mean(make_napari_viewer):
    """Blue-white-red at 0 and 1: as many of each give their average, purple."""
    centre, _ = _draw(make_napari_viewer, [0.0] * 6, [0.0, 1.0] * 3, "bwr")
    from napari.utils.colormaps import ensure_colormap

    mean = ensure_colormap("bwr").map(np.array([0.0, 1.0]))[:, :3].mean(axis=0)
    np.testing.assert_allclose(centre, mean, atol=0.03)


def test_the_window_sets_brightness_and_leaves_the_colour(make_napari_viewer):
    """Above the window's top a spot is full colour; halfway up, half of it."""
    full, _ = _draw(make_napari_viewer, [0.0], [1.0], "bwr", contrast_limits=(0, 1))
    half, _ = _draw(make_napari_viewer, [0.0], [1.0], "bwr", contrast_limits=(0, 2))
    np.testing.assert_allclose(half, full / 2, atol=0.03)


def test_hsv_still_draws_a_lone_localization_in_its_own_hue(make_napari_viewer):
    from napari.utils.colormaps import ensure_colormap

    centre, _ = _draw(make_napari_viewer, [0.0], [1 / 3], "hsv")
    expected = ensure_colormap("hsv").map(np.array([1 / 3]))[0, :3]
    np.testing.assert_allclose(centre, expected, atol=0.03)


def test_a_hidden_channel_adds_nothing(make_napari_viewer):
    centre, _ = _draw(make_napari_viewer, [0.0] * 3, [0.5] * 3, "gray", opacity=0.0)
    assert centre.max() == 0.0


def test_switching_back_restores_the_summed_intensity_image(make_napari_viewer):
    from napari_storm.core import LayerAppearance

    centre, renderer = _draw(make_napari_viewer, [0.0], [1.0], "gray")
    renderer.set_appearance(1, LayerAppearance(color_coded=False))
    assert not renderer.appearance(1).color_coded
    assert renderer.contrast_is_summed(1)


# ---------------------------------------------------------------------------
# Channel controls
# ---------------------------------------------------------------------------


def test_colour_coding_keeps_contrast_and_adds_opacity(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    channel = widget.channel[0]
    widget.set_color_coding(color_by="frame")
    assert not channel.Colormap_selector.isVisibleTo(channel)
    assert channel.Slider_colormap_range.isVisibleTo(channel)
    assert channel.Slider_opacity.isVisibleTo(channel)

    widget.set_color_coding(color_by=None)
    assert channel.Colormap_selector.isVisibleTo(channel)
    assert not channel.Slider_opacity.isVisibleTo(channel)


def test_an_opaque_footprint_colours_each_marker_on_its_own(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    channel = widget.channel[0]
    widget.set_color_coding(color_by="frame")
    widget.Bfootprint.setCurrentIndex(widget.Bfootprint.findData("disc"))

    assert not channel._summed
    assert tuple(_appearance(widget).contrast_limits) == pytest.approx((0.0, 1.0))
    assert not channel.Slider_colormap_range.isVisibleTo(channel)
    assert channel.Slider_opacity.isVisibleTo(channel)


def test_showing_a_hidden_coded_channel_brings_it_back(make_napari_viewer):
    widget = _dock(make_napari_viewer, _dataset())
    channel = widget.channel[0]
    widget.set_color_coding(color_by="frame")
    channel.show_channel()
    assert _appearance(widget).opacity == 0.0
    channel.show_channel()
    assert _appearance(widget).opacity == 1.0


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def test_a_scene_round_trips_its_colour_coding():
    from napari_storm.core import ColorCodingEntry, Scene

    scene = Scene(color_coding=ColorCodingEntry("frame", "coolwarm", (0.0, 500.0)))
    back = Scene.from_dict(scene.to_dict())
    assert back.color_coding == ColorCodingEntry("frame", "coolwarm", (0.0, 500.0))

    auto = Scene(color_coding=ColorCodingEntry("z", "hsv", None))
    assert Scene.from_dict(auto.to_dict()).color_coding.limits is None


def test_a_scene_that_says_nothing_leaves_colour_coding_alone():
    from napari_storm.core import Scene

    raw = Scene().to_dict()
    assert "color_coding" not in raw
    assert Scene.from_dict(raw).color_coding is None


def test_an_old_scene_with_z_colour_coding_still_colours_by_z():
    from napari_storm.core import ColorCodingEntry, Scene

    raw = Scene().to_dict()
    raw["gaussian"]["z_color_encoding"] = True
    assert Scene.from_dict(raw).color_coding == ColorCodingEntry("z")
    raw["gaussian"]["z_color_encoding"] = False
    assert Scene.from_dict(raw).color_coding == ColorCodingEntry(None)


def test_bad_limits_in_a_scene_are_refused():
    from napari_storm.core import Scene, SceneFormatError

    raw = Scene().to_dict()
    raw["color_coding"] = {"color_by": "frame", "limits": [1.0]}
    with pytest.raises(SceneFormatError, match="two values"):
        Scene.from_dict(raw)


def test_saving_and_loading_a_scene_restores_colour_coding(
    make_napari_viewer, tmp_path
):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="frame", colormap="viridis", limits=(100, 900))
    path = widget.save_scene_to(tmp_path / "coded.json")

    widget.set_color_coding(color_by=None, colormap="hsv")
    widget.load_scene_from(path)
    config = widget.render_config
    assert (config.color_by, config.color_colormap) == ("frame", "viridis")
    assert config.color_limits == (100.0, 900.0)
    assert widget.Bcolor_by.currentData() == "frame"
    assert widget.Lcolor_encoding_bar.colormap == "viridis"
    colormap = _appearance(widget).colormap
    assert getattr(colormap, "name", colormap) == "viridis"


def test_a_scene_saved_with_auto_limits_loads_with_auto(make_napari_viewer, tmp_path):
    widget = _dock(make_napari_viewer, _dataset())
    widget.set_color_coding(color_by="photons")
    path = widget.save_scene_to(tmp_path / "auto.json")
    widget.set_color_coding(limits=(1, 2))
    widget.load_scene_from(path)
    assert widget.render_config.color_limits is None
    assert widget.Lcolor_encoding_bar.Cauto.isChecked()


def test_a_scene_saved_without_colour_coding_switches_it_off(
    make_napari_viewer, tmp_path
):
    widget = _dock(make_napari_viewer, _dataset())
    path = widget.save_scene_to(tmp_path / "plain.json")
    widget.set_color_coding(color_by="frame")
    widget.load_scene_from(path)
    assert widget.render_config.color_by is None


def test_a_scene_property_the_data_lacks_is_reported(make_napari_viewer, monkeypatch):
    from napari_storm.core import ColorCodingEntry, Scene

    widget = _dock(make_napari_viewer, _dataset())
    warnings = []
    monkeypatch.setattr(widget, "_warn_user", warnings.append)
    widget.apply_scene(Scene(color_coding=ColorCodingEntry("lifetime")))
    assert widget.render_config.color_by is None
    assert warnings and "lifetime" in warnings[0]
