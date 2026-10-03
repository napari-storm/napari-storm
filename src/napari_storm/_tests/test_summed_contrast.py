"""Contrast applied to the summed image: the lower limit hides sparse regions.

Additive footprints used to be windowed per localization, before the sum, so
the lower contrast limit could only dim every splat alike.  With
`LayerAppearance.summed_contrast` the window acts on the sum, in units of
summed weight -- overlapping localizations, for fixed-size Gaussians.

Rendering reads pixels from `_scene_canvas.render()`, as `test_footprint_palette`
does; what was drawn is found in the image, not assumed at its centre.
"""

import numpy as np
import pytest
from scipy import ndimage

from napari_storm.core import LayerAppearance, RenderRequest
from napari_storm.core.render_planner import SIGMA_TO_SIZE_FACTOR
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer

SIGMA_NM = 200.0
NM_PER_PX = 10.0


def _request(positions_nm, values=None, colormap="gray"):
    """Localizations at *positions_nm* along one axis, all on one plane."""
    positions_nm = np.asarray(positions_nm, dtype=np.float32)
    coords = np.zeros((len(positions_nm), 3), dtype=np.float32)
    coords[:, 0] = 1.0
    coords[:, 2] = positions_nm
    return RenderRequest(
        coords=coords,
        sigmas=np.full((len(positions_nm), 3), SIGMA_NM, dtype=np.float32),
        size=SIGMA_TO_SIZE_FACTOR * SIGMA_NM,
        values=(
            np.ones(len(positions_nm), dtype=np.float32)
            if values is None
            else np.asarray(values, dtype=np.float32)
        ),
        name="summed",
        colormap=colormap,
    )


def _viewer(make_napari_viewer, request, **appearance):
    viewer = make_napari_viewer()
    renderer = InstancedRenderer(viewer)
    renderer.open(1, request)
    if appearance:
        renderer.set_appearance(1, LayerAppearance(**appearance))
    viewer.dims.ndisplay = 2
    viewer.camera.center = (0.0, 0.0)
    viewer.camera.zoom = 1.0 / NM_PER_PX
    return viewer, renderer


def _render(viewer, **kw):
    scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
    return np.asarray(scene_canvas.render(**kw))[..., :3].astype(int)


def _blobs(image, threshold=40):
    """How many separate lit regions the image has."""
    _labels, count = ndimage.label(image.max(axis=2) >= threshold)
    return count


#: One localization alone, and five stacked 1.5 um away.
SPARSE_AND_DENSE = [-750.0] + [750.0] * 5


# ---------------------------------------------------------------- the model


def test_summed_contrast_is_the_default_for_an_additive_footprint(make_napari_viewer):
    _viewer_, renderer = _viewer(make_napari_viewer, _request([0.0]))
    assert renderer.contrast_is_summed(1)
    assert renderer.appearance(1).summed_contrast is True


def test_opaque_footprints_keep_the_per_localization_window(make_napari_viewer):
    _viewer_, renderer = _viewer(make_napari_viewer, _request([0.0]), footprint="disc")
    # Opaque markers have no sum: the nearest one is all a pixel keeps.
    assert not renderer.contrast_is_summed(1)
    renderer.set_appearance(1, LayerAppearance(footprint="glow"))
    assert renderer.contrast_is_summed(1)


def test_z_colour_coding_turns_it_off(make_napari_viewer):
    _viewer_, renderer = _viewer(
        make_napari_viewer, _request([0.0]), summed_contrast=False
    )
    assert not renderer.contrast_is_summed(1)


def test_backends_that_cannot_say_no_by_default():
    from napari_storm.core.renderer import LocalizationRenderer

    assert LocalizationRenderer.contrast_is_summed(object(), 1) is False


# ----------------------------------------------------------- what is drawn


def test_the_default_window_looks_as_it_did(make_napari_viewer):
    """At (0, 1) one Gaussian peaks at full brightness either way."""
    viewer, renderer = _viewer(make_napari_viewer, _request([0.0]))
    summed = _render(viewer)
    renderer.set_appearance(1, LayerAppearance(summed_contrast=False))
    per_localization = _render(viewer)
    assert summed.max() >= 250
    assert np.abs(summed - per_localization).max() <= 3


def test_a_colormap_that_starts_transparent_is_not_squared(make_napari_viewer):
    """napari-storm's channel colormaps ramp alpha from 0 along with colour.

    Blending the resolve by that alpha multiplied each pixel's colour by
    itself, so a lone Gaussian lost most of its visible extent.
    """
    from napari.utils.colormaps import Colormap

    red_ramp = Colormap(colors=[[0, 0, 0, 0], [1, 0, 0, 1]], name="red ramp")
    viewer, renderer = _viewer(make_napari_viewer, _request([0.0], colormap=red_ramp))
    summed = _render(viewer)
    renderer.set_appearance(1, LayerAppearance(summed_contrast=False))
    per_localization = _render(viewer)
    assert per_localization[..., 0].max() >= 250
    assert np.abs(summed - per_localization).max() <= 3


def test_a_colormap_that_does_not_start_at_black_has_no_hard_edge(
    make_napari_viewer,
):
    """Viridis starts at purple, not black.

    Unfaded, every pixel a splat touched took the lowest colour out to the
    edge of its quad and then cut to black: each splat a hard-edged square,
    (69, 3, 86) one pixel and (0, 0, 0) the next.
    """
    viewer, _renderer = _viewer(
        make_napari_viewer,
        _request([0.0], colormap="viridis"),
        contrast_limits=(0.0, 3.0),
    )
    image = _render(viewer).max(axis=2)
    row = image[np.unravel_index(np.argmax(image), image.shape)[0]]
    assert row.max() > 100, "nothing was drawn to measure"
    assert np.abs(np.diff(row)).max() <= 12


def test_the_lower_limit_hides_sparse_regions_and_keeps_dense_ones(
    make_napari_viewer,
):
    # (0, 5): the lone localization peaks at a fifth of full brightness.
    viewer, renderer = _viewer(
        make_napari_viewer, _request(SPARSE_AND_DENSE), contrast_limits=(0.0, 5.0)
    )
    assert _blobs(_render(viewer)) == 2

    renderer.set_appearance(1, LayerAppearance(contrast_limits=(1.5, 10.0)))
    assert _blobs(_render(viewer)) == 1, "the lone localization should vanish"


def test_per_localization_the_lower_limit_could_not(make_napari_viewer):
    """The defect this replaces: every localization has the value 1, so a
    cutoff above 1 removed the dense cluster along with the lone one."""
    viewer, _renderer = _viewer(
        make_napari_viewer,
        _request(SPARSE_AND_DENSE),
        contrast_limits=(1.5, 10.0),
        summed_contrast=False,
    )
    assert _blobs(_render(viewer)) == 0


def test_the_upper_limit_counts_overlaps(make_napari_viewer):
    """Five stacked Gaussians in a window of (0, 10) peak at half brightness."""
    viewer, _renderer = _viewer(
        make_napari_viewer, _request([0.0] * 5), contrast_limits=(0.0, 10.0)
    )
    assert _render(viewer).max() == pytest.approx(127.5, abs=4)


def test_the_sum_is_not_clipped_before_the_window(make_napari_viewer):
    """Forty stacked in (20, 60): the old 8-bit canvas held at most 1."""
    viewer, _renderer = _viewer(
        make_napari_viewer, _request([0.0] * 40), contrast_limits=(20.0, 60.0)
    )
    assert _render(viewer).max() == pytest.approx(127.5, abs=4)


def test_values_weight_the_sum(make_napari_viewer):
    """Variable-size mode: a tight localization counts for more than a loose one."""
    viewer, _renderer = _viewer(
        make_napari_viewer,
        _request([0.0, 0.0], values=[0.5, 0.25]),
        contrast_limits=(0.0, 1.0),
    )
    assert _render(viewer).max() == pytest.approx(0.75 * 255, abs=4)


def test_a_hidden_channel_draws_nothing(make_napari_viewer):
    viewer, _renderer = _viewer(make_napari_viewer, _request([0.0]), opacity=0.0)
    assert _render(viewer).max() <= 1


def test_opacity_scales_the_resolved_image(make_napari_viewer):
    viewer, _renderer = _viewer(make_napari_viewer, _request([0.0]), opacity=0.5)
    assert _render(viewer).max() == pytest.approx(127.5, abs=4)


def test_a_screenshot_at_another_size_still_windows_the_sum(make_napari_viewer):
    """napari renders screenshots into its own buffer; the target follows it."""
    viewer, _renderer = _viewer(
        make_napari_viewer, _request(SPARSE_AND_DENSE), contrast_limits=(1.5, 10.0)
    )
    scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
    width, height = scene_canvas.size
    image = _render(viewer, size=(2 * width, 2 * height))
    assert image.shape[:2] == (2 * height, 2 * width)
    assert _blobs(image) == 1


def test_closing_gives_the_node_its_draw_back(make_napari_viewer):
    viewer, renderer = _viewer(make_napari_viewer, _request([0.0]))
    node = renderer._layers[1]._visual
    assert "draw" in node.__dict__
    renderer.close(1)
    assert "draw" not in node.__dict__


# ------------------------------------------------------------ the controls


def _dock(make_napari_viewer, dataset):
    from napari_storm._dock_widget import napari_storm

    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.get_dataset_from_test_mode([dataset])
    channel = widget.channel[0]
    renderer = widget.data_to_layer_itf.renderer
    dataset_id = widget.localization_datasets[0].dataset_id
    return widget, channel, lambda: renderer.appearance(dataset_id).contrast_limits


def _choose(widget, name):
    widget.Bfootprint.setCurrentIndex(widget.Bfootprint.findData(name))


def test_both_handles_share_a_log_axis_of_overlaps(make_napari_viewer, two_d_dataset):
    _widget, channel, limits = _dock(make_napari_viewer, two_d_dataset)
    assert channel._summed

    channel.Slider_colormap_range.setValue((0.75, 1.0))
    assert limits() == pytest.approx((10.0, 100.0))
    assert channel.cutoff_spin.value() == pytest.approx(10.0)

    channel.cutoff_spin.setValue(1.0)
    assert channel.Slider_colormap_range.value()[0] == pytest.approx(0.5)
    assert limits() == pytest.approx((1.0, 100.0))


def test_a_marker_footprint_keeps_its_own_handles(make_napari_viewer, two_d_dataset):
    widget, channel, limits = _dock(make_napari_viewer, two_d_dataset)
    channel.Slider_colormap_range.setValue((0.75, 0.875))
    summed_limits = limits()

    _choose(widget, "disc")
    assert not channel._summed
    # The per-localization model starts from its own default, not from a
    # threshold of ten overlaps reread as a value.
    assert channel.Slider_colormap_range.value() == pytest.approx((0.0, 0.5))
    assert limits() == pytest.approx((0.0, 1.0))

    _choose(widget, "gaussian")
    assert channel._summed
    assert channel.Slider_colormap_range.value() == pytest.approx((0.75, 0.875))
    assert limits() == pytest.approx(summed_limits)


def test_z_colour_coding_windows_each_localization(make_napari_viewer, three_d_dataset):
    widget, channel, _limits = _dock(make_napari_viewer, three_d_dataset)
    renderer = widget.data_to_layer_itf.renderer
    dataset_id = widget.localization_datasets[0].dataset_id

    widget.Bz_color_coding.setChecked(True)
    assert not renderer.contrast_is_summed(dataset_id)
    assert not channel._summed

    widget.Bz_color_coding.setChecked(False)
    assert renderer.contrast_is_summed(dataset_id)
    assert channel._summed
