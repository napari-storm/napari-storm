import numpy as np
import pytest

from napari_storm import napari_storm
from napari_storm.pyqt.image_import_dialog import ImageImportResult
from napari_storm.pyqt.image_layer_controls import (
    _quarter_turn_matrix,
    _reference_image_rendering_options,
)


def _reference_image(name="reference", rgb=False, rgba=False, orientation="XY"):
    color_channels = 4 if rgba else 3
    is_color = rgb or rgba
    if orientation == "3D":
        shape = (2, 3, 4, color_channels) if is_color else (2, 3, 4)
    else:
        shape = (2, 4, color_channels) if is_color else (2, 4)
    return ImageImportResult(
        file_path=f"{name}.tif",
        img=np.zeros(shape, dtype=np.uint8),
        orientation=orientation,
        px_xy_nm=100.0,
        px_z_nm=250.0,
        x_off_nm=10_000.0,
        y_off_nm=20_000.0,
        z_off_nm=300.0,
        layer_name=name,
    )


def _layer_centre_in_world(layer):
    shape = np.asarray(layer.data.shape, dtype=float)
    if layer.rgb:
        shape = shape[:-1]
    centre = (shape[-layer.ndim :] - 1) / 2
    return np.asarray(layer.data_to_world(tuple(centre)))


def test_reference_images_do_not_change_camera_projection(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    viewer.camera.perspective = 37

    widget._add_image_layer_from_result(_reference_image("first"))
    widget._add_image_layer_from_result(_reference_image("second"))
    first, second = widget.image_layer_controls

    assert viewer.camera.perspective == 37
    first._on_remove()
    assert viewer.camera.perspective == 37
    assert first._layer not in viewer.layers

    second._on_remove()
    assert viewer.camera.perspective == 37
    assert second._layer not in viewer.layers


@pytest.mark.parametrize(
    ("orientation", "normal"),
    [
        # Normals in the layer's (z, y, x) order: the plane is named by the two
        # axes it spans, so its normal is the third.
        ("XY", (1, 0, 0)),   # constant z
        ("XZ", (0, 1, 0)),   # constant y
        ("YZ", (0, 0, 1)),   # constant x
    ],
)
@pytest.mark.parametrize("rgb", [False, True], ids=["grayscale", "rgb"])
def test_planar_references_use_no_depth_plane_rendering(
    make_napari_viewer, orientation, normal, rgb
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(
        _reference_image(orientation=orientation, rgb=rgb)
    )
    layer = widget.image_layer_controls[0]._layer

    spatial_shape = np.asarray(layer.data.shape[:-1] if layer.rgb else layer.data.shape)
    expected_position = (spatial_shape - 1) / 2
    assert layer.depiction == "plane"
    assert layer.blending == "translucent_no_depth"
    np.testing.assert_allclose(layer.plane.position, expected_position)
    np.testing.assert_allclose(layer.plane.normal, normal)
    assert layer.plane.thickness == 1


@pytest.mark.parametrize("rgb", [False, True], ids=["grayscale", "rgb"])
def test_volumetric_reference_keeps_normal_volume_rendering(
    make_napari_viewer, rgb
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(
        _reference_image(orientation="3D", rgb=rgb)
    )
    layer = widget.image_layer_controls[0]._layer

    assert layer.rgb is rgb
    assert layer.depiction == "volume"
    assert _reference_image_rendering_options(
        layer.data, "3D", rgb=rgb
    ) == {}


def test_grayscale_reference_has_live_contrast_and_colormap_controls(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(_reference_image())
    controls = widget.image_layer_controls[0]

    controls._contrast_slider.setValue((20.0, 180.0))
    np.testing.assert_allclose(controls._layer.contrast_limits, (20.0, 180.0))

    controls._colormap_combo.setCurrentText("viridis")
    assert controls._layer.colormap.name == "viridis"


def test_rgb_reference_omits_scalar_appearance_controls(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(_reference_image(rgb=True))
    controls = widget.image_layer_controls[0]

    assert controls._layer.rgb
    assert controls._contrast_slider is None
    assert controls._colormap_combo is None
    assert controls._opacity_slider is None


def test_rgba_reference_has_live_uniform_opacity_control(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(_reference_image(rgba=True))
    controls = widget.image_layer_controls[0]

    assert controls._layer.rgb
    assert controls._contrast_slider is None
    assert controls._colormap_combo is None
    assert controls._opacity_slider.value() == 100

    controls._opacity_slider.setValue(37)

    assert controls._layer.opacity == pytest.approx(0.37)


def test_rotation_buttons_share_rows_with_their_offset_fields(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(_reference_image())
    controls = widget.image_layer_controls[0]
    spins = {
        "x": controls._x_off_spin,
        "y": controls._y_off_spin,
        "z": controls._z_off_spin,
    }

    for axis, spinbox in spins.items():
        row_layout = controls._position_rows[axis].layout()
        assert row_layout.indexOf(spinbox) >= 0
        assert row_layout.indexOf(controls._rotation_buttons[(axis, 1)]) >= 0
        assert row_layout.indexOf(controls._rotation_buttons[(axis, -1)]) >= 0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("rgb", [False, True], ids=["grayscale", "rgb"])
def test_rotation_buttons_turn_about_each_axis_without_moving_the_centre(
    make_napari_viewer, axis, rgb
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget._add_image_layer_from_result(_reference_image(rgb=rgb))
    controls = widget.image_layer_controls[0]
    layer = controls._layer
    centre_before = _layer_centre_in_world(layer)

    controls._rotation_buttons[(axis, 1)].click()

    np.testing.assert_allclose(layer.rotate, _quarter_turn_matrix(axis))
    np.testing.assert_allclose(
        _layer_centre_in_world(layer), centre_before, atol=1e-9
    )

    controls._rotation_buttons[(axis, -1)].click()
    np.testing.assert_allclose(layer.rotate, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        _layer_centre_in_world(layer), centre_before, atol=1e-9
    )
