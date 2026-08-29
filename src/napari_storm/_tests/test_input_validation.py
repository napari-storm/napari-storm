"""Numeric entry must not raise or reach the shader unbounded.

Covers register items P1-10 (no input validation) and P1-15 (Z FWHM field
seeded from the XY value) from docs/modernization-review.md.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.ns_constants import FWHM_TO_SIGMA, MAX_FWHM_NM, MIN_FWHM_NM


class _Field:
    """Minimal stand-in for a QLineEdit."""

    def __init__(self, value):
        self._value = value

    def text(self):
        return self._value


@pytest.mark.parametrize(
    "garbage",
    ["", "-", ".", "1e", "abc", "nan", "inf", "-inf", None],
)
def test_unparseable_input_keeps_current_value(garbage):
    """A debounce timer fires mid-typing; that must never raise (P1-10)."""
    current = 42.0
    assert napari_storm._read_fwhm(_Field(garbage), current) == current


def test_value_is_converted_from_fwhm_to_sigma():
    assert napari_storm._read_fwhm(_Field("20"), 1.0) == pytest.approx(
        20 / FWHM_TO_SIGMA
    )


def test_extreme_values_are_clamped():
    """An unbounded Gaussian footprint can stall the GPU (P0-04)."""
    huge = napari_storm._read_fwhm(_Field("1e12"), 1.0)
    assert huge == pytest.approx(MAX_FWHM_NM / FWHM_TO_SIGMA)

    tiny = napari_storm._read_fwhm(_Field("0"), 1.0)
    assert tiny == pytest.approx(MIN_FWHM_NM / FWHM_TO_SIGMA)

    negative = napari_storm._read_fwhm(_Field("-500"), 1.0)
    assert negative > 0


def test_fwhm_fields_have_validators(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    for name in ("Esigma_xy", "Esigma_z", "Esigma_min_xy", "Esigma_min_z"):
        field = getattr(widget, name)
        assert field.validator() is not None, f"{name} has no validator"


def test_z_field_is_seeded_from_the_z_value(make_napari_viewer, monkeypatch):
    """The Z FWHM box used to display the XY value (P1-15).

    DEFAULT_FIXED_FWHM_XY_NM and _Z_NM are both 20 today, so the defect is
    invisible with stock defaults.  Give the two axes different values at
    construction time to expose it.
    """
    from napari_storm.render_config import RenderConfig

    def _asymmetric_config():
        cfg = RenderConfig()
        cfg.fixed_sigma_xy_nm = 20.0 / FWHM_TO_SIGMA
        cfg.fixed_sigma_z_nm = 55.0 / FWHM_TO_SIGMA
        return cfg

    monkeypatch.setattr("napari_storm._dock_widget.RenderConfig", _asymmetric_config)
    widget = napari_storm(napari_viewer=make_napari_viewer())

    assert float(widget.Esigma_xy.text()) == pytest.approx(20.0, rel=1e-4)
    assert float(widget.Esigma_z.text()) == pytest.approx(55.0, rel=1e-4)


def test_update_sigma_survives_garbage_in_the_field(make_napari_viewer):
    """End-to-end: the slot itself must not raise on partial input."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    before = widget.render_fixed_gauss_sigma_xy_nm
    widget.Esigma_xy.setText("")
    widget.update_sigma()  # must not raise
    assert widget.render_fixed_gauss_sigma_xy_nm == before
