"""The export dialog, which exists to show consequences before they happen.

Two of its jobs are not cosmetic and are tested as behaviour:

* the file size implied by the chosen pixel size is visible *while choosing*,
  because "never downsample" means a fine pixel size can produce a very large
  file and that must not be a surprise;
* a view thinned by the render budget says so before the export runs, not
  after, since afterwards is too late to act on.
"""
import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.image_export import (SCOPE_EVERYTHING, ExportOptions,
                                       plan_from_widget)
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.pyqt.export_dialog import (ExportImageDialog, describe_plan,
                                             format_size)


def _dataset(name="ds", n=200, zdim=False):
    fields = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
    if zdim:
        fields.append(("z_pos_nm", "f4"))
    locs = np.zeros(n, dtype=fields)
    locs["x_pos_nm"] = np.linspace(1_000, 3_000, n)
    locs["y_pos_nm"] = np.linspace(1_000, 7_000, n)
    if zdim:
        locs["z_pos_nm"] = np.linspace(-400, 400, n)
    return LocalizationDataBaseClass(np.rec.array(locs), name=name, zdim_present=zdim)


def _dialog(make_napari_viewer, thinned=False, zdim=False):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_dataset(n=1000, zdim=zdim)])
    if thinned:
        widget.localization_datasets[0].table.limit_active_to(100)
    dialog = ExportImageDialog(lambda options: plan_from_widget(widget, options))
    return dialog, widget


# ------------------------------------------------------------------ format


@pytest.mark.parametrize(
    "nbytes,expected",
    [(512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")],
)
def test_sizes_are_readable(nbytes, expected):
    assert format_size(nbytes) == expected


def test_the_summary_states_dimensions_size_and_count(make_napari_viewer):
    dialog, widget = _dialog(make_napari_viewer)
    plan = plan_from_widget(widget, ExportOptions(pixel_size_nm=10.0))

    text = describe_plan(plan)

    assert "px" in text
    assert "1 channel" in text
    assert "1,000 localizations" in text


# ------------------------------------------------------- live consequences


def test_the_summary_follows_the_pixel_size(make_napari_viewer):
    """The cost of 'never downsample', visible while choosing."""
    dialog, _ = _dialog(make_napari_viewer)

    dialog.pixel_size.setValue(20.0)
    coarse = dialog.plan().nbytes
    dialog.pixel_size.setValue(5.0)
    fine = dialog.plan().nbytes

    assert fine == pytest.approx(16 * coarse, rel=0.05)
    assert dialog.summary.text()


def test_choosing_3d_enables_the_z_step(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer, zdim=True)
    assert not dialog.z_step.isEnabled()

    dialog.everything.setChecked(True)

    assert dialog.z_step.isEnabled()
    assert dialog.options().scope == SCOPE_EVERYTHING


def test_a_3d_choice_produces_a_stack_in_the_summary(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer, zdim=True)
    planes_2d = dialog.plan().shape[1]

    dialog.everything.setChecked(True)
    dialog.z_step.setValue(100.0)

    assert dialog.plan().shape[1] > planes_2d


# ------------------------------------------------------------- the warning


def test_a_thinned_view_is_flagged_before_the_export_runs(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer, thinned=True)

    assert dialog.warning.isVisibleTo(dialog)
    assert "render budget" in dialog.warning.text()
    assert "100" in dialog.warning.text()


def test_an_unthinned_view_shows_no_warning(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer, thinned=False)

    assert not dialog.warning.isVisibleTo(dialog)


# -------------------------------------------------------------- the button


def test_saving_is_blocked_until_there_is_a_path(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer)
    assert not dialog._save_button.isEnabled()

    dialog.path.setText("/tmp/out.ome.tif")

    assert dialog._save_button.isEnabled()


def test_a_plan_that_cannot_be_built_is_reported_not_raised(make_napari_viewer):
    """A dialog that raises on a keystroke is worse than one that explains."""

    def _explode(_options):
        raise ValueError("no datasets loaded")

    dialog = ExportImageDialog(_explode)

    assert dialog.plan() is None
    assert "no datasets loaded" in dialog.summary.text()
    assert not dialog._save_button.isEnabled()


def test_the_options_read_back_what_was_chosen(make_napari_viewer):
    dialog, _ = _dialog(make_napari_viewer, zdim=True)

    dialog.pixel_size.setValue(7.5)
    dialog.everything.setChecked(True)
    dialog.z_step.setValue(33.0)
    dialog.path.setText("/tmp/x.ome.tif")

    options = dialog.options()
    assert options.pixel_size_nm == 7.5
    assert options.scope == SCOPE_EVERYTHING
    assert options.z_step_nm == 33.0
    assert dialog.output_path() == "/tmp/x.ome.tif"
