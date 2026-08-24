"""Render memory and screen-space Gaussian cost are bounded, not unbounded.

Covers P0-04: an over-budget dataset or an over-large Gaussian must produce a
warning and deterministic degraded output rather than taking the process down.
"""
import numpy as np

from napari_storm import memory_budget
from napari_storm._dock_widget import napari_storm
from napari_storm._tests.fixtures import extreme_gaussian_dataset, make_dataset
from napari_storm.localization_dataset_types import LocalizationDataBaseClass
from napari_storm.memory_budget import (RENDER_BYTES_PER_LOCALIZATION,
                                        cap_splat_size_nm,
                                        default_render_budget_mb,
                                        max_localizations_for_budget)


def _dataset(n, name="budgeted"):
    locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    locs["x_pos_nm"] = np.arange(n)
    locs["y_pos_nm"] = np.arange(n) * 2
    return LocalizationDataBaseClass(
        np.rec.array(locs), name=name, zdim_present=False
    )


# ------------------------------------------------------------- budget maths


def test_budget_defaults_and_env_override(monkeypatch):
    monkeypatch.delenv(memory_budget.RENDER_BUDGET_ENV_VAR, raising=False)
    assert default_render_budget_mb() == memory_budget.DEFAULT_RENDER_BUDGET_MB

    monkeypatch.setenv(memory_budget.RENDER_BUDGET_ENV_VAR, "512")
    assert default_render_budget_mb() == 512.0

    # A typo must not stop the plugin from loading.
    monkeypatch.setenv(memory_budget.RENDER_BUDGET_ENV_VAR, "lots")
    assert default_render_budget_mb() == memory_budget.DEFAULT_RENDER_BUDGET_MB
    monkeypatch.setenv(memory_budget.RENDER_BUDGET_ENV_VAR, "-8")
    assert default_render_budget_mb() == memory_budget.DEFAULT_RENDER_BUDGET_MB


def test_zero_budget_means_no_limit():
    assert max_localizations_for_budget(0) is None
    assert max_localizations_for_budget(None) is None


def test_budget_converts_to_a_localization_count():
    assert max_localizations_for_budget(1) == int(1e6 // RENDER_BYTES_PER_LOCALIZATION)
    # Absurdly small budgets degrade to a minimal layer, not an impossible one.
    assert max_localizations_for_budget(1e-9) == 1


# ------------------------------------------------------------------ thinning


def test_limit_active_to_is_a_no_op_under_the_limit():
    dataset = _dataset(10)
    assert dataset.limit_active_to(10) == 0
    assert dataset.limit_active_to(None) == 0
    assert dataset.number_of_active_entries() == 10


def test_limit_active_to_thins_evenly_and_deterministically():
    dataset = _dataset(100)
    dropped = dataset.limit_active_to(10)

    assert dropped == 90
    assert dataset.number_of_active_entries() == 10
    kept = dataset.x_pos_nm
    # Stride sampling: evenly spaced, so spatial coverage survives thinning.
    assert kept.tolist() == [float(i) for i in range(0, 100, 10)]

    again = _dataset(100)
    again.limit_active_to(10)
    assert again.x_pos_nm.tolist() == kept.tolist()


def test_thinning_composes_with_filtering_and_never_touches_locs_all():
    dataset = _dataset(100)
    dataset.apply_filters(np.arange(50), None)
    dataset.limit_active_to(5)

    assert dataset.number_of_entries() == 100
    assert dataset.number_of_active_entries() == 5
    # Only rows the spatial filter kept may survive the budget.
    assert float(np.max(dataset.x_pos_nm)) < 50


# ------------------------------------------------------------- splat clamping


def test_splat_size_is_clamped_to_a_fraction_of_the_field_of_view():
    assert cap_splat_size_nm(100.0, 1000.0) == (100.0, False)
    assert cap_splat_size_nm(900.0, 1000.0) == (500.0, True)


def test_splat_size_is_left_alone_when_the_extent_is_unknown():
    assert cap_splat_size_nm(900.0, None) == (900.0, False)
    assert cap_splat_size_nm(900.0, 0.0) == (900.0, False)
    assert cap_splat_size_nm(900.0, np.inf, fraction=0) == (900.0, False)


# ------------------------------------------------------------- integration


def test_over_budget_dataset_is_thinned_and_reported_once(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    messages = []
    widget.data_to_layer_itf.on_resource_limit_applied = messages.append
    # 1000 localizations at 352 B each is 0.352 MB; allow a tenth of that.
    widget.render_config.render_budget_mb = 0.0352

    widget.get_dataset_from_test_mode([_dataset(1_000)])
    dataset = widget.localization_datasets[0]

    assert dataset.number_of_entries() == 1_000
    assert dataset.number_of_active_entries() == 100
    assert widget.data_to_layer_itf.layer_for(dataset).n_localizations == 100
    assert len(messages) == 1
    assert "render budget" in messages[0]

    # Repeating the update must not repeat the warning.
    widget.data_to_layer_itf.update_layers()
    assert len(messages) == 1


def test_a_dataset_within_budget_is_untouched(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    messages = []
    widget.data_to_layer_itf.on_resource_limit_applied = messages.append

    widget.get_dataset_from_test_mode([_dataset(1_000)])

    assert widget.localization_datasets[0].number_of_active_entries() == 1_000
    assert messages == []


def test_extreme_gaussian_is_capped_instead_of_flooding_the_viewport(
    make_napari_viewer,
):
    """The measured fixture drew one splat ~12 000x the field of view."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    messages = []
    widget.data_to_layer_itf.on_resource_limit_applied = messages.append

    widget.get_dataset_from_test_mode([extreme_gaussian_dataset()])
    from napari_storm.ns_constants import FWHM_TO_SIGMA

    widget.render_config.fixed_sigma_xy_nm = 20_000.0 / FWHM_TO_SIGMA
    widget.data_to_layer_itf.update_layers()

    itf = widget.data_to_layer_itf
    fov_nm = max(
        itf.render_range_x[1] - itf.render_range_x[0],
        itf.render_range_y[1] - itf.render_range_y[0],
    )
    layer = widget.data_to_layer_itf.layer_for(widget.localization_datasets[0])
    splat_nm = layer.billboard_size_nm

    assert splat_nm <= 0.5 * fov_nm + 1e-6
    assert any("clamped" in message for message in messages)


def test_budget_is_shared_between_loaded_datasets(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.render_config.render_budget_mb = 0.352  # 1000 localizations total

    widget.get_dataset_from_test_mode(
        [_dataset(1_000, "a"), _dataset(1_000, "b")]
    )

    # Two datasets, so each gets half the budget.
    assert [d.number_of_active_entries() for d in widget.localization_datasets] == [
        1_000,
        500,
    ]
    # The first dataset's share shrank when the second arrived; it is re-applied
    # the next time that dataset is updated rather than rebuilding a settled layer.
    widget.data_to_layer_itf.update_layers()
    assert [d.number_of_active_entries() for d in widget.localization_datasets] == [
        500,
        500,
    ]


def test_benchmark_fixtures_fit_the_default_budget():
    """The default budget must not silently distort the benchmark fixtures."""
    dataset = make_dataset(1_000)
    allowed = max_localizations_for_budget(default_render_budget_mb())
    assert allowed is not None and allowed > 5_000_000
    assert dataset.limit_active_to(allowed) == 0


def test_the_budget_hides_rows_without_deselecting_them(make_napari_viewer):
    """P0-04 must not silently edit what an export or a save would see."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.render_config.render_budget_mb = 0.0352  # 100 localizations

    widget.get_dataset_from_test_mode([_dataset(1_000)])
    dataset = widget.localization_datasets[0]

    assert dataset.number_of_active_entries() == 100
    assert dataset.number_of_filtered_entries() == 1_000
    assert dataset.is_display_limited
    assert len(dataset.filtered_coordinate_nm("x")) == 1_000


def test_raising_the_budget_restores_the_full_view(make_napari_viewer):
    widget = napari_storm(napari_viewer=make_napari_viewer())
    widget.render_config.render_budget_mb = 0.0352

    widget.get_dataset_from_test_mode([_dataset(1_000)])
    assert widget.localization_datasets[0].is_display_limited

    widget.render_config.render_budget_mb = 0
    widget.data_to_layer_itf.update_layers()

    dataset = widget.localization_datasets[0]
    assert not dataset.is_display_limited
    assert dataset.number_of_active_entries() == 1_000
