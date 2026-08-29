"""A format declares its column names and units for *every* measured column.

Positions could always be declared (`position_columns`, `position_scale_nm`);
sigmas and photon counts could not, because the planner read them through
hardcoded pixel-native names. A format storing nanometres therefore rendered in
fixed-width mode and raised `AttributeError` in variable-width mode -- the one
mode where its fitted widths matter.

These tests pin the symmetric contract, and the unit-correctness property that
had to come with it: the repair floor for an unusable width is a physical
length, so it means the same thing whatever unit the column is stored in.
"""

import numpy as np
import pytest

from napari_storm.core import (
    DatasetTraits,
    GaussianSettings,
    InvalidLocalizationData,
    LocalizationTable,
    RenderPlanner,
)

PIXEL_SIZE_NM = 100.0

NM_DTYPE = [
    ("x_nm", "f4"),
    ("y_nm", "f4"),
    ("z_nm", "f4"),
    ("sigma_x_nm", "f4"),
    ("sigma_y_nm", "f4"),
    ("sigma_z_nm", "f4"),
    ("photons", "f4"),
]

PIXEL_DTYPE = [
    ("x_pos_pixels", "f4"),
    ("y_pos_pixels", "f4"),
    ("z_pos_pixels", "f4"),
    ("sigma_x_pixels", "f4"),
    ("sigma_y_pixels", "f4"),
    ("sigma_z_pixels", "f4"),
    ("photon_count", "f4"),
]

TRAITS = DatasetTraits(
    zdim_present=True,
    sigma_present=True,
    photon_count_present=True,
    pixel_size_nm=PIXEL_SIZE_NM,
)


def _widths(n, rng, zeros=0):
    """Fitted widths in *pixels*, with *zeros* dead rows at the front."""
    sx = rng.uniform(0.1, 0.6, n).astype("f4")
    sy = rng.uniform(0.1, 0.6, n).astype("f4")
    sz = rng.uniform(0.2, 1.2, n).astype("f4")
    sx[:zeros] = 0.0
    return sx, sy, sz


def _nm_table(n=2000, seed=0, zeros=0):
    """A host that stores nanometres, declaring its own names and unit."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = _widths(n, rng, zeros)
    records = np.rec.array(np.zeros(n, dtype=NM_DTYPE))
    records.x_nm = rng.uniform(0, 10_000, n)
    records.y_nm = rng.uniform(0, 10_000, n)
    records.z_nm = rng.uniform(-500, 500, n)
    records.sigma_x_nm = sx * PIXEL_SIZE_NM
    records.sigma_y_nm = sy * PIXEL_SIZE_NM
    records.sigma_z_nm = sz * PIXEL_SIZE_NM
    records.photons = rng.uniform(100, 5000, n)
    return LocalizationTable(
        records,
        position_columns={"x": "x_nm", "y": "y_nm", "z": "z_nm"},
        position_scale_nm=1.0,
        sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
        sigma_scale_nm=1.0,
        photon_column="photons",
        copy=False,
    )


def _pixel_table(n=2000, seed=0, zeros=0):
    """The same measurements, stored the way the bundled readers store them."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = _widths(n, rng, zeros)
    records = np.rec.array(np.zeros(n, dtype=PIXEL_DTYPE))
    records.x_pos_pixels = rng.uniform(0, 10_000, n) / PIXEL_SIZE_NM
    records.y_pos_pixels = rng.uniform(0, 10_000, n) / PIXEL_SIZE_NM
    records.z_pos_pixels = rng.uniform(-500, 500, n) / PIXEL_SIZE_NM
    records.sigma_x_pixels, records.sigma_y_pixels, records.sigma_z_pixels = sx, sy, sz
    records.photon_count = rng.uniform(100, 5000, n)
    return LocalizationTable(
        records,
        # What `StormDataClass` declares; sigmas and photons need nothing,
        # which is the point of the defaults.
        position_columns={
            "x": "x_pos_pixels",
            "y": "y_pos_pixels",
            "z": "z_pos_pixels",
        },
        position_scale_nm=PIXEL_SIZE_NM,
        copy=False,
    )


# ------------------------------------------------------- the contract itself


def test_a_nanometre_format_renders_in_variable_width_mode():
    """The request from the ImSwitch2 integration: this used to raise."""
    request = RenderPlanner().plan(
        _nm_table(), GaussianSettings(mode=1), TRAITS, name="nm-host"
    )

    assert np.all(np.isfinite(request.sigmas))
    assert np.all(np.isfinite(request.values))


def test_declared_columns_are_read_instead_of_the_pixel_native_names():
    """Nothing downstream may reach past the table for a hardcoded name."""
    table = _nm_table()

    assert table.has_sigma_axis("x") and table.has_photons()
    assert not table.has_field("sigma_x_pixels")
    # The whole point: no `sigma_x_pixels` anywhere in these records.
    assert set(table.field_names) == {name for name, _ in NM_DTYPE}


@pytest.mark.parametrize("mode", [0, 1])
def test_the_unit_a_format_stores_does_not_change_what_is_drawn(mode):
    """The same measurements in nm and in pixels must plan identically."""
    settings = GaussianSettings(mode=mode)
    planner = RenderPlanner()

    from_nm = planner.plan(_nm_table(), settings, TRAITS, name="nm")
    from_pixels = planner.plan(_pixel_table(), settings, TRAITS, name="px")

    assert np.allclose(from_nm.sigmas, from_pixels.sigmas, rtol=1e-5)
    assert np.allclose(from_nm.values, from_pixels.values, rtol=1e-5)
    assert from_nm.size == pytest.approx(from_pixels.size, rel=1e-5)


def test_that_holds_when_dead_rows_have_to_be_repaired():
    """The property the old absolute sentinel quietly broke.

    `MIN_USABLE_UNCERTAINTY` was substituted into whatever unit the column
    used, so a repaired row meant 1e-3 px under one schema and 1e-3 nm under
    the other -- a hundredfold difference in width, and because the weight is
    `1 / product`, six orders of magnitude in intensity. The percentile clip
    kept the median honest, so it showed up in no summary statistic; it showed
    up as a value range a handful of dead rows had taken over.
    """
    settings = GaussianSettings(mode=1)
    planner = RenderPlanner()

    from_nm = planner.plan(_nm_table(zeros=5), settings, TRAITS, name="nm")
    from_pixels = planner.plan(_pixel_table(zeros=5), settings, TRAITS, name="px")

    assert np.allclose(from_nm.values, from_pixels.values, rtol=1e-5)


def test_a_repaired_row_is_weighted_as_it_is_actually_drawn():
    """A dead row must not dominate the range it is normalized against."""
    settings = GaussianSettings(mode=1)
    request = RenderPlanner().plan(_nm_table(zeros=5), settings, TRAITS, name="nm")

    repaired, real = request.values[:5], request.values[5:]
    # Before the fix the repaired rows sat ~10^5x above the real ones.
    assert repaired.max() < 10 * real.max()


@pytest.mark.parametrize("width_nm", [4.5, 30.0, 100.0, 200.0])
def test_dead_rows_do_not_set_the_scale_at_any_fitted_width(width_nm):
    """The property the first attempt at this only had at one width.

    Repairing to the declared floor made the units agree but left the value
    blowout in place, because the floor is the *narrowest* width and the weight
    is ``1 / product`` -- so a failed fit became the brightest row in the
    dataset. It looked fixed only because the fixture's widths sat a factor of
    two above the floor. A real PSF is 100-200 nm, twenty-five to fifty times
    it, and there the real localizations were back down in the bottom 0.02% of
    their own value range.

    Parametrized over that whole span deliberately: this is a test that has
    already been passed by a broken implementation once.
    """
    n, dead = 2000, 5
    rng = np.random.default_rng(1)
    widths = rng.normal(width_nm, width_nm * 0.15, n).clip(width_nm * 0.5)
    records = np.rec.array(np.zeros(n, dtype=NM_DTYPE))
    records.sigma_x_nm = widths
    records.sigma_y_nm = widths
    records.sigma_z_nm = widths * 2.0
    # A failed fit writes no width on any axis, not just one.
    for column in ("sigma_x_nm", "sigma_y_nm", "sigma_z_nm"):
        getattr(records, column)[:dead] = 0.0
    table = LocalizationTable(
        records,
        sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
        sigma_scale_nm=1.0,
        copy=False,
    )

    values = RenderPlanner().values(table.selection(), GaussianSettings(mode=1), TRAITS)

    assert values[dead:].max() == pytest.approx(values.max(), rel=1e-6)


def test_a_degenerate_width_distribution_plans_rather_than_raising():
    """A fitter writing one nominal width for most rows used to produce NaN.

    ``_normalized`` puts the majority at zero, the 99th percentile is then zero
    too, and dividing by it produced NaN -- surfacing downstream as "render
    values has no positive finite maximum", which names neither the column nor
    the cause.
    """
    n = 2000
    widths = np.full(n, 150.0, dtype="f4")
    widths[:5] = 80.0  # a handful genuinely fitted, the rest nominal
    records = np.rec.array(np.zeros(n, dtype=NM_DTYPE))
    records.sigma_x_nm = widths
    records.sigma_y_nm = widths
    records.sigma_z_nm = widths * 2.0
    table = LocalizationTable(
        records,
        sigma_columns={"x": "sigma_x_nm", "y": "sigma_y_nm", "z": "sigma_z_nm"},
        sigma_scale_nm=1.0,
        copy=False,
    )

    values = RenderPlanner().values(table.selection(), GaussianSettings(mode=1), TRAITS)

    assert np.all(np.isfinite(values))
    # The tighter rows are still the brighter ones; nothing is inverted.
    assert values[:5].min() >= values[5:].max()


def test_the_repair_is_still_reported_under_the_declared_column_name():
    """A warning naming a column the user does not have is not a warning."""
    reported = []
    RenderPlanner(on_repaired=lambda column, n, total: reported.append(column)).plan(
        _nm_table(zeros=5), GaussianSettings(mode=1), TRAITS, name="nm"
    )

    assert reported and all(name == "sigma_x_nm" for name in reported)


# ------------------------------------------------------------ what was broken


def test_a_two_dimensional_fit_needs_no_axial_width_column():
    """An ordinary 2-D file used to raise a bare AttributeError.

    `sigmas()` read the axial column whenever `sigma_present`, while
    `_variable_values()` guarded on `zdim_present`. A 2-D fit records no axial
    width, so the two disagreed on a file neither had any reason to refuse.
    """
    rng = np.random.default_rng(0)
    records = np.rec.array(
        np.zeros(
            100,
            dtype=[
                ("x_pos_nm", "f4"),
                ("y_pos_nm", "f4"),
                ("sigma_x_pixels", "f4"),
                ("sigma_y_pixels", "f4"),
            ],
        )
    )
    records.x_pos_nm = rng.uniform(0, 1000, 100)
    records.y_pos_nm = rng.uniform(0, 1000, 100)
    records.sigma_x_pixels = np.full(100, 0.3, dtype="f4")
    records.sigma_y_pixels = np.full(100, 0.3, dtype="f4")
    table = LocalizationTable(records, position_scale_nm=1.0)

    request = RenderPlanner().plan(
        table,
        GaussianSettings(mode=1),
        DatasetTraits(
            zdim_present=False, sigma_present=True, pixel_size_nm=PIXEL_SIZE_NM
        ),
        name="flat",
    )

    assert np.all(np.isfinite(request.sigmas))


def test_an_all_dead_width_column_still_refuses_rather_than_guessing():
    """The §2.2 case: declaring a width that was never fitted is an error."""
    table = _nm_table()
    table.set_column("sigma_z_nm", np.zeros(len(table), dtype="f4"))

    with pytest.raises(InvalidLocalizationData, match="sigma_z_nm"):
        RenderPlanner().plan(table, GaussianSettings(mode=1), TRAITS, name="nm")


# --------------------------------------------------------------- the defaults


def test_the_defaults_are_what_the_bundled_readers_already_write():
    """Existing formats keep working without passing anything new."""
    table = _pixel_table()

    assert table.has_sigma_axis("z") and table.has_photons()
    assert table.sigma_column("x") == "sigma_x_pixels"


def test_sigma_scale_follows_the_position_scale_by_default():
    """One unit decision per dataset, not two that can drift apart.

    A reader that learns its pixel size late writes it once; widths must not
    stay converted with the old value.
    """
    table = _pixel_table()
    assert table.sigma_scale_nm == PIXEL_SIZE_NM

    table.position_scale_nm = 160.0

    assert table.sigma_scale_nm == 160.0
    assert table.sigma_nm("x")[0] == pytest.approx(
        table.column("sigma_x_pixels")[0] * 160.0, rel=1e-5
    )


def test_an_explicit_sigma_scale_is_not_overridden_by_the_position_scale():
    table = _pixel_table()
    table.sigma_scale_nm = 1.0

    table.position_scale_nm = 160.0

    assert table.sigma_scale_nm == 1.0
