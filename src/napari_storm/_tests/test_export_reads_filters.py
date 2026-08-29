"""An export must contain what the user selected, not what the GPU could afford.

This is §4.1's third consequence of "never downsample", and the reason Level 2
split one mask into two:

    "Filters are an intentional choice about the data; budget thinning is an
    accommodation to the GPU, and it has no business in a saved result."

The failure this prevents is silent and serious: writing an evenly strided
subsample of a scientist's data to disk because their graphics card was busy,
in a file that looks complete.
"""

import numpy as np

from napari_storm.core import (
    ACTIVE,
    FILTERED,
    DatasetTraits,
    GaussianSettings,
    LocalizationTable,
    RenderPlanner,
)


def _table(n=1000):
    records = np.rec.array(np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")]))
    records.x_pos_nm = np.arange(n, dtype="f4")
    records.y_pos_nm = np.arange(n, dtype="f4") * 2
    return LocalizationTable(records)


def _filtered_and_thinned(n=1000, keep=600, budget=100):
    """A table where the two masks genuinely differ."""
    table = _table(n)
    mask = np.zeros(n, dtype=bool)
    mask[:keep] = True
    table.set_filter_mask(mask)
    hidden = table.limit_active_to(budget)
    assert hidden > 0, "fixture must actually be display-limited"
    return table


# ------------------------------------------------------------- the selection


def test_the_two_selections_differ_when_the_budget_bites():
    table = _filtered_and_thinned()

    assert table.is_display_limited
    assert table.selection(FILTERED).n == 600
    assert table.selection(ACTIVE).n == 100


def test_planning_for_export_covers_every_filtered_localization():
    """The screen draws 100 of them; the file must contain all 600."""
    table = _filtered_and_thinned()
    planner = RenderPlanner()

    drawn = planner.plan(
        table, GaussianSettings(), DatasetTraits(), name="ch", selection=ACTIVE
    )
    exported = planner.plan(
        table, GaussianSettings(), DatasetTraits(), name="ch", selection=FILTERED
    )

    assert len(drawn.coords) == 100
    assert len(exported.coords) == 600
    assert len(exported.values) == 600
    assert len(exported.sigmas) == 600


def test_the_export_selection_is_the_filter_set_exactly():
    """Not a different subsample, and not the whole table either."""
    table = _filtered_and_thinned()

    exported = RenderPlanner().plan(
        table, GaussianSettings(), DatasetTraits(), name="ch", selection=FILTERED
    )

    assert exported.active_ids.tolist() == list(range(600))
    # Column 2 is x, which the fixture sets to arange; coordinates are (z, y, x).
    assert np.array_equal(exported.coords[:, 2], np.arange(600, dtype=np.float32))


def test_the_drawn_selection_is_a_subset_of_the_exported_one():
    """Budget thinning narrows the user's choice; it never widens it."""
    table = _filtered_and_thinned()
    planner = RenderPlanner()

    drawn = planner.plan(
        table, GaussianSettings(), DatasetTraits(), name="ch", selection=ACTIVE
    )
    exported = planner.plan(
        table, GaussianSettings(), DatasetTraits(), name="ch", selection=FILTERED
    )

    assert set(drawn.active_ids).issubset(set(exported.active_ids))


def test_planning_defaults_to_the_drawn_selection():
    """So every existing render path keeps its meaning after the change."""
    table = _filtered_and_thinned()

    default = RenderPlanner().plan(
        table, GaussianSettings(), DatasetTraits(), name="ch"
    )

    assert len(default.coords) == table.n_active


def test_without_a_budget_the_two_selections_agree():
    """The common case: nothing thinned, so nothing to warn about either."""
    table = _table(50)
    mask = np.zeros(50, dtype=bool)
    mask[:30] = True
    table.set_filter_mask(mask)

    assert not table.is_display_limited
    assert table.selection(FILTERED).n == table.selection(ACTIVE).n == 30


def test_an_unknown_selection_is_refused():
    import pytest

    with pytest.raises(ValueError, match="filtered"):
        _table(4).selection("everything")
