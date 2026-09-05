"""The z colour bar names the interval it encodes, not just its direction.

Its ends carried the literal words "min" and "max", pushed apart by a run of
spaces inside one centred label (issue #37). That said which way z ran and
nothing about over what, and the spacing only lined up with the bar's ends at
the font size it was counted out for.

The scene-wide claim needs care: the planner normalizes every dataset against
its own z extent, so one pair of numbers is only true of the whole scene while
there is one dataset in it.
"""

import numpy as np
import pytest

from napari_storm.GUI import ZColorCodingColorBarWidget, _format_z_nm


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "0 nm"),
        (-450.0, "-450 nm"),
        (999.0, "999 nm"),
        (1000.0, "1.00 µm"),
        (-2500.0, "-2.50 µm"),
    ],
)
def test_a_z_coordinate_reads_in_the_unit_that_suits_its_size(value, expected):
    assert _format_z_nm(value) == expected


def test_the_ends_of_the_bar_carry_the_range(qtbot):
    bar = ZColorCodingColorBarWidget()
    qtbot.addWidget(bar)
    bar.set_range(-300.0, 1500.0)
    assert bar.low_label.text() == "-300 nm"
    assert bar.high_label.text() == "1.50 µm"


def test_without_a_range_it_claims_nothing(qtbot):
    """No dataset, or no usable z extent, leaves the old wording in place."""
    bar = ZColorCodingColorBarWidget()
    qtbot.addWidget(bar)
    assert (bar.low_label.text(), bar.high_label.text()) == ("min", "max")

    bar.set_range(0.0, 100.0)
    bar.set_range(np.inf, -np.inf)
    assert (bar.low_label.text(), bar.high_label.text()) == ("min", "max")

    bar.set_range(None, None)
    assert (bar.low_label.text(), bar.high_label.text()) == ("min", "max")


def test_several_datasets_are_told_apart_from_one(qtbot):
    """Each is scaled to its own z, so the numbers stop speaking for all."""
    bar = ZColorCodingColorBarWidget()
    qtbot.addWidget(bar)
    bar.show()

    bar.set_range(0.0, 500.0, shared=True)
    assert not bar.note.isVisible()

    bar.set_range(0.0, 500.0, shared=False)
    assert bar.note.isVisible()


def test_the_note_stays_hidden_when_there_is_no_range_to_qualify(qtbot):
    bar = ZColorCodingColorBarWidget()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_range(None, None, shared=False)
    assert not bar.note.isVisible()
