"""Remove beads: COMET's thin_beads in front of the drift estimate."""

import json

import numpy as np
import pytest

from napari_storm.postprocessing import comet_runner
from napari_storm.postprocessing.comet_runner import (
    CometParameters,
    bead_removal_available,
    bead_text,
    prepare_input,
    remove_beads,
    run_comet,
)
from napari_storm.postprocessing.drift import DriftModel

from .test_postprocessing_review import _load, dock  # noqa: F401  (fixture)

needs_thin_beads = pytest.mark.skipif(
    not bead_removal_available(),
    reason="the installed COMET has no thin_beads (comet-smlm 1.3 and later)",
)

N_FRAMES = 400


def _truth(frames):
    return np.column_stack([frames * 0.1, frames * -0.04, np.zeros(len(frames))])


def _with_bead(bead_motion=1.0, seed=0):
    """A drifting field of blinking emitters plus one bead in every frame.

    The bead follows the drift (bead_motion=1) or moves against it (-1).
    """
    rng = np.random.default_rng(seed)
    emitters = np.column_stack([rng.uniform(0, 3000, (400, 2)), np.zeros(400)])
    frames = np.repeat(np.arange(N_FRAMES), 20)
    field = emitters[rng.integers(0, 400, len(frames))] + _truth(frames)
    field += rng.normal(0, 2, field.shape) * [1, 1, 0]
    bead_frames = np.arange(N_FRAMES)
    bead = (
        1500.0
        + bead_motion * _truth(bead_frames)
        + rng.normal(0, 2, (N_FRAMES, 3)) * [1, 1, 0]
    )
    coords = np.vstack([field, bead])
    times = np.r_[frames, bead_frames].astype(float)
    is_bead = np.r_[np.zeros(len(field), bool), np.ones(N_FRAMES, bool)]
    return coords, times, is_bead


def test_bead_text():
    assert bead_text(None) == ""
    assert bead_text({"n_beads": 0, "n_locs_removed": 0}) == "no beads found"
    assert (
        bead_text({"n_beads": 1, "n_locs_removed": 1234})
        == "1 bead thinned (1,234 rows left out of the estimate)"
    )
    assert bead_text({"n_beads": 6, "n_locs_removed": 5}).startswith("6 beads thinned")


def test_off_by_default():
    assert CometParameters().remove_beads is False
    data = np.zeros((5, 4))
    out, summary = remove_beads(data, CometParameters())
    assert out is data and summary is None


def test_unavailable_without_comet(monkeypatch):
    monkeypatch.setattr(comet_runner, "find_spec", lambda name: None)
    assert not bead_removal_available()


def test_an_older_comet_says_what_it_needs(monkeypatch):
    comet = pytest.importorskip("comet")
    monkeypatch.delattr(comet, "thin_beads", raising=False)
    with pytest.raises(RuntimeError, match="comet-smlm 1.3"):
        remove_beads(np.zeros((5, 4)), CometParameters(remove_beads=True))


@needs_thin_beads
def test_the_bead_is_thinned_and_the_field_kept():
    coords, times, is_bead = _with_bead()
    data = prepare_input(coords, times, CometParameters())
    params = CometParameters(max_drift_nm=30, remove_beads=True)
    out, summary = remove_beads(data, params)
    assert summary["n_beads"] == 1
    assert summary["n_locs_removed"] > 0.5 * is_bead.sum()
    assert len(out) == len(data) - summary["n_locs_removed"]
    # every field row survives: the rows kept are a superset of the field
    field = {tuple(row) for row in data[~is_bead]}
    assert field <= {tuple(row) for row in out}
    json.dumps(summary)


@needs_thin_beads
@pytest.mark.parametrize("remove", [False, True])
def test_the_memory_budget_counts_the_rows_comet_will_pair(monkeypatch, remove):
    coords, times, is_bead = _with_bead()
    seen = {}

    class Stop(Exception):
        pass

    def estimate(rows, *args, **kwargs):
        seen["rows"] = len(rows)
        raise Stop

    monkeypatch.setattr(comet_runner, "estimate_memory", estimate)
    params = CometParameters(window=400, max_drift_nm=30, remove_beads=remove)
    with pytest.raises(Stop):
        run_comet(coords, times, params)
    _, summary = remove_beads(prepare_input(coords, times, params), params)
    removed = summary["n_locs_removed"] if remove else 0
    assert seen["rows"] == len(coords) - removed
    assert (removed > 0) == remove


@needs_thin_beads
def test_a_bead_moving_on_its_own_no_longer_steers_the_drift():
    coords, times, _ = _with_bead(bead_motion=-1.0)
    truth = _truth(times)

    def error(model):
        residual = model.evaluate(times) - truth
        residual -= residual.mean(axis=0)
        return np.abs(residual[:, :2]).max()

    with_bead = run_comet(coords, times, CometParameters(window=400, max_drift_nm=60))
    removed = run_comet(
        coords, times, CometParameters(window=400, max_drift_nm=60, remove_beads=True)
    )
    assert removed.provenance["beads"]["n_beads"] == 1
    assert removed.provenance["parameters"]["remove_beads"] is True
    assert "beads" not in with_bead.provenance
    assert error(removed) < 1.0
    assert error(with_bead) > 2 * error(removed)
    json.dumps(removed.provenance)


# ---- dock ------------------------------------------------------------------


def test_the_checkbox_reaches_the_parameters(dock, monkeypatch):  # noqa: F811
    import napari_storm.PostProcessing as module

    monkeypatch.setattr(module, "bead_removal_available", lambda: True)
    monkeypatch.setattr(module, "find_spec", lambda name: object())
    _load(dock)
    post = dock.postprocessing
    post.sync()
    assert post.widget.remove_beads.isEnabled()
    assert post.parameters().remove_beads is False
    post.widget.remove_beads.setChecked(True)
    assert post.parameters().remove_beads is True


def test_disabled_when_comet_cannot_remove_beads(dock, monkeypatch):  # noqa: F811
    import napari_storm.PostProcessing as module

    monkeypatch.setattr(module, "bead_removal_available", lambda: False)
    monkeypatch.setattr(module, "find_spec", lambda name: object())
    _load(dock)
    post = dock.postprocessing
    post.widget.remove_beads.setChecked(True)
    post.sync()
    assert not post.widget.remove_beads.isEnabled()
    assert "comet-smlm 1.3" in post.widget.remove_beads.toolTip()
    # a box ticked while it was available does not slip through
    assert post.parameters().remove_beads is False


def test_the_status_says_how_many_beads_were_thinned(dock):  # noqa: F811
    d = _load(dock)
    post = dock.postprocessing
    model = DriftModel(
        [0, 99],
        [[0.0, 0, 0], [10.0, 0, 0]],
        provenance={"pairs": 10, "beads": {"n_beads": 2, "n_locs_removed": 300}},
    )
    post.apply(d, model, d.table.column("frame_number"))
    assert (
        "2 beads thinned (300 rows left out of the estimate)"
        in post.widget.status.text()
    )
