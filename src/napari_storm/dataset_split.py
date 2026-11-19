"""Cutting one dataset in two by a filter, each part a channel of its own.

The way in is :func:`split_by_last_filter`: apply a filter, look at what it
leaves, then split. What the last filter kept -- what is on screen -- becomes
``<name> in`` and what it removed ``<name> out``. Localizations removed by
filters applied before it go to neither part, so a split of photon-filtered
data does not bring the dim ones back in "out". The original is replaced: its
rows now live in the parts.

Each part remembers how it was made. Its scene edit log is the original's,
with the last filter turned into the split, so a scene can open the file
again, redo the edits and the split, and end up with the same two channels.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .core import edits_of, record_edit, split_point

__all__ = [
    "SPLIT_PARTS",
    "split_by_last_filter",
    "split_dataset",
    "split_masks",
]

SPLIT_PARTS = ("in", "out")


def _band(band):
    return {
        "parameter": str(band["parameter"]),
        "low": float(band["low"]),
        "high": float(band["high"]),
        "mode": str(band["mode"]),
    }


def split_masks(widget, dataset, band, kept=None):
    """``{"in": mask, "out": mask}`` over *dataset*'s canonical rows.

    *kept* are the rows that may go into either part; by default those every
    filter applied to *dataset* keeps.
    """
    from .DataFilter import DataFilterInterface

    n = dataset.number_of_entries()
    if not dataset.table.has_field(band["parameter"]):
        raise ValueError(f"{dataset.name} has no parameter {band['parameter']!r}")
    if kept is None:
        kept = np.ones(n, dtype=bool)
        kept[widget.data_filter_itf.indices_for(dataset)] = False
    removed_indices = DataFilterInterface.band_indices(dataset, **_band(band))
    if removed_indices is None:
        raise ValueError(f"unknown filter mode {band['mode']!r}")
    removed = np.zeros(n, dtype=bool)
    removed[removed_indices] = True
    return {"in": kept & ~removed, "out": kept & removed}


def split_by_last_filter(widget, dataset):
    """Split *dataset* into what its last filter kept and what it removed.

    The filter becomes the split: it leaves the parts' edit logs, where the
    split takes its place, because a scene that replayed both would filter
    out everything the "out" part is made of before splitting it off.
    """
    log = list(edits_of(dataset))
    filters = [
        index
        for index in range(split_point(log), len(log))
        if log[index].get("kind") == "filter"
    ]
    steps = widget.data_filter_itf.filter_steps_for(dataset)
    if not filters or len(steps) < len(filters):
        raise ValueError(
            f"No filter has been applied to {dataset.name} yet. Apply one, "
            "then split: the parts are what it kept and what it removed."
        )
    at = filters[-1]
    if any(edit.get("kind") == "density" for edit in log[at + 1 :]):
        raise ValueError(
            "Local density was calculated after the last filter, so it counted "
            "neither part on its own. Apply the filter again after calculating "
            "the density, then split."
        )
    kept = np.ones(dataset.number_of_entries(), dtype=bool)
    for removed in steps[:-1]:
        kept[removed] = False
    masks = split_masks(widget, dataset, log[at], kept=kept)
    if not masks["out"].any():
        raise ValueError(
            f"The last filter removed nothing from {dataset.name} that earlier "
            "filters kept, so there is nothing to split off."
        )
    if not masks["in"].any():
        raise ValueError(
            f"The last filter kept nothing of {dataset.name}, so there is "
            "nothing to split."
        )
    return split_dataset(
        widget, dataset, log[at], kept=kept, before=log[:at], after=log[at + 1 :]
    )


def split_dataset(
    widget, dataset, band, parts=SPLIT_PARTS, *, kept=None, before=None, after=()
):
    """Replace *dataset* in *widget*'s session by its *parts* along *band*.

    *kept* limits the rows either part may take (default: what the filters
    keep). Each part's edit log is *before* (default: the dataset's own),
    the split, then *after* -- edits made after the filter the split stands
    for, already in the values the parts copy.

    Returns ``{part: new dataset}``. Raises ValueError, leaving the session
    as it was, when a part would be empty or the dataset cannot be split
    (drift correction applied, an unknown parameter or mode).
    """
    band = _band(band)
    masks = split_masks(widget, dataset, band, kept=kept)
    empty = [part for part in parts if not masks[part].any()]
    if empty:
        raise ValueError(
            "the band leaves nothing "
            + (
                "inside it"
                if empty == ["in"]
                else "outside it" if empty == ["out"] else "at all"
            )
            + f" for {dataset.name}, so there is nothing to split"
        )

    before = list(edits_of(dataset) if before is None else before)
    pieces = {}
    for part in parts:
        piece = dataset.subset(masks[part], name=f"{dataset.name} {part}")
        piece.scene_edits = [dict(edit) for edit in before]
        record_edit(piece, {"kind": "split", **band, "part": part})
        piece.scene_edits.extend(dict(edit) for edit in after)
        pieces[part] = piece

    store = widget.dataset_store
    state = store.state_of(dataset)
    transform = None if state is None else state.transform

    with _view_kept(widget):
        # Added before the original goes: unloading the last dataset would
        # reset the whole session -- render range, grid, scalebar -- on the way.
        widget._apply_loaded_datasets(list(pieces.values()), merge=True)
        if transform is not None:
            from .scene_session import _channel_for, _sync_shift_spins

            for piece in pieces.values():
                store.set_transform(piece.dataset_id, transform)
                _sync_shift_spins(_channel_for(widget, piece), transform)
        widget.unload_dataset(dataset)
        widget.data_to_layer_itf.set_render_range_and_offset(recenter=False)
        widget.data_to_layer_itf.update_layers()
    return pieces


@contextmanager
def _view_kept(widget):
    """Put the camera back where it was: a split moves no data.

    Adding and unloading datasets re-centres the camera on the data, which is
    right when a file opens and wrong here, where the localizations stay
    exactly where the user was looking at them.
    """
    camera = widget.viewer.camera
    center, zoom, angles = tuple(camera.center), camera.zoom, tuple(camera.angles)
    cached = list(getattr(widget.data_to_layer_itf, "camera", []) or [])
    try:
        yield
    finally:
        camera.angles = angles
        camera.zoom = zoom
        camera.center = center
        if len(cached) == 3:
            widget.data_to_layer_itf.camera = [zoom, center, angles]
        widget.zoom = zoom
