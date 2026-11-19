"""Cutting one dataset in two by its filters, each part a channel of its own.

The way in is :func:`split_by_filters`: apply filters, look at what they
leave, then split. What is on screen -- what every filter kept -- becomes
``<name> in``, and everything any filter removed ``<name> out``. Together the
two are the whole dataset. The original is replaced: its rows now live in the
parts.

Each part remembers how it was made. Its scene edit log is the original's,
filters and all, followed by the split, so a scene can open the file again,
redo the edits and the split, and end up with the same two channels.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .core import edits_of, record_edit, split_point

__all__ = [
    "SPLIT_PARTS",
    "split_by_filters",
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


def split_masks(widget, dataset, band=None):
    """``{"in": mask, "out": mask}`` over *dataset*'s canonical rows.

    Without *band*: what the filters keep, and what they removed. With one --
    how splits were recorded at first, and still read from scenes -- the
    rows the filters keep, divided by the band.
    """
    from .DataFilter import DataFilterInterface

    n = dataset.number_of_entries()
    kept = np.ones(n, dtype=bool)
    kept[widget.data_filter_itf.indices_for(dataset)] = False
    if band is None:
        return {"in": kept, "out": ~kept}
    if not dataset.table.has_field(band["parameter"]):
        raise ValueError(f"{dataset.name} has no parameter {band['parameter']!r}")
    removed_indices = DataFilterInterface.band_indices(dataset, **_band(band))
    if removed_indices is None:
        raise ValueError(f"unknown filter mode {band['mode']!r}")
    removed = np.zeros(n, dtype=bool)
    removed[removed_indices] = True
    return {"in": kept & ~removed, "out": kept & removed}


def split_by_filters(widget, dataset):
    """Split *dataset* into what its filters kept and everything they removed."""
    log = edits_of(dataset)
    if not any(edit.get("kind") == "filter" for edit in log[split_point(log) :]):
        raise ValueError(
            f"No filter has been applied to {dataset.name} yet. Apply one or "
            "more, then split: the parts are what you see and what the "
            "filters removed."
        )
    masks = split_masks(widget, dataset)
    if not masks["out"].any():
        raise ValueError(
            f"The filters removed nothing from {dataset.name}, so there is "
            "nothing to split off."
        )
    if not masks["in"].any():
        raise ValueError(
            f"The filters kept nothing of {dataset.name}, so there is nothing "
            "to split."
        )
    return split_dataset(widget, dataset)


def split_dataset(widget, dataset, band=None, parts=SPLIT_PARTS):
    """Replace *dataset* in *widget*'s session by its *parts*.

    Divided by the filters applied to it, or by *band* as well when given
    (see `split_masks`). Each part's edit log is the dataset's own followed
    by the split.

    Returns ``{part: new dataset}``. Raises ValueError, leaving the session
    as it was, when a part would be empty or the dataset cannot be split
    (drift correction applied, an unknown parameter or mode).
    """
    band = None if band is None else _band(band)
    masks = split_masks(widget, dataset, band)
    empty = [part for part in parts if not masks[part].any()]
    if empty:
        raise ValueError(
            f"the {' and '.join(empty)} part of {dataset.name} would be empty, "
            "so there is nothing to split"
        )

    before = [dict(edit) for edit in edits_of(dataset)]
    pieces = {}
    for part in parts:
        piece = dataset.subset(masks[part], name=f"{dataset.name} {part}")
        piece.scene_edits = [dict(edit) for edit in before]
        record_edit(piece, {"kind": "split", **(band or {}), "part": part})
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
