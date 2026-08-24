"""Ownership of the loaded datasets, and typed events when that set changes.

Before this, "which datasets are loaded" was a bare list, and everything that
needed per-dataset state kept its own list beside it and trusted the indices to
stay aligned. They did not: unloading one dataset meant remembering to pop the
matching entry out of the renderer's three arrays, the filter list, the
adjustment list, the info panel and the namespace bookkeeping, in the right
order, from a method that had no way to know who else was listening.

Two things fix that, and this module is both:

* **Stable identity.** Every dataset gets a ``dataset_id`` that does not change
  when its neighbours come and go. Anything holding per-dataset state keys it by
  that id, so there is no index to keep aligned and nothing to renumber.

* **Typed events.** Interested parties subscribe rather than being called. A
  dataset closing emits :class:`DatasetClosed`, and whoever owns state for it
  releases it themselves — the code that unloads a dataset no longer has to
  enumerate its dependants.

Host-free, like the rest of ``napari_storm.core``: it holds datasets and emits
dataclasses, and knows nothing about layers, widgets or the viewer.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from .dataset_state import (AppearanceChanged, DatasetState, MaskChanged,
                            TransformChanged)

__all__ = [
    "DatasetStore",
    "DatasetOpened",
    "DatasetClosed",
    "StoreCleared",
    "BatchCommitted",
]


@dataclass(frozen=True)
class DatasetOpened:
    """A dataset joined the store and now has an id."""

    dataset_id: int


@dataclass(frozen=True)
class DatasetClosed:
    """A dataset left the store.  Release anything keyed by its id."""

    dataset_id: int


@dataclass(frozen=True)
class StoreCleared:
    """Every dataset left at once.

    Emitted *after* the individual :class:`DatasetClosed` events, so a listener
    can either handle them one by one or wait for this and reset wholesale.
    """


@dataclass(frozen=True)
class BatchCommitted:
    """One user gesture's worth of events, delivered together.

    Emitted at the end of :meth:`DatasetStore.batch`. Listeners that would
    otherwise do expensive work per event can ignore the individual ones and
    act once on this.
    """

    events: tuple = field(default_factory=tuple)


class _DatasetView(Sequence):
    """A live, read-only window onto the store's ordered datasets.

    Live because widgets capture this once and expect to keep seeing current
    contents; read-only because a dataset appended behind the store's back gets
    no id, and everything keyed by id would then be quietly wrong about what is
    loaded.
    """

    def __init__(self, datasets):
        self._datasets = datasets

    def __len__(self):
        return len(self._datasets)

    def __getitem__(self, index):
        return self._datasets[index]

    def __iter__(self):
        return iter(self._datasets)

    def __contains__(self, item):
        return item in self._datasets

    def __eq__(self, other):
        if isinstance(other, _DatasetView):
            return self._datasets == other._datasets
        if isinstance(other, (list, tuple)):
            return self._datasets == list(other)
        return NotImplemented

    def __repr__(self):
        return repr(self._datasets)


class DatasetStore:
    """Owns the loaded datasets, in order, each with a stable id.

    :attr:`datasets` is the same view object for the lifetime of the store, so
    code that captured a reference to it keeps seeing current contents -- and
    cannot change them, because a dataset appended behind the store's back
    would get no id and nothing keyed by id would know it exists.
    """

    def __init__(self):
        self._datasets = []
        self._view = _DatasetView(self._datasets)
        self._by_id = {}
        self._states = {}
        self._next_id = 1
        self._listeners = []
        self._batch = None

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    @property
    def datasets(self):
        """Live, read-only view of the ordered datasets."""
        return self._view

    @property
    def ids(self):
        """Dataset ids, in the same order as :attr:`datasets`."""
        return [dataset.dataset_id for dataset in self._datasets]

    def __len__(self):
        return len(self._datasets)

    def __iter__(self):
        return iter(self._datasets)

    def __contains__(self, dataset):
        return dataset in self._datasets

    def get(self, dataset_id):
        """The dataset with *dataset_id*, or None if it is not loaded."""
        return self._by_id.get(dataset_id)

    def index_of(self, dataset_id):
        """Current position of *dataset_id*, or -1.

        Positions still matter to the Qt widgets, which lay controls out in
        order. They are derived from identity here rather than being the
        identity.
        """
        dataset = self._by_id.get(dataset_id)
        return -1 if dataset is None else self._datasets.index(dataset)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add(self, dataset):
        """Take ownership of *dataset*, assign its id, and return that id.

        The id is written onto the dataset as ``dataset_id``: identity travels
        with the object, so no caller needs the store to look it up again.

        Adding the same object twice is refused. It used to give the dataset a
        second id, overwrite the first on the object, leave the first dangling
        in the lookup map and the dataset in the list twice -- after which
        removing it once left a half-open dataset behind.
        """
        if dataset in self._datasets:
            raise ValueError(
                f"dataset {getattr(dataset, 'name', dataset)!r} is already open "
                f"as id {dataset.dataset_id}"
            )
        dataset_id = self._next_id
        self._next_id += 1
        dataset.dataset_id = dataset_id
        self._datasets.append(dataset)
        self._by_id[dataset_id] = dataset
        self._states[dataset_id] = DatasetState(
            dataset_id=dataset_id, name=getattr(dataset, "name", "") or ""
        )
        self._emit(DatasetOpened(dataset_id))
        return dataset_id

    def remove(self, dataset_or_id):
        """Drop one dataset.  Returns its id, or None if it was not loaded."""
        dataset = (
            self._by_id.get(dataset_or_id)
            if isinstance(dataset_or_id, int)
            else dataset_or_id
        )
        if dataset is None or dataset not in self._datasets:
            return None
        dataset_id = dataset.dataset_id
        self._datasets.remove(dataset)
        self._by_id.pop(dataset_id, None)
        self._states.pop(dataset_id, None)
        self._emit(DatasetClosed(dataset_id))
        return dataset_id

    def clear(self):
        """Drop everything, emitting one DatasetClosed each, then StoreCleared."""
        for dataset in list(self._datasets):
            self.remove(dataset)
        self._emit(StoreCleared())

    # ------------------------------------------------------------------
    # Per-dataset state
    # ------------------------------------------------------------------

    def state(self, dataset_id):
        """The :class:`DatasetState` for *dataset_id*, or None."""
        return self._states.get(dataset_id)

    def state_of(self, dataset):
        return self._states.get(getattr(dataset, "dataset_id", None))

    def set_appearance(self, dataset_id, **fields):
        """Change how a dataset is displayed and announce it.

        Unspecified fields keep their value, so a control that owns one slider
        sends only what it moved.
        """
        state = self._states.get(dataset_id)
        if state is None:
            raise KeyError(f"dataset {dataset_id} is not loaded")
        appearance = state.with_appearance(**fields)
        self._emit(AppearanceChanged(dataset_id, appearance))
        return appearance

    def set_transform(self, dataset_id, transform):
        """Move a dataset in world space and announce it."""
        state = self._states.get(dataset_id)
        if state is None:
            raise KeyError(f"dataset {dataset_id} is not loaded")
        state.transform = transform
        self._emit(TransformChanged(dataset_id, transform))
        return transform

    def notify_mask_changed(self, dataset_id):
        """Announce that which localizations are drawn has changed.

        The mask itself lives on the table, which is host-free and has no
        business knowing about listeners; the store is where a change to it
        becomes something the application can react to.
        """
        if dataset_id in self._states:
            self._emit(MaskChanged(dataset_id))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def subscribe(self, listener):
        """Register ``listener(event)``.  Returns it, so it can be unsubscribed."""
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _emit(self, event):
        if self._batch is not None:
            self._batch.append(event)
        for listener in list(self._listeners):
            listener(event)

    @contextmanager
    def batch(self):
        """Group the events of one gesture and follow them with BatchCommitted.

        Individual events are still delivered as they happen -- a listener that
        must release a resource cannot be asked to wait -- but a listener that
        only needs to recompute something once can act on the batch instead.
        Nesting is flattened into the outermost batch.
        """
        if self._batch is not None:
            yield self
            return
        self._batch = []
        try:
            yield self
        finally:
            collected = tuple(self._batch)
            self._batch = None
            if collected:
                self._emit(BatchCommitted(collected))
