"""Stable dataset identity and typed lifecycle events, tested headlessly.

Level 2 of docs/modernization-review.md, and §4.2's rule that renderer
resources are keyed by stable dataset IDs rather than by positions in parallel
lists. The point of the store is that unloading a dataset cannot silently
misalign somebody else's bookkeeping.
"""
import pytest

from napari_storm.core import (BatchCommitted, DatasetClosed, DatasetOpened,
                               DatasetStore, StoreCleared)


class _Dataset:
    """Anything with attributes; the store does not care what a dataset is."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


def _store_with(*names):
    store = DatasetStore()
    datasets = [_Dataset(name) for name in names]
    for dataset in datasets:
        store.add(dataset)
    return store, datasets


# ------------------------------------------------------------------- identity


def test_add_assigns_an_id_and_records_it_on_the_dataset():
    store, (first, second) = _store_with("a", "b")
    assert first.dataset_id != second.dataset_id
    assert store.get(first.dataset_id) is first
    assert store.ids == [first.dataset_id, second.dataset_id]


def test_ids_are_not_reused_after_a_close():
    """A recycled id would silently attach stale state to a new dataset."""
    store, (first,) = _store_with("a")
    retired = first.dataset_id
    store.remove(first)

    replacement = _Dataset("b")
    store.add(replacement)
    assert replacement.dataset_id != retired
    assert store.get(retired) is None


def test_ids_survive_their_neighbours_being_removed():
    store, (first, second, third) = _store_with("a", "b", "c")
    ids_before = (first.dataset_id, third.dataset_id)

    store.remove(second)

    assert (first.dataset_id, third.dataset_id) == ids_before
    assert store.datasets == [first, third]


def test_positions_are_derived_from_identity_not_the_other_way_round():
    store, (first, second, third) = _store_with("a", "b", "c")
    store.remove(first)
    assert store.index_of(second.dataset_id) == 0
    assert store.index_of(third.dataset_id) == 1
    assert store.index_of(999) == -1


def test_the_dataset_list_is_a_live_view():
    """Widgets capture this list once and expect to keep seeing the truth."""
    store, (first,) = _store_with("a")
    captured = store.datasets

    second = _Dataset("b")
    store.add(second)
    assert captured == [first, second]

    store.remove(first)
    assert captured == [second]


def test_removing_something_absent_is_a_no_op():
    store, (first,) = _store_with("a")
    assert store.remove(_Dataset("stranger")) is None
    assert store.remove(404) is None
    assert store.datasets == [first]


def test_remove_accepts_an_id_or_the_dataset():
    store, (first, second) = _store_with("a", "b")
    assert store.remove(first.dataset_id) == first.dataset_id
    assert store.remove(second) == second.dataset_id
    assert len(store) == 0


# --------------------------------------------------------------------- events


def test_opening_and_closing_emit_typed_events():
    store = DatasetStore()
    seen = []
    store.subscribe(seen.append)

    dataset = _Dataset("a")
    store.add(dataset)
    store.remove(dataset)

    assert seen == [
        DatasetOpened(dataset.dataset_id),
        DatasetClosed(dataset.dataset_id),
    ]


def test_clear_closes_each_dataset_before_announcing_the_reset():
    """A listener releasing per-id resources needs the individual events."""
    store, datasets = _store_with("a", "b")
    seen = []
    store.subscribe(seen.append)

    store.clear()

    assert seen == [
        DatasetClosed(datasets[0].dataset_id),
        DatasetClosed(datasets[1].dataset_id),
        StoreCleared(),
    ]
    assert store.datasets == []


def test_unsubscribing_stops_delivery():
    store = DatasetStore()
    seen = []
    listener = store.subscribe(seen.append)
    store.unsubscribe(listener)
    store.add(_Dataset("a"))
    assert seen == []


def test_a_listener_added_during_delivery_does_not_receive_that_event():
    """Iterating a copy keeps a mid-delivery subscribe from being surprising."""
    store = DatasetStore()
    seen = []

    def subscribe_more(event):
        store.subscribe(seen.append)

    store.subscribe(subscribe_more)
    store.add(_Dataset("a"))
    assert seen == []
    store.add(_Dataset("b"))
    assert len(seen) == 1


def test_a_batch_follows_its_events_with_one_summary():
    store = DatasetStore()
    seen = []
    store.subscribe(seen.append)

    with store.batch():
        first = _Dataset("a")
        second = _Dataset("b")
        store.add(first)
        store.add(second)

    assert seen[:2] == [
        DatasetOpened(first.dataset_id),
        DatasetOpened(second.dataset_id),
    ]
    assert seen[2] == BatchCommitted(
        (DatasetOpened(first.dataset_id), DatasetOpened(second.dataset_id))
    )


def test_an_empty_batch_says_nothing():
    store = DatasetStore()
    seen = []
    store.subscribe(seen.append)
    with store.batch():
        pass
    assert seen == []


def test_nested_batches_commit_once():
    store = DatasetStore()
    seen = []
    store.subscribe(seen.append)

    with store.batch():
        store.add(_Dataset("a"))
        with store.batch():
            store.add(_Dataset("b"))

    assert sum(isinstance(event, BatchCommitted) for event in seen) == 1


def test_a_batch_commits_even_when_the_gesture_fails():
    store = DatasetStore()
    seen = []
    store.subscribe(seen.append)

    with pytest.raises(RuntimeError):
        with store.batch():
            store.add(_Dataset("a"))
            raise RuntimeError("import failed part-way")

    assert isinstance(seen[-1], BatchCommitted)
