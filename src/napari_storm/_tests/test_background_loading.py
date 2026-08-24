"""File loading runs off the Qt main thread, reports progress, and cancels.

Covers P1-09 of docs/modernization-review.md.  The point of the gate is not
that a read is abortable mid-syscall -- it is not -- but that the window stays
responsive, the user is told what is happening, and cancelling leaves the
session exactly as it was.
"""
import threading
import time

import h5py
import numpy as np
import pytest
from qtpy.QtWidgets import (QApplication, QFileDialog, QProgressDialog,
                            QPushButton)

from napari_storm._dock_widget import napari_storm
from napari_storm.background_loading import (LoadCancelled, LoadHandle,
                                             load_in_background,
                                             on_main_thread,
                                             run_on_main_thread)


@pytest.fixture
def ns_file(tmp_path):
    """A minimal .ns dataset the known-filetype importer can read."""
    path = tmp_path / "background.ns"
    locs = np.rec.array(
        np.zeros(8, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
    )
    locs.x_pos_nm = np.linspace(10_000, 12_000, 8)
    locs.y_pos_nm = np.linspace(40_000, 50_000, 8)
    with h5py.File(path, "w") as file:
        stored = file.create_dataset("dataset", data=locs)
        stored.attrs["name"] = "background"
        stored.attrs["zdim_present"] = False
        stored.attrs["dataset_class"] = "LocalizationDataBaseClass"
    return str(path)


def _pump_until(predicate, timeout=15.0):
    """Run the Qt event loop until *predicate* holds, or fail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------- cancellation


def test_checkpoint_reports_progress_and_passes_through():
    seen = []
    handle = LoadHandle(progress_sink=seen.append)
    handle.checkpoint("reading")
    assert seen == ["reading"]
    assert handle.description == "reading"


def test_checkpoint_raises_after_cancel():
    handle = LoadHandle()
    handle.checkpoint("reading")
    handle.cancel()
    assert handle.cancelled
    with pytest.raises(LoadCancelled):
        handle.checkpoint("still reading")


# ----------------------------------------------------------- main-thread hop


def test_run_on_main_thread_is_a_no_op_on_the_main_thread(qtbot):
    assert on_main_thread()
    assert run_on_main_thread(lambda value: value * 2, 21) == 42


def test_run_on_main_thread_marshals_from_a_worker(qtbot):
    """A dialog cannot be built off the GUI thread; the call has to come back."""
    result = {}

    def worker():
        result["thread_was_main"] = on_main_thread()
        result["value"] = run_on_main_thread(
            lambda: ("ran on main" if on_main_thread() else "ran off main")
        )

    thread = threading.Thread(target=worker)
    thread.start()
    assert _pump_until(lambda: "value" in result and not thread.is_alive())
    thread.join()

    assert result["thread_was_main"] is False
    assert result["value"] == "ran on main"


def test_run_on_main_thread_reraises_on_the_caller(qtbot):
    errors = []

    def boom():
        raise RuntimeError("from the GUI thread")

    def worker():
        try:
            run_on_main_thread(boom)
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    assert _pump_until(lambda: errors and not thread.is_alive())
    thread.join()
    assert errors == ["from the GUI thread"]


# ------------------------------------------------------------- worker plumbing


def test_synchronous_fallback_delivers_the_result():
    results = []
    worker = load_in_background(
        lambda handle: "loaded",
        on_result=results.append,
        force_synchronous=True,
    )
    assert worker is None
    assert results == ["loaded"]


def test_synchronous_fallback_routes_errors():
    errors = []

    def boom(handle):
        raise ValueError("bad file")

    load_in_background(
        boom,
        on_result=lambda result: pytest.fail("should not have produced a result"),
        on_error=errors.append,
        force_synchronous=True,
    )
    assert [str(error) for error in errors] == ["bad file"]


def test_a_cancelled_load_reports_cancellation_not_a_result():
    events = []

    def cancel_partway(handle):
        handle.checkpoint("step one")
        handle.cancel()
        handle.checkpoint("step two")
        return "should never be delivered"

    load_in_background(
        cancel_partway,
        on_result=lambda result: events.append(("result", result)),
        on_cancelled=lambda: events.append(("cancelled", None)),
        force_synchronous=True,
    )
    assert events == [("cancelled", None)]


# --------------------------------------------------------------- integration


def test_background_open_loads_a_dataset_and_builds_a_layer(
    make_napari_viewer, ns_file
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    assert widget.open_localization_data_file_and_get_dataset(
        file_path=ns_file, background=True
    )
    assert _pump_until(lambda: widget.n_datasets == 1)

    dataset = widget.localization_datasets[0]
    assert dataset.number_of_entries() == 8
    assert widget.data_to_layer_itf.layer_for(dataset) in viewer.layers


def test_background_open_does_not_block_the_main_thread(
    make_napari_viewer, ns_file, monkeypatch
):
    """The GUI thread must keep processing events while the file is read."""
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)

    started = threading.Event()
    release = threading.Event()
    original = widget.file_to_data_itf.load_ns

    def slow_load_ns(file_path):
        started.set()
        release.wait(10)
        return original(file_path)

    monkeypatch.setattr(widget.file_to_data_itf, "load_ns", slow_load_ns)
    widget.open_localization_data_file_and_get_dataset(
        file_path=ns_file, background=True
    )

    # The read is parked inside the worker; the main thread is still ours.
    assert _pump_until(started.is_set)
    assert widget.n_datasets == 0
    release.set()
    assert _pump_until(lambda: widget.n_datasets == 1)


def test_cancelling_a_background_open_leaves_the_session_untouched(
    make_napari_viewer, ns_file, monkeypatch
):
    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda: ("", ""))
    widget.open_localization_data_file_and_get_dataset(file_path=ns_file)
    assert widget.n_datasets == 1
    keep = widget.localization_datasets[0]

    started = threading.Event()
    release = threading.Event()
    original = widget.file_to_data_itf.load_ns

    def slow_load_ns(file_path):
        started.set()
        release.wait(10)
        return original(file_path)

    monkeypatch.setattr(widget.file_to_data_itf, "load_ns", slow_load_ns)
    worker = widget.open_localization_data_file_and_get_dataset(
        file_path=ns_file, background=True
    )
    try:
        assert _pump_until(started.is_set)

        # Press the dialog's own Cancel button, so what gets exercised is the
        # affordance the user actually has.  QProgressDialog.cancel() would not
        # do: it resets the dialog without emitting canceled().
        dialogs = widget.findChildren(QProgressDialog)
        assert len(dialogs) == 1
        cancel_button = dialogs[0].findChild(QPushButton)
        assert cancel_button is not None
        cancel_button.click()
        assert worker.load_handle.cancelled
    finally:
        # Let the parked read finish however the assertions above went, so a
        # failure here does not leave a thread running into pytest's teardown.
        release.set()

    # The result arrives after the cancel, and must be discarded rather than
    # applied: a read cannot be aborted mid-syscall, only disowned.
    assert _pump_until(lambda: not worker.is_running)
    QApplication.processEvents()

    assert widget.localization_datasets == [keep]
    assert widget.n_datasets == 1
    assert widget.file_to_data_itf.dataset_names == ["background"]


def test_the_reader_hook_path_stays_synchronous(make_napari_viewer, ns_file):
    """napari's reader contract has no callback to defer a load into."""
    widget = napari_storm(napari_viewer=make_napari_viewer())
    assert widget.open_localization_data_file_and_get_dataset(file_path=ns_file) is True
    assert widget.n_datasets == 1
