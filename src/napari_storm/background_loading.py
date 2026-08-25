"""Run file loading off the Qt main thread, with progress and cancellation.

Every operation in this plugin used to run on the Qt main thread: file read,
dataset construction, geometry expansion and buffer upload.  On the benchmark
fixtures that is seconds of a frozen window with no progress and no way out,
and a user cannot tell a freeze from a crash -- they report both as a crash
(§3.8 and P1-09 of ``docs/modernization-review.md``).

Two things make that awkward here, and both are handled in this module:

* **Importers ask questions.**  Several loaders pop a ``QInputDialog`` for a
  missing pixel size.  A dialog cannot be constructed off the GUI thread, so
  :func:`run_on_main_thread` marshals the call back and blocks the worker until
  the user answers.  Importers keep their existing shape.

* **A file read is not interruptible.**  ``h5py.File(...)`` and ``np.loadtxt``
  cannot be stopped part-way.  Cancellation is honoured at
  :meth:`LoadHandle.checkpoint` boundaries and, failing that, by discarding a
  result that arrives after the user cancelled.  The window stays responsive
  either way, which is the property the acceptance gate asks for; this module
  does not pretend to abort a syscall.

Note what stays on the GUI thread: building the billboard geometry and handing
it to napari.  That is a main-thread requirement of the renderer, not an
oversight, and it is the larger cost on big datasets -- see the benchmark's
``open`` column.  This module bounds the read, not the upload.
"""
from __future__ import annotations

import threading

from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from qtpy.QtWidgets import QApplication, QProgressDialog

__all__ = [
    "LoadCancelled",
    "LoadHandle",
    "on_main_thread",
    "run_on_main_thread",
    "load_in_background",
]


class LoadCancelled(Exception):
    """Raised inside a worker when the user cancelled at a checkpoint."""


#: Sentinel returned in place of a result when the load was cancelled.
_CANCELLED = object()


# --------------------------------------------------------------------------
# Marshalling calls back onto the GUI thread
# --------------------------------------------------------------------------


class _MainThreadInvoker(QObject):
    """Runs a callable on the thread this object lives on."""

    _requested = Signal(object)

    def __init__(self):
        super().__init__()
        self._requested.connect(self._run, Qt.BlockingQueuedConnection)

    # @Slot matters here, and is not decoration for its own sake.  Connecting a
    # signal to an undecorated Python callable makes PyQt wrap it in a
    # PyQtSlotProxy QObject whose thread affinity is fixed at connect time.
    # moveToThread() below moves this object but not that proxy, so the receiver
    # stayed on the worker thread and Qt refused the emit with
    # "Dead lock detected while activating a BlockingQueuedConnection".
    @Slot(object)
    def _run(self, request):
        function, args, kwargs, result = request
        try:
            result["value"] = function(*args, **kwargs)
        except BaseException as error:  # re-raised on the calling thread
            result["error"] = error

    def call(self, function, args, kwargs):
        result = {}
        self._requested.emit((function, args, kwargs, result))
        if "error" in result:
            raise result["error"]
        return result.get("value")


_invoker = None
_invoker_lock = threading.Lock()


def _main_thread_invoker():
    """Return the invoker, creating it if needed.  Always GUI-thread owned.

    Prefer calling this once from the GUI thread before starting a worker (see
    :func:`load_in_background`): creating it there means the object never has to
    be migrated at all.
    """
    global _invoker
    with _invoker_lock:
        app = QApplication.instance()
        if app is None:
            return None
        if _invoker is None:
            _invoker = _MainThreadInvoker()
            _invoker.moveToThread(app.thread())
        return _invoker


def on_main_thread():
    """True when the caller is already on the GUI thread (or there is no GUI)."""
    app = QApplication.instance()
    if app is None:
        return True
    return QThread.currentThread() is app.thread()


def run_on_main_thread(function, *args, **kwargs):
    """Call *function* on the GUI thread and return its result.

    A no-op indirection when already on the GUI thread, so importers can call it
    unconditionally.  From a worker it blocks until the GUI thread has run the
    call, which is what a modal question needs anyway.
    """
    if on_main_thread():
        return function(*args, **kwargs)
    invoker = _main_thread_invoker()
    if invoker is None:
        return function(*args, **kwargs)
    return invoker.call(function, args, kwargs)


# --------------------------------------------------------------------------
# Cancellation token
# --------------------------------------------------------------------------


class LoadHandle:
    """Cancellation token and progress sink handed to a loader."""

    def __init__(self, progress_sink=None):
        self._cancelled = threading.Event()
        self._progress_sink = progress_sink
        self.description = ""

    def set_progress_sink(self, sink):
        self._progress_sink = sink

    def cancel(self):
        self._cancelled.set()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def checkpoint(self, description=None):
        """Report progress and abort if the user has cancelled.

        Call between discrete steps -- per file, per channel, per chunk.  This
        is the only place a load can actually stop early; a cancellation during
        a single blocking read is detected once that read returns.
        """
        if description is not None:
            self.description = description
            if self._progress_sink is not None:
                self._progress_sink(description)
        if self.cancelled:
            raise LoadCancelled(description or "load cancelled")


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def _thread_worker():
    """napari's thread_worker, or None when it cannot be used here."""
    if QApplication.instance() is None:
        return None
    try:
        from napari.qt.threading import thread_worker
    except ImportError:
        return None
    return thread_worker


def load_in_background(
    function,
    *,
    on_result,
    on_error=None,
    on_cancelled=None,
    description="Loading…",
    parent=None,
    force_synchronous=False,
):
    """Run ``function(handle)`` off the GUI thread with a cancellable dialog.

    Falls back to running inline when there is no Qt application or no napari
    worker available, so headless callers exercise the same code rather than a
    second, untested path.  Returns the worker, or ``None`` when it ran inline.
    """
    handle = LoadHandle()

    def _guarded(load_handle):
        """Cancellation is an outcome, not an error, so keep it off that path."""
        try:
            return function(load_handle)
        except LoadCancelled:
            return _CANCELLED

    def _finish(result):
        if result is _CANCELLED or handle.cancelled:
            if on_cancelled is not None:
                on_cancelled()
            return
        on_result(result)

    def _failed(error):
        if on_error is None:
            raise error
        on_error(error)

    worker_factory = None if force_synchronous else _thread_worker()
    if worker_factory is not None:
        # Build the marshalling object here, while we are still on the GUI
        # thread, so a loader asking a question never has to migrate it.
        _main_thread_invoker()
    if worker_factory is None:
        try:
            _finish(_guarded(handle))
        except Exception as error:  # noqa: BLE001 - routed to the caller
            _failed(error)
        return None

    # An indeterminate dialog: the importers report which step they are on, not
    # a percentage, and inventing one would be a lie about work we cannot see.
    dialog = QProgressDialog(description, "Cancel", 0, 0, parent)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.canceled.connect(handle.cancel)
    # setLabelText touches a widget, so it has to happen on the GUI thread even
    # though checkpoint() is called from the worker.
    handle.set_progress_sink(
        lambda text: run_on_main_thread(dialog.setLabelText, text)
    )

    # QProgressDialog only auto-shows from setValue(), which an indeterminate
    # dialog never calls.  Show it on a timer instead so a fast load does not
    # flash a dialog at the user.
    reveal = QTimer()
    reveal.setSingleShot(True)
    reveal.timeout.connect(dialog.show)
    reveal.start(400)

    def _close_dialog():
        reveal.stop()
        dialog.close()
        dialog.deleteLater()

    # The callbacks go in through `connect` rather than worker.errored.connect()
    # afterwards: superqt attaches its own re-raising handler to `errored` unless
    # a handler is supplied here, and that handler turns any load failure into an
    # exception in the Qt event loop on top of our own reporting.
    worker = worker_factory(
        _guarded,
        start_thread=False,
        connect={
            "returned": _finish,
            "errored": _failed,
            "finished": _close_dialog,
        },
    )(handle)
    # The handle is the programmatic equivalent of the dialog's Cancel button.
    worker.load_handle = handle
    # Keep the dialog and its timer alive for as long as the worker is.
    worker._napari_storm_progress = (dialog, reveal)
    worker.start()
    return worker
