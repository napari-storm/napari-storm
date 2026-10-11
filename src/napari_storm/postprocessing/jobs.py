"""One worker, one latest pending request; callbacks are polled by the UI."""

import gc
from threading import Event, Lock, Thread


class Cancelled(Exception):
    pass


class JobRunner:
    def __init__(self):
        # Daemon threads, not an executor: Python joins executor threads at exit,
        # so quitting napari would wait for a pair search that cannot be
        # interrupted. A daemon thread is simply dropped.
        self._future = None
        self._pending = None
        self._cancel = Event()
        self._lock = Lock()
        self._progress = None
        self.generation = 0
        self.closed = False

    @property
    def busy(self):
        return self._future is not None or self._pending is not None

    def submit(self, operation):
        if self.closed:
            raise RuntimeError("Worker is closed")
        self.generation += 1
        self._cancel.set()
        self._pending = (self.generation, operation)
        if self._future is None:
            self._start()
        return self.generation

    def _start(self):
        generation, operation = self._pending
        self._pending = None
        token = Event()
        self._cancel = token

        def progress(stage, info=None):
            if token.is_set():
                raise Cancelled()
            with self._lock:
                self._progress = (stage, info or {})

        def work():
            try:
                progress("starting")
                value = operation(progress)
                progress("finished")
                return generation, value, None
            except Exception as error:
                # Return text, not an exception/traceback retaining large arrays.
                message = (
                    "Cancelled"
                    if isinstance(error, Cancelled)
                    else (str(error) or type(error).__name__)
                )
                del error
                gc.collect()
                return generation, None, message

        future = _Future()

        def target():
            future.set(work())

        self._future = future
        Thread(target=target, name="storm-postprocessing", daemon=True).start()

    def poll(self):
        with self._lock:
            progress = self._progress
            self._progress = None
        result = None
        if self._future is not None and self._future.done():
            result = self._future.result()
            self._future = None
            if result[0] != self.generation:
                result = None
            if self._pending is not None:
                self._start()
        return progress, result

    def cancel(self):
        self._pending = None
        self._cancel.set()

    def close(self):
        self.closed = True
        self.generation += 1
        self.cancel()


class _Future:
    """The one result a worker thread hands to the polling UI."""

    def __init__(self):
        self._done = Event()
        self._value = None

    def set(self, value):
        self._value = value
        self._done.set()

    def done(self):
        return self._done.is_set()

    def result(self):
        return self._value
