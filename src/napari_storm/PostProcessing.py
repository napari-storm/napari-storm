"""Post-processing jobs and reversible dataset state, owned by the dock."""

import hashlib
import logging
import math
import time
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QFileDialog, QInputDialog, QListWidgetItem, QMessageBox

from .core.dataset_state import DriftView
from .core.dataset_store import DatasetClosed, DatasetOpened, StoreCleared
from .core.traces import find_trace_column
from .postprocessing.comet_runner import CometParameters, prepare_input, run_comet, subsample
from .postprocessing.drift import DriftModel
from .postprocessing.jobs import JobRunner
from .postprocessing.pair_budget import estimate_memory
from .pyqt.postprocessing_tab import PostProcessingWindow

_log = logging.getLogger(__name__)
OVER_BUDGET = "over the comfortable memory budget"


def run_schedule(max_drift_nm, target_sigma_nm):
    """(σ0, expected steps E, most steps K) of a COMET run, as §5.4 states them.

    COMET divides σ by 1.5 per accepted step, keeps refining below the target
    while its updates shrink, and stops at 1 nm at the latest.
    """
    sigma0 = max_drift_nm // 3
    expected = max(1, math.ceil(math.log(sigma0 / target_sigma_nm) / math.log(1.5)) + 1)
    most = math.ceil(math.log(max(sigma0, 1.0)) / math.log(1.5)) + 1
    return sigma0, expected, max(most, expected)


class RunProgress:
    """COMET's stages turned into a fraction, a sentence and an ETA."""

    STAGES = {
        "preparing": "Preparing…",
        "counting pairs": "Counting pairs…",
        "memory estimate": "Checking memory…",
        "loading COMET": "Loading COMET…",
        "segmentation": "Windows…",
        "pairs_start": "Pair search…",
        "pairs_done": "Compiling kernels (first run only)…",
        "interpolation": "Interpolating…",
        "apply": "Applying…",
    }

    def __init__(self, params):
        self.sigma0, self.expected, self.most = run_schedule(
            params.max_drift_nm, params.target_sigma_nm
        )
        self.started = time.monotonic()
        self.first_run_started = None
        self.run = 0
        self.sigma = self.sigma0
        self.fraction = 0.0
        self.text = "Starting…"

    def update(self, stage, info):
        if stage in ("run_start", "evaluation", "run_end"):
            self.run = int(info.get("run", self.run))
            self.sigma = float(info.get("sigma_nm", self.sigma))
            if self.first_run_started is None:
                self.first_run_started = time.monotonic()
            done = max(0, self.run - 1) + (
                1.0 if stage == "run_end" else 0.5 if stage == "evaluation" else 0.0
            )
            self.fraction = 0.1 + 0.85 * min(1.0, done / self.expected)
            text = f"Step {self.run} of ~{self.expected} (≤ {self.most}) · σ {self.sigma:.1f} nm"
            if self.run > 1:
                per_step = (time.monotonic() - self.first_run_started) / (self.run - 1)
                remaining = max(0.0, (self.expected - self.run + 1) * per_step)
                text += f" · ETA {remaining:.0f} s"
            self.text = text
        elif stage in self.STAGES:
            self.text = self.STAGES[stage]
            if stage == "pairs_done" and info.get("n_pairs") is not None:
                self.text = f"{info['n_pairs']:,} pairs · " + self.text
            self.fraction = {"interpolation": 0.96, "apply": 0.98}.get(
                stage, min(self.fraction, 0.1)
            )
        return self.fraction, self.text


def _digest(*arrays):
    """A short fingerprint of arrays' contents (shape and bytes)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


class PostProcessingInterface:
    def __init__(self, parent):
        self.parent = parent
        self.store = parent.dataset_store
        self.widget = PostProcessingWindow(parent)
        self.runner = JobRunner()
        self.pair_runner = JobRunner()
        self.pair_sets = {}
        self.pair_keys = {}
        self.candidates = {}
        self._fiducial_layers = {}
        self._job = None
        self._lock = None
        self._pair_job = None
        self._time_scale = None
        self._display_times = {}
        self._camera = None
        self._plot = None
        self._playing = False
        self._previous_dataset = None
        self._closed = False
        self.store.subscribe(self.on_store_event)
        self.widget.dataset.currentIndexChanged.connect(self._selection_changed)
        for name, method in {
            "detect": self.detect,
            "exclude": self.exclude,
            "restore": self.restore_fiducials,
            "group": self.group,
            "estimate": self.estimate,
            "run": self.run,
            "cancel": self.cancel,
            "undo": lambda: self.transition("undo"),
            "reapply": lambda: self.transition("reapply"),
            "discard": lambda: self.transition("discard"),
            "save": self.save,
            "load": self.load,
            "transfer": self.transfer,
            "export": self.export,
            "play": self.play,
        }.items():
            self.widget.actions[name].clicked.connect(
                lambda checked=False, f=method: self.call(f)
            )
        self.widget.candidates.itemChanged.connect(self._candidate_changed)
        self.widget.fraction.valueChanged.connect(self.preview)
        self.widget.pairs.toggled.connect(self.toggle_pairs)
        self.widget.inputs["radius"].valueChanged.connect(
            lambda _: self.invalidate_pairs()
        )
        self.widget.pair_mode.currentIndexChanged.connect(
            lambda _: self.invalidate_pairs()
        )
        self.widget.time_view.toggled.connect(
            lambda on: self.call(lambda: self.time_view(on))
        )
        self.timer = QTimer(parent)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.play_timer = QTimer(parent)
        self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self._play_step)
        self.preview_timer = QTimer(parent)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._apply_preview)
        renderer = self.parent.data_to_layer_itf.renderer
        if hasattr(renderer, "_pairs"):
            renderer._pairs.on_traces_removed_by_host = self._pairs_removed
        self.sync()

    def _pairs_removed(self, dataset_id):
        self.store.set_appearance(dataset_id, pairs=False)
        if self.dataset is not None and self.dataset.dataset_id == dataset_id:
            self.widget.pairs.blockSignals(True)
            self.widget.pairs.setChecked(False)
            self.widget.pairs.blockSignals(False)

    @property
    def dataset(self):
        return self.store.get(self.widget.dataset.currentData())

    def call(self, operation):
        try:
            operation()
        except Exception as error:
            _log.exception("post-processing action failed")
            self.widget.status.setText(str(error) or type(error).__name__)

    def on_store_event(self, event):
        if isinstance(event, (DatasetOpened, DatasetClosed, StoreCleared)):
            if self._time_scale is not None:
                self._leave_time_view()
            self.play_timer.stop()
            self._playing = False
            if isinstance(event, (DatasetClosed, StoreCleared)):
                if self._job and self.store.get(self._job[0]) is not self._job[1]:
                    self.cancel()
                if (
                    self._pair_job
                    and self.store.get(self._pair_job[0]) is not self._pair_job[1]
                ):
                    self.pair_runner.cancel()
            selected = self.widget.dataset.currentData()
            self.widget.dataset.blockSignals(True)
            self.widget.dataset.clear()
            for dataset in self.store:
                self.widget.dataset.addItem(dataset.name, dataset.dataset_id)
            index = self.widget.dataset.findData(selected)
            self.widget.dataset.setCurrentIndex(max(0, index))
            self.widget.dataset.blockSignals(False)
            for key in list(self.pair_sets):
                if self.store.get(key) is None:
                    self.pair_sets.pop(key, None)
                    self.pair_keys.pop(key, None)
                    self.candidates.pop(key, None)
            for key in list(self._fiducial_layers):
                if self.store.get(key) is None:
                    layer = self._fiducial_layers.pop(key)
                    if layer in self.parent.viewer.layers:
                        self.parent.viewer.layers.remove(layer)
            self.sync()

    def _selection_changed(self, *args):
        self.play_timer.stop()
        self._playing = False
        self.sync()

    def _candidate_changed(self, item):
        d = self.dataset
        if d is not None:
            row = self.widget.candidates.row(item)
            candidates = self.candidates.get(d.dataset_id, [])
            if 0 <= row < len(candidates):
                candidates[row]["selected"] = item.checkState() == Qt.CheckState.Checked

    def sync(self, *args):
        d = self.dataset
        busy = self.runner.busy or self._playing
        self.widget.dataset.setEnabled(not busy)
        state = None if d is None else self.store.state_of(d).drift
        available = find_spec("comet") is not None
        for key, button in self.widget.actions.items():
            button.setEnabled(d is not None and not busy)
        for key in ("run", "detect"):
            self.widget.actions[key].setEnabled(
                d is not None and not busy and available
            )
        for key in ("discard", "save", "reapply"):
            self.widget.actions[key].setEnabled(state is not None and not busy)
        # §4.2: in the Undone state only Re-apply, Discard (and Run) apply
        for key in ("transfer", "play", "export"):
            self.widget.actions[key].setEnabled(
                state is not None and state.applied and not busy
            )
        self.widget.actions["undo"].setEnabled(
            state is not None and state.applied and not busy
        )
        self.widget.actions["reapply"].setEnabled(
            state is not None and not state.applied and not busy
        )
        self.widget.actions["cancel"].setEnabled(busy)
        self.widget.fraction.setEnabled(
            state is not None and state.applied and not busy
        )
        self.widget.pairs.setEnabled(state is not None and state.applied and not busy)
        self.widget.time_view.setEnabled(
            d is not None and not d.zdim_present and not busy
        )
        self.widget.input_mode.setEnabled(
            d is not None and not self.is_minflux(d) and not busy
        )
        self.widget.actions["group"].setEnabled(
            d is not None and not self.is_minflux(d) and not busy
        )
        if d is not None and d.dataset_id != self._previous_dataset:
            self._previous_dataset = d.dataset_id
            self.widget.pairs.blockSignals(True)
            self.widget.pairs.setChecked(bool(self.store.state_of(d).appearance.pairs))
            self.widget.pairs.blockSignals(False)
            minflux = self.is_minflux(d)
            self.widget.inputs["window"].setValue(50 if minflux else 60)
            self.widget.inputs["max_drift"].setValue(100 if minflux else 300)
            # The plot and status describe one dataset; switching shows that one's.
            if state is not None:
                self.draw_plot(state.model, d)
            elif self._plot is not None:
                self._plot.figure.clear()
                self._plot.draw_idle()
            self.widget.status.setText(getattr(d, "postprocessing_status", "No drift"))
            self.widget.pair_status.setText(
                "At most 200,000 vectors. Pairs are selected on corrected positions; this is not independent validation."
            )
            precision = 10.0
            if d.table.has_sigma_axis("x"):
                sigmas = d.table.sigma_nm("x")[d.table.filtered_ids]
                good = np.isfinite(sigmas) & (sigmas > 0)
                if good.any():
                    precision = float(np.median(sigmas[good]))
            self.widget.inputs["radius"].setValue(
                precision * (4 if d.zdim_present else 3.5)
            )
        self.widget.candidates.blockSignals(True)
        self.widget.candidates.clear()
        if d is not None:
            for candidate in self.candidates.get(d.dataset_id, []):
                x, y = candidate["center_nm"]
                item = QListWidgetItem(
                    f"({x/1000:.2f}, {y/1000:.2f}) µm · radius {candidate['radius_p99_nm']:.1f} nm"
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if candidate.get("selected", False)
                    else Qt.CheckState.Unchecked
                )
                self.widget.candidates.addItem(item)
        self.widget.candidates.blockSignals(False)
        self.widget.fraction.blockSignals(True)
        self.widget.fraction.setValue(round(state.alpha * 10) if state else 10)
        self.widget.fraction.blockSignals(False)
        self.widget.comet_note.setText(
            ""
            if available
            else 'Install "napari-storm[comet]" to run COMET or detect fiducials. '
            "Load / Save drift, Undo, Grouping and Explore work without it."
        )

    @staticmethod
    def is_minflux(d):
        return "minflux" in str(d.dataset_type).lower() or (
            d.table.has_field("trace_id") and not d.table.has_field("frame_number")
        )

    def clock(self, d):
        table = d.table
        if not self.is_minflux(d):
            name = next(
                (n for n in ("frame_number", "frame") if table.has_field(n)), None
            )
            if name is None:
                raise ValueError("This dataset has no acquisition frames")
            times = np.asarray(table.column(name), dtype=float)
            if not np.isfinite(times).all() or np.any(times != np.floor(times)):
                raise ValueError("Acquisition frames must be finite integers")
            return times, "frame", None, None
        trace_name = find_trace_column(table)
        if trace_name is None:
            raise ValueError("MINFLUX correction requires trace ids")
        traces = table.column(trace_name)
        if np.any(traces < 0):
            raise ValueError("MINFLUX correction needs a valid trace id for every row")
        _, by_id = np.unique(traces, return_inverse=True)
        rank_times = None
        ranks = by_id
        if table.has_field("time_s"):
            times = table.column("time_s").astype(float)
            means = np.bincount(by_id, weights=times) / np.bincount(by_id)
            # D12: traces in time order -- the order of measurement -- whatever
            # their ids. Ties keep id order.
            order = np.argsort(means, kind="stable")
            position = np.empty_like(order)
            position[order] = np.arange(len(order))
            ranks = position[by_id]
            means = means[order]
            if np.isfinite(means).all() and np.all(np.diff(means) > 0):
                rank_times = means
        return ranks.astype(float), "trace", rank_times, traces

    @staticmethod
    def coords(d, *, raw=False):
        columns = []
        for axis in ("x", "y", "z"):
            if axis == "z" and not d.zdim_present:
                columns.append(np.zeros(len(d.table)))
            else:
                columns.append(
                    d.table.raw_coordinate_nm(axis)
                    if raw
                    else d.table.coordinate_nm(axis)
                )
        return np.column_stack(columns).astype(float)

    def parameters(self):
        w = self.widget.inputs
        return CometParameters(
            window=w["window"].value(),
            max_drift_nm=w["max_drift"].value(),
            keep_percent=self.widget.keep.value(),
        )

    def weights(self, d, ids):
        result = np.ones((len(ids), 3))
        for i, axis in enumerate(("x", "y", "z")):
            if d.table.has_sigma_axis(axis):
                sigma = d.table.sigma_nm(axis)[ids]
                valid = np.isfinite(sigma) & (sigma > 0)
                result[:, i] = (
                    1
                    / np.where(
                        valid, sigma, np.median(sigma[valid]) if valid.any() else 1.0
                    )
                    ** 2
                )
            elif d.table.has_photons():
                photons = d.table.photons()[ids]
                result[:, i] = np.where(
                    np.isfinite(photons) & (photons > 0), photons, 1.0
                )
        return result

    def input(self, d, params):
        times, kind, rank_times, traces = self.clock(d)
        ids = d.table.filtered_ids
        if not len(ids):
            raise ValueError("No selected localizations")
        coords = self.coords(d, raw=True)[ids]
        selected_times = times[ids]
        time_origin = float(selected_times.min())
        weights = self.weights(d, ids)
        if kind == "trace":
            _, inv = np.unique(selected_times, return_inverse=True)
            coords = np.column_stack(
                [
                    np.bincount(inv, weights=coords[:, i] * weights[:, i])
                    / np.bincount(inv, weights=weights[:, i])
                    for i in range(3)
                ]
            )
            selected_times = np.unique(selected_times)
        elif self.widget.input_mode.currentIndex() == 1:
            if not d.table.has_field("group_id"):
                raise ValueError("Group the localizations first")
            from .postprocessing.grouping import group_means

            coords, selected_times = group_means(
                coords,
                selected_times,
                d.table.column("group_id")[ids],
                window=None,
                weights=weights,
            )
        return coords, selected_times, times, kind, rank_times, time_origin

    def start(self, operation, callback):
        if self.runner.busy:
            raise ValueError("Wait for the current job")
        if self.pair_runner.busy:
            self.pair_runner.cancel()
            raise ValueError("Stopping the pair search; retry when it has finished")
        d = self.dataset
        if d is None:
            return
        self.protect_clock(d)
        self._job = (d.dataset_id, d, callback)
        self.runner.submit(operation)
        # Locked only once the job exists, so a failing submit cannot leave the
        # dataset locked for the session.
        self._lock = d.table.position_write_lock()
        self._lock.__enter__()
        self.widget.status.setText("Starting…")
        self.sync()

    def run(self):
        d, params = self.dataset, self.parameters()
        coords, times, row_times, kind, rank_times, time_origin = self.input(d, params)
        fingerprint = self.fingerprint(d, kind, rank_times)
        in_group = self.store.state_of(d).drift is not None and len(self.members(d)) > 1
        allow = getattr(self, "_allow_over_budget", None) == (d.dataset_id, params)
        self._allow_over_budget = None

        def done(model):
            model.provenance.update(fingerprint)
            self.apply(d, model, row_times)
            if in_group:
                self.widget.status.setText(
                    self.widget.status.text() + " · this dataset left its drift group"
                )

        self.start(
            lambda p: run_comet(
                coords,
                times,
                params,
                progress=p,
                clock=kind,
                rank_times=rank_times,
                total_rows=len(d.table),
                time_origin=time_origin,
                allow_over_budget=allow,
            ),
            done,
        )
        self._run_request = (d.dataset_id, params)
        self._run_progress = RunProgress(params)

    def fingerprint(self, d, kind, rank_times):
        """What a drift was estimated from, to recognise the dataset again (§4.1)."""
        times = self.clock(d)[0]
        result = {
            "rows": int(len(d.table)),
            "time_range": [float(np.min(times)), float(np.max(times))],
            "raw_positions": _digest(self.coords(d, raw=True)),
        }
        if kind == "trace":
            result["trace_membership"] = _digest(
                d.table.column(find_trace_column(d.table))
            )
            result["rank_times"] = None if rank_times is None else _digest(rank_times)
        return result

    def estimate(self):
        d, params = self.dataset, self.parameters()
        coords, times, _, _, _, time_origin = self.input(d, params)

        def work(progress):
            data = subsample(prepare_input(coords, times, params, time_origin=time_origin), params)
            progress("counting pairs", {})
            estimate = estimate_memory(
                data[:, :3],
                params.max_drift_nm,
                total_rows=len(d.table),
                frame_count=int(data[:, 3].max()) + 1,
            )
            return estimate

        def done(estimate):
            if not estimate.fits and estimate.pairs:
                # pairs fall with the square of the share kept: suggest one that fits
                room = estimate.physical_bytes - estimate.resident_bytes - estimate.reserve_bytes
                share = np.sqrt(max(room, 0) / max(estimate.additional_bytes, 1))
                suggestion = int(max(1, min(99, np.floor(params.keep_percent * share))))
                self.widget.keep.setValue(suggestion)
                self.widget.status.setText(
                    f"{estimate.pairs:,} pairs · {estimate.additional_bytes/1e9:.2f} GB is over budget · "
                    f"'Keep localizations' set to {suggestion} %; Estimate again to confirm")
                return
            self.widget.status.setText(
                f"{estimate.pairs:,} pairs · estimated additional memory {estimate.additional_bytes/1e9:.2f} GB · "
                + (
                    "within budget"
                    if estimate.fits
                    else "over budget; reduce max drift or selection"
                )
            )

        self.start(work, done)

    def protect_clock(self, d):
        """Guard the columns this dataset's drift depends on, whatever their names."""
        table = d.table
        names = [n for n in ("frame_number", "frame", "time_s") if table.has_field(n)]
        if self.is_minflux(d):
            names.append(find_trace_column(table))
        table.protect_columns(*names)

    def apply(self, d, model, row_times, *, group=None, scale=(1.0, 1.0, 1.0)):
        self.protect_clock(d)
        values = model.evaluate(row_times) * np.asarray(scale)
        d.table.apply_position_delta(values, zdim_present=d.zdim_present)
        view = DriftView(
            model,
            np.array(row_times, copy=True),
            group=group or object(),
            scale=tuple(scale),
        )
        self.widen_ranges()
        self.pair_sets.pop(d.dataset_id, None)
        self.pair_keys.pop(d.dataset_id, None)
        self.store.set_drift(d.dataset_id, view)
        self.widget.status.setText(
            f"Drift applied · anchored at start · {len(model.knots)} windows · {model.provenance.get('pairs', 0):,} pairs"
        )
        self.draw_plot(model, d)
        self.refresh_status(d)
        self.sync()

    def widen_ranges(self):
        dtl = self.parent.data_to_layer_itf
        config = self.parent.render_config
        previous = {}
        for axis in ("x", "y", "z"):
            extent = getattr(dtl, f"render_range_{axis}")
            percent = np.asarray(getattr(config, f"range_{axis}_percent"))
            previous[axis] = (
                np.array_equal(percent, [0, 100]),
                dtl.percent_to_absolute(extent, percent),
            )
        for d in self.store:
            for raw in (False, True):
                xyz = self.coords(d, raw=raw)
                valid = np.isfinite(xyz).all(axis=1)
                for i, a in enumerate(("x", "y", "z")):
                    xyz[:, i] = dtl.transform_of(d).apply_axis(a, xyz[:, i])
                dtl.set_render_range(d.zdim_present, xyz[valid, ::-1])
        for axis, (full, absolute) in previous.items():
            extent = getattr(dtl, f"render_range_{axis}")
            span = extent[1] - extent[0]
            if full:
                percent = np.array([0.0, 100.0])
            elif np.isfinite(span) and span > 0:
                percent = (np.asarray(absolute) - extent[0]) / span * 100
            else:
                continue
            setattr(config, f"range_{axis}_percent", percent)
            slider = getattr(self.parent, f"Srender_range{axis}")
            slider.blockSignals(True)
            slider.setValue(tuple(percent))
            slider.blockSignals(False)

    def members(self, d):
        view = self.store.state_of(d).drift
        return [
            x
            for x in self.store
            if self.store.state_of(x).drift is not None
            and self.store.state_of(x).drift.group is view.group
        ]

    def transition(self, action):
        d = self.dataset
        if d is None or self.store.state_of(d).drift is None:
            return
        self.play_timer.stop()
        members = self.members(d)
        for other in members:
            view = self.store.state_of(other).drift
            if action == "discard":
                other.table.discard_position_stash()
                updated = None
                self.pair_sets.pop(other.dataset_id, None)
                self.pair_keys.pop(other.dataset_id, None)
            elif action == "undo":
                other.table.restore_positions()
                updated = replace(view, applied=False, alpha=1.0)
            else:
                other.table.apply_position_delta(
                    view.displacement(np.arange(len(other.table))),
                    zdim_present=other.zdim_present,
                )
                updated = replace(view, applied=True, alpha=1.0)
            self.store.set_drift(other.dataset_id, updated)
        self.widen_ranges()
        for other in members:
            self.parent.data_to_layer_itf.refresh_dataset(other)
            self.refresh_status(other)
        if action == "discard" and self._plot is not None:
            self._plot.figure.clear()
            self._plot.draw_idle()
        self.sync()

    def preview(self, value):
        self.widget.fraction_label.setText(
            f"Correction: {value*10} % (exports use applied data)"
        )
        self.preview_timer.start(30)

    def _apply_preview(self):
        self.preview_timer.stop()
        d = self.dataset
        if d is None:
            return
        view = self.store.state_of(d).drift
        if view is None or not view.applied:
            return
        for other in self.members(d):
            state = self.store.state_of(other).drift
            self.store.set_drift(
                other.dataset_id,
                replace(state, alpha=self.widget.fraction.value() / 10),
                positions_only=True,
            )
            self.refresh_status(other, preview_only=True)

    def play(self):
        self._playing = True
        self.sync()
        self.widget.fraction.setValue(0)
        self._apply_preview()
        self.play_timer.start(150)

    def _play_step(self):
        value = self.widget.fraction.value() + 1
        self.widget.fraction.setValue(min(10, value))
        self._apply_preview()
        if value < 10:
            self.play_timer.start(150)
        else:
            self._playing = False
            self.sync()

    def plan_arguments(self, d):
        view = self.store.state_of(d).drift
        if view is None and self._time_scale is None:
            return {}

        def offset(ids):
            delta = (
                view.offset(ids)
                if view is not None and view.applied
                else np.zeros((len(ids), 3))
            )
            if self._time_scale is not None and not d.zdim_present:
                times = self._display_times[d.dataset_id]
                scale, origin = self._time_scale
                delta[:, 2] = scale * (times[ids] - origin)
            return delta

        return {
            "position_offset": offset,
            "pairs": (
                self.pair_sets.get(d.dataset_id)
                if view is not None and view.applied
                else None
            ),
        }

    def time_view(self, enabled):
        try:
            self._time_view(enabled)
        except Exception:
            self._leave_time_view()
            raise

    def _leave_time_view(self):
        self.widget.time_view.blockSignals(True)
        self.widget.time_view.setChecked(False)
        self.widget.time_view.blockSignals(False)
        self._time_view(False)

    def _time_view(self, enabled):
        viewer = self.parent.viewer
        if enabled:
            self._display_times = {
                d.dataset_id: (
                    np.asarray(d.table.column("time_s"), float)
                    if d.table.has_field("time_s")
                    else self.clock(d)[0]
                )
                for d in self.store
            }
            units = {
                (
                    "seconds"
                    if d.table.has_field("time_s")
                    else "ranks" if self.is_minflux(d) else "frames"
                )
                for d in self.store
            }
            if len(units) != 1:
                raise ValueError(
                    "Time view requires a shared time unit across datasets"
                )
            clocks = list(self._display_times.values())
            if not all(np.isfinite(t).all() for t in clocks):
                raise ValueError("Time view needs finite acquisition times")
            origin = min(t.min() for t in clocks)
            span = max(t.max() for t in clocks) - origin
            if span <= 0:
                raise ValueError("Time view needs an acquisition time span")
            extents = []
            for d in self.store:
                xy = self.coords(d)[:, :2][d.table.finite_position_mask()]
                if len(xy):
                    extents.append(np.ptp(xy, axis=0).max())
            extent = max(extents, default=0.0)
            if not np.isfinite(extent) or extent <= 0:
                raise ValueError("Time view needs localizations spread in x and y")
            self._time_scale = (extent / (2 * span), origin)
            self.widget.time_view.setText(
                f"x, y, time · 1 µm = {1000/self._time_scale[0]:.1f} {next(iter(units))}"
            )
            self._camera = (
                viewer.dims.ndisplay,
                tuple(viewer.camera.center),
                viewer.camera.zoom,
                tuple(viewer.camera.angles),
            )
            viewer.dims.ndisplay = 3
            viewer.camera.angles = (30, 30, 30)
        else:
            self._time_scale = None
            self._display_times.clear()
            self.widget.time_view.setText("x, y, time view (2D)")
            if self._camera:
                ndim, center, zoom, angles = self._camera
                viewer.dims.ndisplay = ndim
                viewer.camera.center = center
                viewer.camera.zoom = zoom
                viewer.camera.angles = angles
                self._camera = None
        for d in self.store:
            self.parent.data_to_layer_itf.refresh_positions(d)

    def detect(self):
        d = self.dataset
        coords = self.coords(d, raw=True)
        times = self.clock(d)[0]
        valid = d.table.finite_position_mask()
        radius = self.widget.inputs["max_drift"].value()
        from .postprocessing.fiducials import detect_fiducials

        def done(report):
            candidates = [dict(c, selected=True) for c in report["accepted"]] + [
                dict(c, selected=False) for c in report["needs_review"]
            ]
            self.candidates[d.dataset_id] = candidates
            self.draw_fiducials(d, candidates)
            if self.store.state_of(d).drift is not None:
                self.draw_plot(self.store.state_of(d).drift.model, d)
            self.sync()
            self.widget.status.setText(
                f"{len(report['accepted'])} accepted beads; {len(report['needs_review'])} to review. Field bounds estimated from data."
            )

        self.start(
            lambda p: detect_fiducials(coords[valid], times[valid], radius, p), done
        )

    def exclude(self):
        from .postprocessing.fiducials import exclusion_mask

        d = self.dataset
        candidates = self.candidates.get(d.dataset_id, [])
        chosen = [
            c
            for i, c in enumerate(candidates)
            if self.widget.candidates.item(i).checkState() == Qt.CheckState.Checked
        ]
        # The ticked beads replace the previous choice: unticking a bead and
        # excluding again brings its rows back.
        d.table.set_excluded(np.ones(len(d.table), dtype=bool), exclude=False)
        if chosen:
            d.table.set_excluded(exclusion_mask(self.coords(d, raw=True), chosen))
        self.parent.data_to_layer_itf.refresh_dataset(d)
        self.invalidate_pairs()
        self.refresh_status(d)
        self.widget.status.setText(
            f"{np.count_nonzero(d.table.excluded & 1):,} localizations excluded as fiducials"
        )

    def restore_fiducials(self):
        d = self.dataset
        d.table.set_excluded(np.ones(len(d.table), dtype=bool), exclude=False)
        self.parent.data_to_layer_itf.refresh_dataset(d)
        self.invalidate_pairs()
        self.refresh_status(d)

    def group(self):
        from .postprocessing.grouping import link_localizations

        d = self.dataset
        ids = d.table.filtered_ids
        coords = self.coords(d, raw=True)[ids]
        # §8: anisotropic in 3D -- the distance is in units of xy precision
        coords[:, 2] *= self.z_scale(d, ids)
        frames = self.clock(d)[0][ids]
        weights = self.weights(d, ids)
        w = self.widget.inputs
        options = dict(
            max_distance=w["distance"].value(),
            max_dark_frames=w["dark"].value(),
            max_frames=w["duration"].value(),
        )

        def done(groups):
            all_groups = np.full(len(d.table), -1, dtype=np.int64)
            all_groups[ids] = groups
            d.table.set_side_column("group_id", all_groups)
            d.trace_column = "group_id"
            self.parent._sync_trace_controls()
            self.parent.data_to_layer_itf.refresh_positions(d)
            self.invalidate_pairs()
            self.widget.status.setText(
                f"{len(np.unique(groups)):,} groups. Use Connect traces to view them."
            )

        self.start(
            lambda p: link_localizations(
                coords, frames, weights=weights, progress=p, **options
            ),
            done,
        )

    @staticmethod
    def z_scale(d, ids):
        """σxy/σz: scaling z by it makes distances isotropic in units of precision."""
        if (
            d.zdim_present
            and d.table.has_sigma_axis("x")
            and d.table.has_sigma_axis("z")
        ):
            sx, sz = d.table.sigma_nm("x")[ids], d.table.sigma_nm("z")[ids]
            valid = np.isfinite(sx) & np.isfinite(sz) & (sx > 0) & (sz > 0)
            if valid.any():
                return float(np.median(sx[valid]) / np.median(sz[valid]))
        return 1.0

    def toggle_pairs(self, enabled):
        d = self.dataset
        if d is None:
            return
        self.store.set_appearance(d.dataset_id, pairs=enabled)
        if enabled:
            self.widget.fraction.setValue(0)
            self.invalidate_pairs()

    def invalidate_pairs(self):
        if self.dataset is not None:
            self.pair_keys.pop(self.dataset.dataset_id, None)

    def request_pairs(self, d, key):
        from .postprocessing.pair_network import find_pairs

        view = self.store.state_of(d).drift
        ids = d.table.filtered_ids
        coords = self.coords(d)[ids]
        raw_coords = self.coords(d, raw=True)[ids]
        times = view.row_times[ids]
        knots = view.model.knots
        windows = np.searchsorted((knots[1:] + knots[:-1]) / 2, times)
        first, last = view.model.frame_range
        windows[(times < first) | (times > last)] = -1
        group_name = (
            "group_id" if d.table.has_field("group_id") else find_trace_column(d.table)
        )
        groups = None if group_name is None else d.table.column(group_name)[ids].copy()
        z_scale = self.z_scale(d, ids)
        radius = self.widget.inputs["radius"].value()
        chains = self.widget.pair_mode.currentIndex() == 1
        self._pair_job = (d.dataset_id, d, key)
        self.pair_keys[d.dataset_id] = key
        self.pair_runner.submit(
            lambda p: find_pairs(
                coords,
                times,
                windows,
                ids,
                radius,
                groups=groups,
                z_scale=z_scale,
                chains=chains,
                min_dt=1.0 if groups is None else 0.0,
                progress=p,
                raw_coords=raw_coords,
            )
        )
        self.widget.pair_status.setText("Updating pair network…")

    def poll(self):
        if self._closed:
            return
        progress, result = self.runner.poll()
        if progress:
            stage, info = progress
            tracker = getattr(self, "_run_progress", None)
            if tracker is not None:
                fraction, text = tracker.update(stage, info)
                self.widget.progress.show()
                self.widget.progress.setValue(int(1000 * fraction))
                self.widget.status.setText(text)
            elif stage not in ("starting", "finished"):
                self.widget.status.setText(stage.capitalize() + "…")
        if result is not None:
            _, value, error = result
            self._run_progress = None
            self.widget.progress.hide()
            job, self._job = self._job, None
            if self._lock is not None:
                self._lock.__exit__(None, None, None)
                self._lock = None
            if job is not None and self.store.get(job[0]) is job[1]:
                if (
                    error
                    and OVER_BUDGET in error
                    and getattr(self, "_run_request", None)
                ):
                    self.widget.status.setText(error + " · dataset unchanged")
                    request, self._run_request = self._run_request, None
                    self.sync()
                    answer = QMessageBox.question(
                        self.widget,
                        "Memory",
                        error + "\n\nRun anyway? napari could be "
                        "stopped by the system if memory runs out.",
                    )
                    if answer == QMessageBox.StandardButton.Yes:
                        self._allow_over_budget = request
                        self.call(self.run)
                    return
                if error:
                    self.widget.status.setText(error + " · dataset unchanged")
                else:
                    self.call(lambda: job[2](value))
            self.sync()
        _, pairs_result = self.pair_runner.poll()
        if pairs_result is not None and self._pair_job is not None:
            _, value, error = pairs_result
            identity, d, key = self._pair_job
            if self.store.get(identity) is d and self.pair_keys.get(identity) == key:
                if error:
                    self.widget.pair_status.setText(error)
                    if error == "Cancelled":
                        # not a verdict on these settings: ask again next poll
                        self.pair_keys.pop(identity, None)
                else:
                    self.pair_sets[identity] = value
                    self.widget.pair_status.setText(
                        f"{len(value.row_ids):,} vectors in a {value.region_side_nm/1000:.2f} µm square. Close pairs: raw {value.raw_count:,} → corrected {value.corrected_count:,}. Selected on corrected data."
                    )
                    self.parent.data_to_layer_itf.refresh_positions(d, pairs_only=True)
        d = self.dataset
        if d is not None and not self.runner.busy and self.widget.pairs.isChecked():
            view = self.store.state_of(d).drift
            if view is not None and view.applied:
                key = (
                    id(view.model),  # held by the view while it is current
                    d.table.selection_version,
                    (
                        _digest(d.table.column("group_id"))
                        if d.table.has_field("group_id")
                        else None
                    ),
                    self.widget.inputs["radius"].value(),
                    self.widget.pair_mode.currentIndex(),
                )
                if self.pair_keys.get(d.dataset_id) != key:
                    self.call(lambda: self.request_pairs(d, key))

    def cancel(self):
        self.runner.cancel()
        self.play_timer.stop()
        if self._playing:
            self._playing = False
            self.widget.fraction.setValue(10)
            self._apply_preview()
            self.sync()

    def refresh_status(self, d, *, preview_only=False):
        state = self.store.state_of(d).drift
        d.postprocessing_status = (
            ("Drift applied" if state.applied else "Drift undone")
            if state
            else "No drift"
        )
        if state is not None:
            first, last = state.model.frame_range
            outside = np.count_nonzero(
                (state.row_times < first) | (state.row_times > last)
            )
            d.postprocessing_status += (
                f" · anchored at start · {outside:,} rows clamped at ends"
            )
        d.postprocessing_status += (
            f" · {np.count_nonzero(d.table.excluded & 1):,} fiducial rows excluded"
        )
        index = self.store.index_of(d.dataset_id)
        if 0 <= index < len(self.parent.channel):
            label = d.name
            if state is not None and state.applied and state.alpha != 1.0:
                label += f" · preview {state.alpha:.0%}"
            self.parent.channel[index].Label.setText(label)
        if not preview_only:
            self.parent.Lnumberoflocs.refresh_dataset(index)

    def draw_fiducials(self, d, candidates):
        from .napari_particles.trace_overlay import _view_kept

        old = self._fiducial_layers.pop(d.dataset_id, None)
        viewer = self.parent.viewer
        with _view_kept(viewer):
            if old is not None and old in viewer.layers:
                viewer.layers.remove(old)
            if not candidates:
                return
            transform = self.store.state_of(d).transform
            shapes = []
            for c in candidates:
                x, y = c["center_nm"]
                r = max(c["radius_p99_nm"], 1.0)
                # Ellipse corners, in the same world coordinates as localizations.
                xy = np.array(
                    [[x - r, y - r], [x - r, y + r], [x + r, y + r], [x + r, y - r]]
                )
                shape = np.ones((4, 3))
                shape[:, 1] = transform.apply_axis("y", xy[:, 1])
                shape[:, 2] = transform.apply_axis("x", xy[:, 0])
                shapes.append(shape)
            self._fiducial_layers[d.dataset_id] = viewer.add_shapes(
                shapes,
                shape_type="ellipse",
                name=d.name + " fiducial review (raw xy)",
                face_color="transparent",
                edge_color="yellow",
                edge_width=2,
            )

    def draw_plot(self, model, dataset=None):
        if self._plot is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._plot = FigureCanvasQTAgg(Figure(figsize=(4, 2)))
            self.widget.plot_layout.addWidget(self._plot)
        self._plot.figure.clear()
        axes = self._plot.figure.add_subplot(111)
        t = np.linspace(model.frame_range[0], model.frame_range[1], 300)
        for i, name in enumerate(("dx", "dy", "dz")):
            axes.plot(t, model.evaluate(t)[:, i], label=name)
        axes.scatter(model.knots, model.evaluate(model.knots)[:, 0], s=6, color="C0")
        if dataset is not None and self.candidates.get(dataset.dataset_id):
            raw = self.coords(dataset, raw=True)
            times = self.clock(dataset)[0]
            windows = np.searchsorted((model.knots[:-1] + model.knots[1:]) / 2, times)
            for c in self.candidates[dataset.dataset_id]:
                if not c.get("selected", False):
                    continue
                chosen = (
                    np.sum((raw[:, :2] - c["center_nm"]) ** 2, axis=1)
                    <= c["radius_p99_nm"] ** 2
                ) & np.isfinite(raw).all(axis=1)
                counts = np.bincount(windows[chosen], minlength=len(model.knots))
                for i in range(3 if dataset.zdim_present else 2):
                    sums = np.bincount(
                        windows[chosen],
                        weights=raw[chosen, i],
                        minlength=len(model.knots),
                    )
                    means = np.divide(
                        sums, counts, out=np.full(len(counts), np.nan), where=counts > 0
                    )
                    early = means[: max(1, int(np.ceil(0.02 * len(means))))]
                    if np.isfinite(early).any():
                        means -= np.nanmean(early)
                        axes.plot(
                            model.knots, means, linestyle=":", alpha=0.5, color=f"C{i}"
                        )
        axes.set_xlabel("Trace rank" if model.clock == "trace" else "Frame")
        axes.set_ylabel("Drift [nm]")
        axes.legend(fontsize=7)
        self._plot.figure.tight_layout()
        self._plot.draw_idle()

    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self.widget, "Save drift", "", "HDF5 (*.h5)"
        )
        if path:
            self.store.state_of(self.dataset).drift.model.save(path)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(
            self.widget, "Load drift", "", "HDF5 (*.h5)"
        )
        if not path:
            return
        d = self.dataset
        times, kind, rank_times, _ = self.clock(d)
        # A file COMET itself wrote has no clock; it is this dataset's kind.
        model = DriftModel.load(path, legacy_clock=kind)
        if kind != model.clock:
            raise ValueError("Drift and dataset use different acquisition clocks")
        if kind == "trace":
            expected = model.provenance.get("trace_membership")
            if expected is not None and expected != _digest(
                d.table.column(find_trace_column(d.table))
            ):
                raise ValueError(
                    "Trace identities differ; use Apply to other datasets with a shared clock"
                )
        first, last = model.frame_range
        if times.max() < first or times.min() > last:
            raise ValueError(
                "The drift was estimated over a time range this dataset does not reach"
            )
        self.apply(d, model, times)

    def transfer(self):
        source = self.dataset
        state = self.store.state_of(source).drift
        choices = [
            d for d in self.store if d is not source and not d.table.has_position_stash
        ]
        if not choices:
            raise ValueError("Load another dataset without an existing drift first")
        labels = [f"{d.dataset_id}: {d.name}" for d in choices]
        label, ok = QInputDialog.getItem(
            self.widget, "Apply drift", "Target dataset", labels, 0, False
        )
        if not ok:
            return
        d = choices[labels.index(label)]
        times, kind, rank_times, _ = self.clock(d)
        model = state.model
        if kind != model.clock:
            raise ValueError("Drift and target use different clocks")
        if kind == "trace":
            if rank_times is None or model.rank_times is None:
                raise ValueError(
                    "Transfer requires finite, strictly increasing trace times in both datasets"
                )
            if (
                rank_times.min() < model.rank_times[0]
                or rank_times.max() > model.rank_times[-1]
            ):
                raise ValueError("Target time range exceeds source acquisition")
            times = np.interp(
                rank_times[times.astype(int)],
                model.rank_times,
                np.arange(len(model.rank_times)),
            )
        first, last = model.frame_range
        if times.min() < first or times.max() > last:
            raise ValueError(
                "Target time range lies outside the range the drift was estimated over"
            )
        answer = QMessageBox.question(
            self.widget,
            "Shared acquisition clock",
            "Were these datasets recorded simultaneously using the same acquisition clock?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        src_scale = np.asarray(self.store.state_of(source).transform.scale)
        dst_scale = np.asarray(self.store.state_of(d).transform.scale)
        if np.any(dst_scale == 0):
            raise ValueError("Cannot transfer through a zero world scale")
        self.apply(
            d,
            model,
            times,
            group=state.group,
            scale=np.asarray(state.scale) * src_scale / dst_scale,
        )

    def export(self):
        from .postprocessing.io import save_localizations

        d = self.dataset
        path, _ = QFileDialog.getSaveFileName(
            self.widget, "Save applied localizations", "", "napari-storm (*.ns)"
        )
        if path:
            save_localizations(path, d, self.store.state_of(d).drift)

    def close(self):
        self._closed = True
        self.timer.stop()
        self.play_timer.stop()
        self.preview_timer.stop()
        self.runner.close()
        self.pair_runner.close()
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None
        self.store.unsubscribe(self.on_store_event)
