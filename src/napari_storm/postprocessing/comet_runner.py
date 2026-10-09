"""Use COMET 1.x localization-count windows without changing its optimizer."""

from dataclasses import asdict, dataclass, field

import numpy as np

from .drift import DriftModel
from .pair_budget import estimate_memory


@dataclass(frozen=True)
class CometParameters:
    window: int = 60
    max_drift_nm: float = 300.0
    target_sigma_nm: float = field(default=10.0, init=False)

    def __post_init__(self):
        if (
            not np.isfinite(self.window)
            or self.window < 1
            or int(self.window) != self.window
        ):
            raise ValueError("Window must be a positive integer")
        if (
            not np.isfinite([self.max_drift_nm, self.target_sigma_nm]).all()
            or min(self.max_drift_nm, self.target_sigma_nm) <= 0
        ):
            raise ValueError("Drift radius and sigma must be positive and finite")
        if self.max_drift_nm // 3 <= 0:
            raise ValueError(
                "Max drift must be at least 3 nm for COMET's initial sigma"
            )


def prepare_input(coords, times, params, *, time_origin=None):
    """Validate input for COMET's original count-based segmentation.

    No minimum population, custom merging, smoothing or sampling is imposed.
    COMET keeps complete frames together and includes its existing tail policy.
    """
    coords, times = np.asarray(coords, dtype=float), np.asarray(times, dtype=float)
    if times.ndim != 1 or coords.shape != (len(times), 3):
        raise ValueError("COMET input needs coordinates shaped (N, 3) and N times")
    if not np.isfinite(coords).all() or not np.isfinite(times).all():
        raise ValueError("COMET input needs finite coordinates and times")
    if len(times) < 2 or np.ptp(times) == 0:
        raise ValueError("At least two distinct acquisition times are required")
    if not np.equal(times, np.floor(times)).all():
        raise ValueError("COMET requires integer acquisition frames or trace ranks")
    if times.min() < 0 or times.max() >= np.iinfo(np.int64).max:
        raise ValueError("COMET requires nonnegative frames within int64 range")
    return np.column_stack((coords, times))


def run_comet(
    coords,
    times,
    params=None,
    *,
    progress=None,
    clock="frame",
    rank_times=None,
    total_rows=None,
    time_origin=None,
    allow_over_budget=False,
):
    params = params or CometParameters()
    report = progress or (lambda stage, info=None: None)
    report("preparing", {})
    data = prepare_input(coords, times, params, time_origin=time_origin)
    import comet
    from comet.core.segmenter import segment_by_num_locs_per_window
    from packaging.version import Version

    if not Version("1.2.0") <= Version(comet.__version__) < Version("2"):
        raise RuntimeError(
            f"COMET {comet.__version__} is installed; 1.2.x is required. "
            'Run: pip uninstall py-comet, then pip install "napari-storm[comet]" '
            "(COMET is published as comet-smlm since 1.2)."
        )
    segmentation = segment_by_num_locs_per_window(data[:, 3], params.window)
    if segmentation.n_segments < 2:
        raise ValueError("Fewer than two windows; reduce localizations per window")
    report("counting pairs", {})
    budget = estimate_memory(
        data[:, :3],
        params.max_drift_nm,
        total_rows=total_rows,
        frame_count=int(data[:, 3].max()) + 1,
    )
    report(
        "memory estimate",
        {"pairs": budget.pairs, "additional_gb": budget.additional_bytes / 1e9},
    )
    if not budget.possible:
        raise MemoryError(
            f"Estimated additional memory {budget.additional_bytes / 1e9:.2f} GB exceeds physical memory. "
            "Reduce max drift or select a smaller estimation region."
        )
    if not budget.fits and not allow_over_budget:
        # The dock recognises this text and asks before running anyway.
        raise MemoryError(
            f"Estimated additional memory {budget.additional_bytes / 1e9:.2f} GB is over the comfortable "
            "memory budget (current use plus reserve)."
        )
    if budget.pairs == 0:
        raise ValueError(
            "No neighboring pairs; increase max drift or use a larger region"
        )
    _, details = comet.comet_run_kd(
        data,
        segmentation_mode=1,
        segmentation_var=params.window,
        max_drift_nm=params.max_drift_nm,
        target_sigma_nm=params.target_sigma_nm,
        mode="cpu",
        display=False,
        interactive=False,
        return_details=True,
        progress=report,
    )
    knots = np.asarray(details.knot_frames)
    provenance = {
        "comet_version": comet.__version__,
        "parameters": asdict(params),
        "pairs": int(details.n_pairs),
        "backend": details.backend,
        "evaluations": int(details.n_evaluations),
        "timings_s": details.timings_s,
    }
    provenance.update(
        sigma_accepted_nm=details.sigma_accepted_nm,
        runs=int(details.n_runs),
        auto_downsampled=bool(details.auto_downsampled),
        windows=int(segmentation.n_segments),
    )
    valid = np.isfinite(details.knot_drift_nm).all(axis=1)
    return DriftModel(
        knots[valid],
        details.knot_drift_nm[valid],
        clock=clock,
        rank_times=rank_times,
        provenance=provenance,
        frame_range=(float(np.min(times)), float(np.max(times))),
    )
