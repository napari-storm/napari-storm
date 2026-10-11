"""An anchored drift with an explicit clock and clamped interpolation."""

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class DriftModel:
    knots: np.ndarray
    displacements_nm: np.ndarray
    clock: str = "frame"
    rank_times: np.ndarray = None
    provenance: dict = field(default_factory=dict)
    anchor_nm: np.ndarray = None
    interpolation: str = "cubic"
    #: First and last time of the data the drift was estimated from. Between it
    #: and the end knots the spline is used, as COMET does; outside it the drift
    #: is held at the nearer end. None (files from before) means the end knots.
    frame_range: tuple = None

    def __post_init__(self):
        k = np.array(self.knots, dtype=np.float64, copy=True)
        d = np.array(self.displacements_nm, dtype=np.float64, copy=True)
        if k.ndim != 1 or len(k) < 2 or d.shape != (len(k), 3):
            raise ValueError(
                "Drift needs at least two knots and three coordinates per knot"
            )
        if (
            not np.isfinite(k).all()
            or not np.isfinite(d).all()
            or np.any(np.diff(k) <= 0)
        ):
            raise ValueError("Drift knots must be finite and strictly increasing")
        if self.clock not in ("frame", "trace") or self.interpolation not in (
            "cubic",
            "linear",
        ):
            raise ValueError("Unsupported drift clock or interpolation")
        span = (
            (k[0], k[-1])
            if self.frame_range is None
            else tuple(float(v) for v in self.frame_range)
        )
        if (
            len(span) != 2
            or not np.isfinite(span).all()
            or span[0] > min(span[1], k[0])
            or span[1] < k[-1]
        ):
            raise ValueError("The drift's frame range must enclose its knots")
        # Fix only the arbitrary translation: zero drift at the acquisition start.
        # No fraction of windows or minimum population is needed to define it.
        anchor = (
            (
                np.asarray(CubicSpline(k, d, axis=0)(span[0]))
                if self.interpolation == "cubic"
                else d[0].copy()
            )
            if self.anchor_nm is None
            else np.array(self.anchor_nm, dtype=float, copy=True)
        )
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("Invalid drift anchor")
        object.__setattr__(self, "frame_range", span)
        for name, array in (
            ("knots", k),
            ("displacements_nm", d),
            ("anchor_nm", anchor),
        ):
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if self.rank_times is not None:
            times = np.array(self.rank_times, dtype=float, copy=True)
            if (
                times.ndim != 1
                or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError(
                    "Trace times must be finite and strictly increasing for transfer"
                )
            times.flags.writeable = False
            object.__setattr__(self, "rank_times", times)

    def evaluate(self, times):
        t = np.asarray(times, dtype=np.float64)
        if not np.isfinite(t).all():
            raise ValueError("Cannot evaluate drift at non-finite times")
        t = np.clip(t, self.frame_range[0], self.frame_range[1])
        if self.interpolation == "cubic":
            result = CubicSpline(self.knots, self.displacements_nm, axis=0)(t)
        else:
            result = np.stack(
                [
                    np.interp(t, self.knots, self.displacements_nm[:, i])
                    for i in range(3)
                ],
                axis=-1,
            )
        return result - self.anchor_nm

    def evaluate_time(self, times):
        if self.clock != "trace":
            return self.evaluate(times)
        if self.rank_times is None:
            raise ValueError("This drift has no unambiguous trace clock for transfer")
        return self.evaluate(
            np.interp(times, self.rank_times, np.arange(len(self.rank_times)))
        )

    def save(self, path):
        import h5py

        with h5py.File(path, "w") as f:
            g = f.create_group("drift")
            g["center_frames"] = self.knots
            g["drift_per_segment_nm"] = self.displacements_nm
            g["anchor_nm"] = self.anchor_nm
            if self.rank_times is not None:
                g["rank_times_s"] = self.rank_times
            g["frame_range"] = np.asarray(self.frame_range, dtype=float)
            f.attrs["napari_storm_drift_schema"] = 2
            f.attrs["clock"] = self.clock
            f.attrs["interpolation"] = self.interpolation
            f.attrs["provenance"] = json.dumps(self.provenance)

    @classmethod
    def load(cls, path, *, legacy_clock=None):
        import h5py

        with h5py.File(path, "r") as f:
            # COMET's original layout has no clock metadata; do not guess.
            clock = f.attrs.get("clock", legacy_clock)
            if clock is None:
                raise ValueError(
                    "Legacy COMET file has no clock metadata; specify frame or trace explicitly"
                )
            g = f["drift"]
            knots = g["center_frames"][:]
            displacements = g["drift_per_segment_nm"][:]
            # COMET's QC backends write NaN for flagged windows; COMET itself
            # interpolates over the others, and so does this.
            valid = np.isfinite(knots) & np.isfinite(displacements).all(axis=1)
            return cls(
                knots[valid],
                displacements[valid],
                frame_range=tuple(g["frame_range"][:]) if "frame_range" in g else None,
                clock=clock,
                rank_times=g["rank_times_s"][:] if "rank_times_s" in g else None,
                anchor_nm=g["anchor_nm"][:] if "anchor_nm" in g else None,
                interpolation=f.attrs.get("interpolation", "cubic"),
                provenance=json.loads(f.attrs.get("provenance", "{}")),
            )
