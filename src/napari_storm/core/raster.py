"""CPU rasterization of the Gaussian model, at a pixel size the caller chose.

This is the export half of §4.1. It is deliberately **not** a screenshot: it
evaluates the same Gaussian model the renderer draws, on a grid the user
specifies, in double the precision the canvas has. The plan states the intent
plainly -- "the export is the reference and the canvas is the approximation" --
and the difference is not an accident to be minimised later:

* the shader locks its antialias texture scale below a distance cutoff,
* additive blending saturates in the canvas' 8-bit output,
* the perspective camera foreshortens.

None of those apply here. What comes out is the analytic sum of Gaussians,
which is why the golden test pins it against a closed-form evaluation rather
than against a screen capture.

**Never downsample.** The requested pixel size is honoured exactly, so a wide
field at a fine pixel size produces a *large file*, never a coarser image. That
would be untenable if the whole array had to exist in memory at once, so
nothing here ever allocates the output: :func:`rasterize_tiles` yields one tile
at a time and the caller streams them to disk. Peak memory is set by the tile,
not by the image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GaussianGrid",
    "TILE_PIXELS",
    "rasterize_tiles",
    "rasterize",
    "splat_extent_nm",
]

#: How far out a Gaussian is evaluated, in sigmas. Matches the renderer's
#: billboard, which is cut at five sigma -- about 4% of peak amplitude -- so the
#: export inherits the same truncation rather than inventing a second one.
SPLAT_SIGMAS = 5.0

#: Target tile edge in pixels. A 1024x1024 float32 tile is 4 MB, which keeps the
#: streaming path comfortably inside any budget while staying large enough that
#: per-tile overhead does not dominate.
TILE_PIXELS = 1024


def splat_extent_nm(sigmas_nm):
    """How far one localization's Gaussian reaches, in nanometres."""
    return float(SPLAT_SIGMAS * np.max(sigmas_nm))


@dataclass(frozen=True)
class GaussianGrid:
    """The raster the export is written onto.

    Attributes:
        origin_nm: world coordinate of the *centre* of pixel ``(0, 0, 0)``,
            as ``(z, y, x)``.
        pixel_size_nm: ``(z, y, x)`` sampling. The z entry is the slice step and
            is independent of the xy pixel size, because Z sampling in an SMLM
            dataset is not tied to lateral sampling.
        shape: ``(nz, ny, nx)``. ``nz == 1`` is the 2-D projection.
    """

    origin_nm: tuple
    pixel_size_nm: tuple
    shape: tuple

    @classmethod
    def covering(cls, bounds_nm, pixel_size_nm, z_step_nm=None):
        """The smallest grid at *pixel_size_nm* that covers *bounds_nm*.

        *bounds_nm* is ``((z0, z1), (y0, y1), (x0, x1))``; pass a degenerate z
        range for a 2-D projection. The pixel size is never adjusted to make the
        result fit -- the grid grows instead, which is the whole point.
        """
        pixel_size_nm = float(pixel_size_nm)
        if not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
            raise ValueError("pixel size must be a positive, finite number of nm")

        (z0, z1), (y0, y1), (x0, x1) = bounds_nm
        if z_step_nm is None:
            nz, z_step_nm, z_origin = 1, pixel_size_nm, 0.5 * (z0 + z1)
        else:
            z_step_nm = float(z_step_nm)
            if not np.isfinite(z_step_nm) or z_step_nm <= 0:
                raise ValueError("z step must be a positive, finite number of nm")
            nz = max(1, int(np.ceil((z1 - z0) / z_step_nm)))
            z_origin = z0 + 0.5 * z_step_nm

        ny = max(1, int(np.ceil((y1 - y0) / pixel_size_nm)))
        nx = max(1, int(np.ceil((x1 - x0) / pixel_size_nm)))
        return cls(
            origin_nm=(z_origin, y0 + 0.5 * pixel_size_nm, x0 + 0.5 * pixel_size_nm),
            pixel_size_nm=(z_step_nm, pixel_size_nm, pixel_size_nm),
            shape=(nz, ny, nx),
        )

    @property
    def is_2d(self):
        return self.shape[0] == 1

    @property
    def n_pixels(self):
        nz, ny, nx = self.shape
        return int(nz) * int(ny) * int(nx)

    def nbytes(self, dtype=np.float32):
        """What the *whole* image would cost in memory, which is never paid."""
        return self.n_pixels * np.dtype(dtype).itemsize

    def axis_coordinates_nm(self, axis, start, stop):
        """World coordinates of pixel centres ``[start, stop)`` along *axis*."""
        return (
            self.origin_nm[axis]
            + np.arange(start, stop, dtype=np.float64) * self.pixel_size_nm[axis]
        )


def _tile_ranges(length, step):
    for start in range(0, length, step):
        yield start, min(start + step, length)


def rasterize_tiles(coords_nm, sigmas_nm, values, grid, tile_pixels=TILE_PIXELS):
    """Yield ``(z, y_slice, x_slice, tile)`` covering *grid*, one at a time.

    *coords_nm* is ``(N, 3)`` in ``(z, y, x)`` world nanometres, *sigmas_nm* is
    ``(N, 3)`` in the same order, and *values* is ``(N,)`` amplitude.

    Only the localizations whose splat reaches a given tile are evaluated for
    it, so cost scales with the data actually in view rather than with N per
    tile. The output is float32 and unbounded: intensity is a sum, and clipping
    it to a display range is a presentation decision for whoever writes the
    file, not a property of the measurement.
    """
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    sigmas_nm = np.asarray(sigmas_nm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if coords_nm.ndim != 2 or coords_nm.shape[1] != 3:
        raise ValueError("coords_nm must be (N, 3) in (z, y, x)")
    if sigmas_nm.shape != coords_nm.shape:
        raise ValueError("sigmas_nm must have the same shape as coords_nm")
    if values.shape != (len(coords_nm),):
        raise ValueError("values must be (N,)")

    nz, ny, nx = grid.shape
    # Per-localization reach, so a tile can select its own contributors.
    reach = SPLAT_SIGMAS * sigmas_nm
    # A zero sigma would make the exponent divide by zero; the renderer clamps
    # sigmas upstream, and a defensive floor here keeps a hand-built call safe.
    safe_sigmas = np.maximum(sigmas_nm, np.finfo(np.float64).tiny)

    for z_index in range(nz):
        z_nm = grid.origin_nm[0] + z_index * grid.pixel_size_nm[0]
        if grid.is_2d:
            # The 2-D export is a projection: every localization contributes
            # its full amplitude, with no z falloff applied.
            in_slab = np.ones(len(coords_nm), dtype=bool)
            z_weight = np.ones(len(coords_nm))
        else:
            in_slab = np.abs(coords_nm[:, 0] - z_nm) <= reach[:, 0]
            dz = (coords_nm[:, 0] - z_nm) / safe_sigmas[:, 0]
            z_weight = np.exp(-0.5 * dz * dz)

        for y0, y1 in _tile_ranges(ny, tile_pixels):
            ys = grid.axis_coordinates_nm(1, y0, y1)
            for x0, x1 in _tile_ranges(nx, tile_pixels):
                xs = grid.axis_coordinates_nm(2, x0, x1)
                near = (
                    in_slab
                    & (coords_nm[:, 1] + reach[:, 1] >= ys[0])
                    & (coords_nm[:, 1] - reach[:, 1] <= ys[-1])
                    & (coords_nm[:, 2] + reach[:, 2] >= xs[0])
                    & (coords_nm[:, 2] - reach[:, 2] <= xs[-1])
                )
                tile = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
                if np.any(near):
                    _accumulate(
                        tile,
                        coords_nm[near],
                        safe_sigmas[near],
                        values[near] * z_weight[near],
                        ys,
                        xs,
                    )
                yield z_index, slice(y0, y1), slice(x0, x1), tile


def _accumulate(tile, coords_nm, sigmas_nm, amplitudes, ys, xs):
    """Sum the Gaussians of *coords_nm* onto *tile*.

    Each localization is evaluated only over the pixels its own five-sigma box
    covers. Doing it per localization rather than as one big broadcast is what
    keeps peak memory at one tile: an ``(N, ny, nx)`` intermediate for a crowded
    tile would be gigabytes.
    """
    y_step = ys[1] - ys[0] if len(ys) > 1 else 1.0
    x_step = xs[1] - xs[0] if len(xs) > 1 else 1.0

    for centre, sigma, amplitude in zip(coords_nm, sigmas_nm, amplitudes):
        if amplitude == 0.0:
            continue
        y_reach = SPLAT_SIGMAS * sigma[1]
        x_reach = SPLAT_SIGMAS * sigma[2]
        y_lo = int(np.floor((centre[1] - y_reach - ys[0]) / y_step))
        y_hi = int(np.ceil((centre[1] + y_reach - ys[0]) / y_step)) + 1
        x_lo = int(np.floor((centre[2] - x_reach - xs[0]) / x_step))
        x_hi = int(np.ceil((centre[2] + x_reach - xs[0]) / x_step)) + 1
        y_lo, y_hi = max(y_lo, 0), min(y_hi, len(ys))
        x_lo, x_hi = max(x_lo, 0), min(x_hi, len(xs))
        if y_lo >= y_hi or x_lo >= x_hi:
            continue

        dy = (ys[y_lo:y_hi] - centre[1]) / sigma[1]
        dx = (xs[x_lo:x_hi] - centre[2]) / sigma[2]
        # The index box above is derived with floor/ceil, so it rounds *outward*
        # and can keep a rim of pixels a fraction beyond five sigma. Cutting on
        # the coordinate instead makes the support exactly the same square
        # whatever the pixel size happens to be -- and a square is the right
        # shape, because the renderer's billboard is a quad, not a disc.
        gy = np.where(np.abs(dy) > SPLAT_SIGMAS, 0.0, np.exp(-0.5 * dy * dy))
        gx = np.where(np.abs(dx) > SPLAT_SIGMAS, 0.0, np.exp(-0.5 * dx * dx))
        # Separable, so the 2-D Gaussian is one outer product rather than a
        # full 2-D exponential.
        tile[y_lo:y_hi, x_lo:x_hi] += amplitude * np.outer(gy, gx)


def rasterize(coords_nm, sigmas_nm, values, grid, tile_pixels=TILE_PIXELS):
    """The whole image at once.  For tests and small grids only.

    Production paths stream :func:`rasterize_tiles` to disk; this materialises
    the array the streaming exists to avoid, and says so rather than being
    quietly available for a 100 GB export.
    """
    out = np.zeros(grid.shape, dtype=np.float32)
    for z_index, y_slice, x_slice, tile in rasterize_tiles(
        coords_nm, sigmas_nm, values, grid, tile_pixels
    ):
        out[z_index, y_slice, x_slice] = tile
    return out
