"""Writing the rasterized reconstruction out as a calibrated OME-TIFF.

The other half of §4.1. :mod:`raster` decides what the pixels are; this decides
what the file says they *mean*, which is the part that makes the export usable
as a measurement rather than as a picture.

**The pixel size is written down, with its unit.** OME `PhysicalSizeX/Y/Z` plus
an explicit `PhysicalSizeXUnit`. A pixel size with no declared unit is precisely
the defect this feature exists to avoid; reproducing it in the output would be
worse than not writing one at all.

**One channel per dataset**, with the colormap travelling as OME channel
metadata rather than baked into the pixel values. Burning a colour lookup into
the samples would make the file pretty and unquantifiable.

**The file may be larger than memory.** Tiles are streamed from
:func:`~napari_storm.core.raster.rasterize_tiles` straight into the writer, so
the ceiling on an export is disk space. This is the opposite policy from the
screen: the render budget degrades the *view* under memory pressure, and the
exporter must never degrade the *file*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .raster import TILE_PIXELS, GaussianGrid, rasterize_tiles

__all__ = [
    "ExportChannel",
    "ExportPlan",
    "ExportWarning",
    "plan_export",
    "tile_count",
    "write_ome_tiff",
]

#: OME's unit token for nanometres.  Written explicitly on every axis.
NANOMETRE = "nm"


@dataclass(frozen=True)
class ExportChannel:
    """One dataset, ready to rasterize.

    Attributes:
        name: channel name written into the OME metadata.
        coords_nm: ``(N, 3)`` localization centres in ``(z, y, x)`` nanometres.
        sigmas_nm: ``(N, 3)`` Gaussian widths in the same axis order.
        values: ``(N,)`` per-localization amplitude.
        colormap: colour name recorded as channel metadata, never baked in.
        n_displayed: how many of these rows the *screen* is currently drawing.
            Equal to ``len(coords_nm)`` unless the render budget thinned the
            view, which is the condition the export warning exists for.
    """

    name: str
    coords_nm: np.ndarray
    sigmas_nm: np.ndarray
    values: np.ndarray
    colormap: str = None
    n_displayed: int = None

    @property
    def n_localizations(self):
        return len(self.coords_nm)

    @property
    def was_display_limited(self):
        return (
            self.n_displayed is not None
            and self.n_displayed < self.n_localizations
        )


@dataclass(frozen=True)
class ExportWarning:
    """Something the user should know before the file is written.

    Carried as data rather than raised or printed, so the same plan can be shown
    in a dialog, logged, or asserted in a test without the exporter deciding
    which.
    """

    kind: str
    message: str


@dataclass(frozen=True)
class ExportPlan:
    """Everything about an export that can be known before writing a byte."""

    grid: GaussianGrid
    channels: tuple
    warnings: tuple = field(default=())

    @property
    def shape(self):
        """``(channels, z, y, x)`` of the file that will be written."""
        return (len(self.channels),) + tuple(self.grid.shape)

    @property
    def nbytes(self):
        """Size of the pixel data on disk, uncompressed."""
        return len(self.channels) * self.grid.nbytes(np.float32)

    @property
    def n_localizations(self):
        return sum(channel.n_localizations for channel in self.channels)


def plan_export(channels, bounds_nm, pixel_size_nm, z_step_nm=None):
    """Work out the grid and collect anything worth warning about.

    Separate from writing on purpose: a 40 GB export should be describable, and
    refusable, before it starts rather than after.
    """
    if not channels:
        raise ValueError("an export needs at least one channel")

    grid = GaussianGrid.covering(bounds_nm, pixel_size_nm, z_step_nm)
    warnings = []

    thinned = [c for c in channels if c.was_display_limited]
    if thinned:
        detail = "; ".join(
            f"{c.name}: {c.n_displayed:,} shown, {c.n_localizations:,} exported"
            for c in thinned
        )
        warnings.append(
            ExportWarning(
                "display_limited",
                "The view is showing fewer localizations than this export "
                "contains, because the render budget thinned it to fit in "
                f"memory ({detail}). The file is built from every localization "
                "your filters left active, so it will not match the screen "
                "pixel for pixel -- it has more data in it, not less.",
            )
        )

    empty = [c for c in channels if c.n_localizations == 0]
    if empty:
        warnings.append(
            ExportWarning(
                "empty_channel",
                "No localizations to draw for: "
                + ", ".join(c.name for c in empty)
                + ". Those channels will be written as blank planes.",
            )
        )

    return ExportPlan(grid=grid, channels=tuple(channels), warnings=tuple(warnings))


def _ome_metadata(plan):
    """OME-XML fields tifffile writes into the ImageDescription."""
    grid = plan.grid
    metadata = {
        "axes": "CZYX",
        "PhysicalSizeX": float(grid.pixel_size_nm[2]),
        "PhysicalSizeXUnit": NANOMETRE,
        "PhysicalSizeY": float(grid.pixel_size_nm[1]),
        "PhysicalSizeYUnit": NANOMETRE,
        "PhysicalSizeZ": float(grid.pixel_size_nm[0]),
        "PhysicalSizeZUnit": NANOMETRE,
        "Channel": {"Name": [channel.name for channel in plan.channels]},
    }
    # Where the raster sits in world space, one Plane per (channel, z).  The
    # pixel size alone says how big a pixel is, not where the image is: without
    # this, re-importing an export lands it at the origin rather than at the
    # render range it was cut from.  Plane order for CZYX is channel-major.
    nz = grid.shape[0]
    z_positions = [
        float(grid.origin_nm[0] + index * grid.pixel_size_nm[0])
        for index in range(nz)
    ] * len(plan.channels)
    n_planes = len(z_positions)
    metadata["Plane"] = {
        "PositionX": [float(grid.origin_nm[2])] * n_planes,
        "PositionXUnit": [NANOMETRE] * n_planes,
        "PositionY": [float(grid.origin_nm[1])] * n_planes,
        "PositionYUnit": [NANOMETRE] * n_planes,
        "PositionZ": z_positions,
        "PositionZUnit": [NANOMETRE] * n_planes,
    }

    colormaps = [channel.colormap for channel in plan.channels]
    if any(colormaps):
        # Not an OME core field, so it goes in as a plain annotation rather than
        # being smuggled into one that means something else.
        metadata["Channel"]["napari_storm_colormap"] = [
            "" if name is None else str(name) for name in colormaps
        ]
    return metadata


def write_ome_tiff(path, plan, progress=None, should_cancel=None):
    """Rasterize *plan* and stream it to *path*, one tile at a time.

    Args:
        path: destination ``.ome.tif``.
        plan: from :func:`plan_export`.
        progress: ``callable(done_tiles, total_tiles)``, called as each tile
            is rasterized so a long export can show where it has got to.
        should_cancel: ``callable() -> bool``, polled per tile. A cancelled
            export deletes the partial file and raises
            :class:`InterruptedError`, rather than leaving a truncated image
            that looks like a finished one.

    Returns the :class:`ExportPlan` that was written.
    """
    import tifffile

    path = str(path)
    try:
        with tifffile.TiffWriter(path, ome=True, bigtiff=True) as writer:
            # One CZYX series, so the channels stay a single image rather than
            # becoming unrelated pages a reader has to guess how to group.
            # `tile=` is what makes the writer consume tiles instead of planes,
            # and so what keeps peak memory at one tile rather than one plane:
            # a 20000 x 20000 plane would be 1.6 GB on its own.
            writer.write(
                _tile_iterator(plan, progress, should_cancel),
                shape=plan.shape,
                dtype=np.float32,
                tile=(TILE_PIXELS, TILE_PIXELS),
                metadata=_ome_metadata(plan),
            )
    except InterruptedError:
        _remove_quietly(path)
        raise
    return plan


def _remove_quietly(path):
    import os

    try:
        os.remove(path)
    except OSError:
        # Never mask a cancellation with a cleanup failure.
        pass


def tile_count(plan):
    """How many tiles :func:`write_ome_tiff` will write, for progress reporting."""
    _nz, ny, nx = plan.grid.shape
    per_plane = _ceil_div(ny, TILE_PIXELS) * _ceil_div(nx, TILE_PIXELS)
    return len(plan.channels) * plan.grid.shape[0] * per_plane


def _ceil_div(a, b):
    return -(-int(a) // int(b))


def _tile_iterator(plan, progress=None, should_cancel=None):
    """Yield tiles in the order tifffile writes them: C, Z, tile row, tile column.

    A generator rather than an array. This is the whole reason an export can be
    larger than memory: nothing here ever holds more than one tile.
    """
    total = tile_count(plan)
    done = 0

    for channel in plan.channels:
        for _z, _ys, _xs, tile in rasterize_tiles(
            channel.coords_nm,
            channel.sigmas_nm,
            channel.values,
            plan.grid,
            tile_pixels=TILE_PIXELS,
        ):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("export cancelled")
            # Reported before the yield, not after: a generator resumes only
            # when the consumer asks for the next item, and tifffile stops
            # asking once it has the last tile -- so anything after the final
            # yield never runs, and the last tile would never be counted.
            done += 1
            if progress is not None:
                progress(done, total)
            yield tile
