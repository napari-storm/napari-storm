"""Turning what is on screen into an :class:`ExportPlan`.

The adapter between the widget and the host-free exporter in
:mod:`napari_storm.core.ome_export`. Everything scientific -- the Gaussian
model, the raster, the calibration -- lives there; what lives here is knowing
where the plugin keeps its datasets, its render range and its settings.

There is no axis-order conversion here any more, and the reason is worth
recording. The planner used to return coordinates as ``(z, x, y)`` and sigmas
as ``(z, y, x)``, so this module reordered the first and not the second -- and
the docstring warned that assuming one order for both "swaps x and y widths on
anisotropic data". That warning was correct and the renderer was the code
committing the error, pairing the two column-wise all along. Both are now
``(z, y, x)``, napari's order and this exporter's, so a channel is built by
reading the request rather than by rearranging it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .core import ACTIVE, FILTERED
from .core.ome_export import ExportChannel, plan_export
from .core.render_planner import SIGMA_TO_SIZE_FACTOR

__all__ = [
    "SCOPE_CURRENT_VIEW",
    "SCOPE_EVERYTHING",
    "ExportOptions",
    "channel_for",
    "current_view_z_nm",
    "export_bounds_nm",
    "plan_from_widget",
]

#: The default: X/Y bounded by the current render range, every localization
#: inside the current Z range projected onto one plane.
SCOPE_CURRENT_VIEW = "current_view"

#: Opt-in: a Z-stack over the whole dataset extent, ignoring the render range.
SCOPE_EVERYTHING = "everything"

#: Which plane a 2-D export projects onto.  The rasterizer always collapses
#: axis 0 and draws axes 1 and 2, so a projection is a permutation of the
#: ``(z, y, x)`` columns rather than a second code path: put the axis being
#: summed over first, and the remaining two are the image, row axis first.
PROJECTION_XY = "xy"
PROJECTION_XZ = "xz"
PROJECTION_YZ = "yz"

_PROJECTION_AXES = {
    PROJECTION_XY: (0, 1, 2),  # sum z, image (y, x) -- what exports did before
    PROJECTION_XZ: (1, 0, 2),  # sum y, image (z, x)
    PROJECTION_YZ: (2, 0, 1),  # sum x, image (z, y)
}


def projection_axes(projection):
    """The ``(z, y, x)`` column order *projection* rasterizes in."""
    try:
        return _PROJECTION_AXES[projection]
    except KeyError:
        raise ValueError(
            f"unknown projection {projection!r}; "
            f"expected one of {sorted(_PROJECTION_AXES)}"
        ) from None


def project_channel(channel, projection):
    """*channel* with its coordinates and widths permuted for *projection*.

    Widths move with the coordinates because a Gaussian's extent is per axis:
    projecting onto XZ and keeping the y width would draw the wrong shape and
    would silently look plausible.
    """
    axes = projection_axes(projection)
    if axes == (0, 1, 2):
        return channel
    return replace(
        channel,
        coords_nm=np.ascontiguousarray(channel.coords_nm[:, axes]),
        sigmas_nm=np.ascontiguousarray(channel.sigmas_nm[:, axes]),
    )


def project_bounds(bounds_nm, projection):
    """*bounds_nm* reordered to match :func:`project_channel`."""
    axes = projection_axes(projection)
    return tuple(tuple(bounds_nm[axis]) for axis in axes)


@dataclass(frozen=True)
class ExportOptions:
    """What the user chose in the dialog."""

    pixel_size_nm: float = 10.0
    scope: str = SCOPE_CURRENT_VIEW
    z_step_nm: float = 50.0
    #: Only consulted for a 2-D export: a volume has every plane in it already.
    projection: str = PROJECTION_XY

    @property
    def is_3d(self):
        return self.scope == SCOPE_EVERYTHING


def channel_for(interface, dataset, planner=None):
    """One dataset as an :class:`ExportChannel`, over its **filter** selection.

    Not the active selection: the render budget's strided subsample is an
    accommodation to the GPU, and a saved file must contain what the user's
    filters left, however much that is.
    """
    planner = planner or interface.planner
    settings = interface.gaussian_settings()
    traits = interface.traits_of(dataset)
    transform = interface.transform_of(dataset)
    table = dataset.table

    request = planner.plan(
        table,
        settings,
        traits,
        name=getattr(dataset, "name", "channel"),
        transform=transform,
        selection=FILTERED,
    )

    # Already (z, y, x), like the sigmas beside it -- see the module docstring.
    coords_zyx = request.coords
    # The planner normalizes sigmas against the largest and reports the
    # billboard edge, which is a fixed multiple of it; undo that to recover
    # nanometres, because a raster is measured in nanometres and not in
    # billboards.
    largest_sigma_nm = request.size / SIGMA_TO_SIZE_FACTOR
    sigmas_zyx = np.asarray(request.sigmas, dtype=np.float64) * largest_sigma_nm

    appearance = interface.renderer.appearance(getattr(dataset, "dataset_id", None))
    colormap = None if appearance is None else appearance.colormap

    return ExportChannel(
        name=getattr(dataset, "name", "channel"),
        coords_nm=np.ascontiguousarray(coords_zyx, dtype=np.float64),
        sigmas_nm=sigmas_zyx,
        values=np.asarray(request.values, dtype=np.float64),
        colormap=None if colormap is None else str(colormap),
        n_displayed=table.selection(ACTIVE).n,
    )


def export_bounds_nm(widget, options, channels=None):
    """``((z0, z1), (y0, y1), (x0, x1))`` for this export, in nanometres.

    For the current view that is the render range the user set. For everything
    it is the extent of the localizations actually being written, which is not
    the same as the dataset extent once a filter has removed the outliers that
    used to define it.
    """
    interface = widget.data_to_layer_itf

    if options.is_3d:
        if not channels:
            raise ValueError("a 3-D export needs at least one channel")
        stacked = np.concatenate([c.coords_nm for c in channels if len(c.coords_nm)])
        if not len(stacked):
            raise ValueError("no localizations to export")
        # Pad by the widest splat so the Gaussians are not clipped at the edge
        # of their own bounding box.
        pad = (
            max(
                (float(np.max(c.sigmas_nm)) for c in channels if len(c.sigmas_nm)),
                default=0.0,
            )
            * SIGMA_TO_SIZE_FACTOR
        )
        low = stacked.min(axis=0) - pad
        high = stacked.max(axis=0) + pad
        return ((low[0], high[0]), (low[1], high[1]), (low[2], high[2]))

    x0, x1 = interface.percent_to_absolute(
        interface.render_range_x, widget.render_range_slider_x_percent
    )
    y0, y1 = interface.percent_to_absolute(
        interface.render_range_y, widget.render_range_slider_y_percent
    )
    z0, z1 = current_view_z_nm(widget)
    return ((float(z0), float(z1)), (float(y0), float(y1)), (float(x0), float(x1)))


def current_view_z_nm(widget):
    """``(z0, z1)`` for a 2-D export: the z window it projects through.

    What makes the grid one plane is ``z_step_nm=None``, not a degenerate
    interval, so this reports the real window even for XY.  It used to report
    ``(0, 0)`` unconditionally, which XY tolerated -- the rasterizer ignores
    the collapsed axis entirely -- and which XZ and YZ could not: a projection
    puts z on a *visible* image axis, and an empty interval there is an image
    one pixel high, positioned at the origin, with the data outside it.
    """
    interface = widget.data_to_layer_itf
    z_range = interface.render_range_z
    if not getattr(widget, "zdim", False) or not np.all(np.isfinite(z_range)):
        # Genuinely flat: there is no window to project through.
        return 0.0, 0.0
    return interface.percent_to_absolute(z_range, widget.render_range_slider_z_percent)


def run_export(
    path,
    plan,
    parent=None,
    force_synchronous=False,
    on_finished=None,
    on_error=None,
    on_cancelled=None,
):
    """Write *plan* to *path* without freezing the window.

    Unlike a file read, this **is** interruptible -- the writer polls between
    tiles -- and its progress is real rather than indeterminate, because the
    tile count is known before it starts. So this does not reuse
    `background_loading`'s spinner; it shows a determinate bar that means what
    it says.

    Returns the worker, or ``None`` when it ran inline.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QProgressDialog

    from .background_loading import _thread_worker, run_on_main_thread
    from .core.ome_export import tile_count, write_ome_tiff

    total = tile_count(plan)
    cancelled = {"flag": False}

    worker_factory = None if force_synchronous else _thread_worker()
    if worker_factory is None:
        try:
            write_ome_tiff(path, plan)
        except InterruptedError:
            if on_cancelled is not None:
                on_cancelled()
            return None
        except Exception as error:  # noqa: BLE001 - routed to the caller
            if on_error is None:
                raise
            on_error(error)
            return None
        if on_finished is not None:
            on_finished(path)
        return None

    dialog = QProgressDialog(
        f"Writing {plan.shape[3]:,} x {plan.shape[2]:,} px...",
        "Cancel",
        0,
        total,
        parent,
    )
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.canceled.connect(lambda: cancelled.__setitem__("flag", True))

    def _progress(done, of_total):
        # Touches a widget, so it has to cross back to the GUI thread even
        # though the writer calls it from the worker.
        run_on_main_thread(dialog.setValue, int(done))

    def _write(_handle=None):
        return write_ome_tiff(
            path,
            plan,
            progress=_progress,
            should_cancel=lambda: cancelled["flag"],
        )

    def _done(_result):
        if on_finished is not None:
            on_finished(path)

    def _failed(error):
        if isinstance(error, InterruptedError):
            if on_cancelled is not None:
                on_cancelled()
            return
        if on_error is None:
            raise error
        on_error(error)

    def _close():
        dialog.close()
        dialog.deleteLater()

    worker = worker_factory(
        _write,
        start_thread=False,
        connect={"returned": _done, "errored": _failed, "finished": _close},
    )()
    worker._napari_storm_progress = dialog
    worker.start()
    return worker


def plan_from_widget(widget, options):
    """Everything about the export, knowable before a byte is written."""
    interface = widget.data_to_layer_itf
    datasets = [
        dataset
        for dataset in widget.localization_datasets
        if getattr(dataset, "table", None) is not None
    ]
    if not datasets:
        raise ValueError("there are no datasets loaded to export")

    channels = [channel_for(interface, dataset) for dataset in datasets]
    bounds = export_bounds_nm(widget, options, channels)
    if not options.is_3d:
        channels = [project_channel(c, options.projection) for c in channels]
        bounds = project_bounds(bounds, options.projection)
    return plan_export(
        channels,
        bounds,
        pixel_size_nm=options.pixel_size_nm,
        z_step_nm=options.z_step_nm if options.is_3d else None,
    )
