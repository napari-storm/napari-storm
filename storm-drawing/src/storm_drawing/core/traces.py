"""Traces: one molecule localized repeatedly, connected in time order.

A MINFLUX acquisition localizes the same molecule again and again and says so:
every localization carries the id of its trace (``tid``, stored by the readers
as ``trace_id``).  Drawn as points, a trace is a smear.  Drawn as a path through
its own localizations, in the order they were measured, it is what a tracking
acquisition is *for* -- and for a fixed sample it shows how far a molecule's
repeated localizations wander.  Other formats have a track or group id that
means the same thing: Picasso's ``group``, a tracker's track id, a custom
table's declared column.

This module arranges rows; it draws nothing.  What it hands a backend is a
:class:`TraceVertices`: the selected localizations regrouped by trace and put in
time order, with the *same* world coordinates the splat renderer receives -- the
same rows, the same :class:`~napari_storm.core.WorldTransform`, the same
``(z, y, x)`` order, flat data pinned to the same plane.  A trajectory therefore
passes exactly through its own drawn localizations, after any alignment,
filter or render-range crop, because there is only one place coordinates are
computed (`RenderPlanner.coordinates`) and this uses it.

The ordering is the part that fails silently.  MINFLUX writes the iterations of
concurrent traces interleaved, in acquisition order; a path drawn through them
in file order zig-zags between molecules, and a path drawn through one trace's
rows in the wrong order doubles back on itself and still renders happily.
``_tests/test_traces.py`` walks synthetic traces along straight lines at a
fixed step so that either mistake shows up as a step of the wrong length.

Traces are a visualisation.  They are not part of the reconstruction and no
export writes them.

Host-free like the rest of ``napari_storm.core``: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .localization_table import ACTIVE
from .validation import InvalidLocalizationData
from .world_transform import IDENTITY

__all__ = [
    "AUTO",
    "BUILTIN_COLOR_BY",
    "COLOR_BY_PROGRESS",
    "COLOR_BY_TIME",
    "COLOR_BY_TRACE",
    "DEFAULT_TIME_COLUMNS",
    "DEFAULT_TRACE_COLUMNS",
    "DEFAULT_TRACE_WIDTH_PX",
    "TraceVertices",
    "arrange_traces",
    "find_trace_column",
    "is_trace_column",
    "plan_traces",
    "resolve_time_column",
    "trace_spread_is_meaningful",
    "validate_trace_color_by",
    "validate_trace_width_px",
]

#: Columns that identify a trace, tried in order when a caller does not name
#: one.  ``trace_id`` is where every MINFLUX reader here stores ``tid``.  Other
#: names -- Picasso's ``group``, a custom table's own -- are passed explicitly:
#: a column called ``group`` means a picked region as often as it means a
#: molecule, and connecting the localizations of a pick is a choice to make on
#: purpose rather than by default.
DEFAULT_TRACE_COLUMNS = ("trace_id",)

#: Time columns tried, in order, when a caller asks for :data:`AUTO`: MINFLUX's
#: ``tim`` (stored as ``time_s``), then the camera frame number the STORM
#: readers and Picasso write.
DEFAULT_TIME_COLUMNS = ("time_s", "frame_number", "frame")

#: Ask for the first of :data:`DEFAULT_TIME_COLUMNS` the table has.
AUTO = "auto"

#: Colour each trace its own hue.  Stable across filtering: the hue follows the
#: trace's own id, not its position among the traces that happen to be drawn.
COLOR_BY_TRACE = "trace"

#: Colour by when each localization was measured -- the time column, or the
#: order within the trace where there is none.
COLOR_BY_TIME = "time"

#: Colour by how far along its own trace a localization is, 0 at the first
#: and 1 at the last: the direction a tracked molecule moved.
COLOR_BY_PROGRESS = "progress"

#: What every :class:`TraceVertices` can be coloured by, whatever was planned.
BUILTIN_COLOR_BY = (COLOR_BY_TRACE, COLOR_BY_TIME, COLOR_BY_PROGRESS)

#: Line width, in screen pixels, unless a host says otherwise.
DEFAULT_TRACE_WIDTH_PX = 2.0

#: The golden ratio's fractional part.  Multiplying consecutive integers by it
#: and keeping the fraction spreads them evenly around a cyclic colormap, so
#: traces with neighbouring ids -- which MINFLUX measures one after another, and
#: which therefore tend to sit next to each other -- get distant hues.
_GOLDEN = 0.6180339887498949


def trace_spread_is_meaningful(is_tracking):
    """Whether a per-trace standard deviation estimates localization precision.

    It does for a fixed sample: the spread of repeated localizations of a
    molecule that is not going anywhere is the precision.  It does not for a
    tracking run, where that spread is the trajectory -- on the MINFLUX files
    this was checked against, 4 nm fixed against 25 nm tracking, and the
    difference is the molecule moving, not the instrument doing worse.

    Whether an acquisition *is* tracking cannot be read off the file: ``bot``
    marks the first row of every trace in a fixed-sample dataset too, ``eot``
    is unset in both, and ``sta``/``thi``/``sqi`` do not differ.  pyMINFLUX
    takes it as a constructor argument for the same reason.  So the caller
    says so, and nothing here guesses.

    Nothing in napari-storm estimates a width from trace spread.  Drawing a
    trace as a path is safe either way -- it shows the localizations, not a
    statistic of them -- and this function exists so that a future sigma
    estimator has to ask before treating a trajectory as a precision.
    """
    return not is_tracking


# ----------------------------------------------------------------------
# Which columns
# ----------------------------------------------------------------------


def _integer_valued(values):
    """True for an integer column, or a float column holding only integers."""
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        return True
    if values.dtype.kind == "f":
        finite = np.isfinite(values)
        return bool(finite.all() and np.array_equal(np.floor(values), values))
    return False


def is_trace_column(table, name):
    """Whether *name* is a column of *table* that can identify traces.

    It has to exist and hold integers.  A float column holding only whole
    numbers counts -- a dataframe that went through pandas often arrives that
    way -- but one with a fraction or a NaN in it does not: there is no trace
    0.5, and guessing which trace a NaN belongs to would draw a line that was
    never measured.
    """
    if name is None or not table.has_field(name):
        return False
    return _integer_valued(table.column(name))


def find_trace_column(table, candidates=DEFAULT_TRACE_COLUMNS):
    """The first of *candidates* that :func:`is_trace_column`, or None."""
    for name in candidates:
        if is_trace_column(table, name):
            return name
    return None


def resolve_time_column(table, time_column=AUTO):
    """The column to order each trace by, or None to use the row order.

    *time_column* may be :data:`AUTO` -- the first of
    :data:`DEFAULT_TIME_COLUMNS` present -- a column name, which must exist, or
    None, which asks for the row order outright.
    """
    if time_column is None:
        return None
    if time_column == AUTO:
        return next(
            (name for name in DEFAULT_TIME_COLUMNS if table.has_field(name)), None
        )
    if not table.has_field(time_column):
        raise InvalidLocalizationData(
            f"no time column {time_column!r} in this table; it has "
            f"{', '.join(table.field_names) or 'no columns'}"
        )
    return time_column


def validate_trace_color_by(name):
    """*name* as a colour-by key, or a ValueError if it cannot be one."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"trace_color_by must be a non-empty string, not {name!r}")
    return name


def validate_trace_width_px(width_px):
    """*width_px* as a float, or a ValueError if it is not a line width."""
    try:
        value = float(width_px)
    except (TypeError, ValueError):
        raise ValueError(f"trace_width_px must be a number, not {width_px!r}") from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"trace_width_px must be finite and > 0, not {width_px!r}")
    return value


# ----------------------------------------------------------------------
# Ordering
# ----------------------------------------------------------------------


def arrange_traces(trace, time=None, min_vertices=2):
    """Row positions grouped by trace and sorted by time within each.

    Args:
        trace: ``(N,)`` trace id per row.
        time: ``(N,)`` sort key per row, or None for the row order.
        min_vertices: traces with fewer rows than this are left out.  The
            default, 2, drops traces of one localization: there is nothing to
            connect.

    Returns ``(order, starts)``: *order* indexes the rows, every trace's rows
    contiguous and in ascending id order; *starts* is where each trace begins
    in *order*.

    **Rows with a negative id belong to no trace** and are left out.  Clustering
    and linking tools write -1 for "noise" or "unlinked", and joining every one
    of those into a single trace would draw a line across the whole field of
    view.

    Ties in *time* keep the row order -- explicitly, as the last sort key,
    rather than by trusting a sort to be stable.  Rows are written in
    acquisition order, so a tie (a float32 time column cannot resolve
    neighbouring MINFLUX iterations of a long run) is settled the way the
    instrument settled it.
    """
    trace = np.asarray(trace)
    n = len(trace)
    if n == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    position = np.arange(n, dtype=np.intp)
    member = trace >= 0
    if not member.all():
        position = position[member]
    keys = [position]
    if time is not None:
        keys.append(np.asarray(time)[position])
    keys.append(trace[position])
    order = position[np.lexsort(keys)]

    sorted_trace = trace[order]
    if len(order) == 0:
        return order, np.empty(0, dtype=np.intp)
    new_trace = np.empty(len(order), dtype=bool)
    new_trace[0] = True
    np.not_equal(sorted_trace[1:], sorted_trace[:-1], out=new_trace[1:])
    starts = np.flatnonzero(new_trace)
    if min_vertices > 1:
        lengths = np.diff(np.append(starts, len(order)))
        long_enough = lengths >= min_vertices
        if not long_enough.all():
            order = order[np.repeat(long_enough, lengths)]
            lengths = lengths[long_enough]
            starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.intp)
            if len(lengths) == 0:
                starts = np.empty(0, dtype=np.intp)
    return order, starts


# ----------------------------------------------------------------------
# The result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TraceVertices:
    """The selected localizations, regrouped by trace and put in time order.

    Every trace's vertices are contiguous and in time order; traces follow one
    another in ascending id.  Connecting consecutive vertices of the same trace
    draws every trajectory, and nothing else.

    Attributes:
        coords: ``(N, 3)`` float32 world coordinates in ``(z, y, x)`` order --
            for each vertex, exactly the coordinate the splat renderer was
            handed for that localization.
        trace_index: ``(N,)`` which trace each vertex belongs to, as a dense
            index ``0 .. n_traces - 1`` into :attr:`trace_ids`.
        trace_ids: ``(n_traces,)`` each trace's own id from the trace column,
            ascending.
        time: ``(N,)`` float64, what each trace was sorted by: the time
            column, or each vertex's position in its trace when there is none.
        localization_ids: ``(N,)`` the stable localization id -- the
            canonical row index -- of each vertex.
        trace_column: the column the traces came from.
        time_column: the column :attr:`time` came from, or None when each
            trace kept its row order.
        columns: extra per-vertex columns planned for colouring, by name,
            already in vertex order.
    """

    coords: np.ndarray
    trace_index: np.ndarray
    trace_ids: np.ndarray
    time: np.ndarray
    localization_ids: np.ndarray
    trace_column: str
    time_column: str = None
    columns: dict = None

    def __len__(self):
        return len(self.coords)

    @property
    def n_vertices(self):
        return len(self.coords)

    @property
    def n_traces(self):
        return len(self.trace_ids)

    @property
    def n_segments(self):
        """Lines drawn: one fewer than each trace's vertices."""
        return self.n_vertices - self.n_traces

    @property
    def starts(self):
        """Where each trace begins among the vertices."""
        if self.n_vertices == 0:
            return np.empty(0, dtype=np.intp)
        new_trace = np.empty(self.n_vertices, dtype=bool)
        new_trace[0] = True
        np.not_equal(self.trace_index[1:], self.trace_index[:-1], out=new_trace[1:])
        return np.flatnonzero(new_trace)

    @property
    def lengths(self):
        """Vertices per trace, in :attr:`trace_ids` order."""
        return np.bincount(self.trace_index, minlength=self.n_traces)

    @property
    def vertex_trace_ids(self):
        """Each vertex's trace id from the trace column."""
        return self.trace_ids[self.trace_index]

    @property
    def color_by_options(self):
        """Every name :meth:`color_values` answers for."""
        return BUILTIN_COLOR_BY + tuple(self.columns or ())

    def color_values(self, color_by):
        """One float64 per vertex to colour by, unnormalized.

        ``"trace"`` is in ``[0, 1)`` already and follows the trace's own id,
        so a trace keeps its hue when a filter removes its neighbours -- a
        cyclic colormap should take it as it is.  The others are raw values
        for a backend to scale to its colormap, as it would a contrast range.
        """
        if color_by == COLOR_BY_TRACE:
            ids = self.vertex_trace_ids.astype(np.float64)
            return np.modf(ids * _GOLDEN)[0] % 1.0
        if color_by == COLOR_BY_TIME:
            return np.asarray(self.time, dtype=np.float64)
        if color_by == COLOR_BY_PROGRESS:
            starts = self.starts
            lengths = np.diff(np.append(starts, self.n_vertices))
            rank = np.arange(self.n_vertices) - np.repeat(starts, lengths)
            span = np.repeat(np.maximum(lengths - 1, 1), lengths)
            return rank / span
        columns = self.columns or {}
        if color_by in columns:
            return np.asarray(columns[color_by], dtype=np.float64)
        raise KeyError(
            f"cannot colour traces by {color_by!r}: these vertices were planned "
            f"with {', '.join(self.color_by_options)}. Plan with "
            f"trace_properties=({color_by!r},) to colour by a column."
        )


def _empty(trace_column, time_column):
    return TraceVertices(
        coords=np.empty((0, 3), dtype=np.float32),
        trace_index=np.empty(0, dtype=np.int64),
        trace_ids=np.empty(0, dtype=np.int64),
        time=np.empty(0, dtype=np.float64),
        localization_ids=np.empty(0, dtype=np.intp),
        trace_column=trace_column,
        time_column=time_column,
        columns={},
    )


def plan_traces(
    table,
    traits,
    *,
    trace_column=None,
    time_column=AUTO,
    transform=IDENTITY,
    selection=ACTIVE,
    properties=(),
    coords=None,
):
    """Arrange *table*'s selected rows as trajectories.

    Args:
        table: a :class:`~napari_storm.core.LocalizationTable`.
        traits: the dataset's :class:`~napari_storm.core.DatasetTraits`, which
            decide -- as for the splats -- whether z is a coordinate or the
            flat-data plane.
        trace_column: the integer column identifying traces.  None takes the
            first of :data:`DEFAULT_TRACE_COLUMNS` the table has.
        time_column: what each trace is ordered by.  :data:`AUTO`, the default,
            uses the table's time -- the first of :data:`DEFAULT_TIME_COLUMNS`
            present -- **and otherwise the row order within the trace**, which
            is acquisition order for every reader here.  A name uses that
            column; None asks for the row order outright.
        transform: the dataset's :class:`~napari_storm.core.WorldTransform`.
        selection: :data:`~napari_storm.core.ACTIVE` -- the rows the splat
            renderer draws, the default -- or
            :data:`~napari_storm.core.FILTERED`.  Under the render budget's
            display limit the active rows are an even subsample, and a trace
            is then drawn through the localizations that are drawn.
        properties: extra numeric columns to carry per vertex, for colouring.
        coords: the ``(N, 3)`` coordinates already planned for *selection*,
            as :meth:`RenderPlanner.plan` passes them, so they are computed
            once and are the same array the splats get.  Computed here when
            omitted.

    Returns a :class:`TraceVertices`.  Traces of a single selected localization
    have nothing to connect and are left out; so are rows with a negative id.
    """
    if trace_column is None:
        trace_column = find_trace_column(table)
        if trace_column is None:
            raise InvalidLocalizationData(
                "this table has no trace column: connecting localizations needs "
                f"an integer column such as {', '.join(DEFAULT_TRACE_COLUMNS)}"
            )
    elif not is_trace_column(table, trace_column):
        if not table.has_field(trace_column):
            raise InvalidLocalizationData(
                f"no trace column {trace_column!r} in this table; it has "
                f"{', '.join(table.field_names) or 'no columns'}"
            )
        raise InvalidLocalizationData(
            f"{trace_column!r} cannot identify traces: it has to hold whole "
            "numbers, with no NaN"
        )
    time_column = resolve_time_column(table, time_column)
    for name in properties:
        if not table.has_field(name):
            raise InvalidLocalizationData(f"no column {name!r} to colour traces by")
        if table.column(name).dtype.kind not in "iufb":
            raise InvalidLocalizationData(
                f"{name!r} is not numeric and cannot colour traces"
            )

    rows = table.selection(selection)
    if coords is None:
        # Imported here: the planner imports this module to plan traces with
        # a request, and the coordinates must come from its one formula.
        from .render_planner import RenderPlanner

        coords = RenderPlanner().coordinates(rows, traits, transform)
    ids = rows.ids
    if len(coords) != len(ids):
        raise ValueError(f"coords has {len(coords)} rows for a selection of {len(ids)}")
    if len(ids) == 0:
        return _empty(trace_column, time_column)

    trace = table.column(trace_column)[ids]
    if trace.dtype.kind == "f":
        trace = trace.astype(np.int64)
    time = None if time_column is None else table.column(time_column)[ids]
    order, starts = arrange_traces(trace, time)
    if len(order) == 0:
        return _empty(trace_column, time_column)

    lengths = np.diff(np.append(starts, len(order)))
    trace_ids = np.asarray(trace[order[starts]])
    trace_index = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    if time is None:
        sort_key = np.arange(len(order), dtype=np.float64) - np.repeat(starts, lengths)
    else:
        sort_key = np.asarray(time[order], dtype=np.float64)
    localization_ids = ids[order]
    return TraceVertices(
        coords=np.ascontiguousarray(coords[order]),
        trace_index=trace_index,
        trace_ids=trace_ids,
        time=sort_key,
        localization_ids=localization_ids,
        trace_column=trace_column,
        time_column=time_column,
        columns={
            name: np.asarray(table.column(name)[localization_ids], dtype=np.float64)
            for name in properties
        },
    )
