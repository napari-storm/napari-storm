"""Canonical columnar localization data, independent of any host application.

**The contract.**

* ``records`` is the single source of truth. It is written once at construction
  and afterwards only through :meth:`LocalizationTable.adjust_column`, which is
  also the one place derived caches are dropped. It is never reordered, never
  shortened, and never filtered in place.

* Because of that, **the row index into ``records`` is the stable localization
  ID.** :attr:`LocalizationTable.active_ids` returns the IDs currently selected;
  they remain meaningful across any number of filter changes.

* Selection is a **boolean mask** over ``records``, so filtering costs one
  boolean array rather than a copy of the record array, and
  :attr:`LocalizationTable.active_records` is a zero-copy read-only view while
  nothing is excluded.

* There are **two** masks, and the distinction is load-bearing. The
  :attr:`~LocalizationTable.filter_mask` is what the user asked for. The
  *display limit* on top of it is what the renderer can afford — an evenly
  strided subsample imposed by the memory budget. :attr:`active_mask` is their
  intersection: what gets drawn. Anything that leaves the process (an export, a
  saved dataset, a reported count of the user's own data) must read the filter
  set; only the GPU sees the display set. Collapsing the two would silently
  write a subsample of a scientist's data to disk because their graphics card
  was busy.

* Coordinates are exposed in **nanometres** by :meth:`coordinate_nm`, whatever
  unit the underlying column happens to use. A format that stores camera pixels
  declares its column names and a scale factor; nothing downstream needs to know
  which convention a given file used.

* **The same is true of the other measured columns.** Fitted widths come back
  from :meth:`sigma_nm` and photon counts from :meth:`photons`, under names the
  format declares. This used to be true of positions only, and the planner read
  uncertainty through hardcoded pixel-native field names -- so a format storing
  nanometres could render in fixed-width mode and crashed in variable-width
  mode, which is the mode where its fitted widths are worth anything. Naming
  and unit conversion belong to the same object for every column or for none.

Axis *order* is deliberately not this class's business. Positions are keyed by
axis name here; the renderer's ``(z, y, x)`` column order is applied at that
boundary, where it belongs.
"""
from __future__ import annotations

import numpy as np

from .validation import non_finite_mask

__all__ = [
    "ACTIVE",
    "AXES",
    "COORDINATE_DTYPE",
    "DEFAULT_PHOTON_COLUMN",
    "DEFAULT_POSITION_COLUMNS",
    "DEFAULT_SIGMA_COLUMNS",
    "FILTERED",
    "LocalizationTable",
    "TableSelection",
]

#: The rows the user selected.  What leaves the process -- exports, saves,
#: reported counts of the user's own data.
FILTERED = "filtered"

#: The rows actually drawn: :data:`FILTERED` narrowed by the display limit.
#: Only the GPU should see this one.
ACTIVE = "active"

#: Spatial axes, in name order.  Not a memory layout -- see the module docstring.
AXES = ("x", "y", "z")

#: Column names used by formats that already store nanometres.
DEFAULT_POSITION_COLUMNS = {"x": "x_pos_nm", "y": "y_pos_nm", "z": "z_pos_nm"}

#: Fitted-width column names, defaulting to the pixel-native names the bundled
#: STORM readers write.  A format that fits in nanometres declares its own and
#: passes ``sigma_scale_nm=1.0``.
DEFAULT_SIGMA_COLUMNS = {
    "x": "sigma_x_pixels",
    "y": "sigma_y_pixels",
    "z": "sigma_z_pixels",
}

#: Photon-count column name.  Unlike sigmas this carries no length unit, so
#: there is no scale factor to go with it.
DEFAULT_PHOTON_COLUMN = "photon_count"

#: Coordinates are cached in float32: at the ~1e5 nm range of a typical field of
#: view that resolves to well under 0.01 nm, two orders of magnitude below
#: single-molecule localization precision, and it halves the largest arrays the
#: renderer derives from them.
COORDINATE_DTYPE = np.float32


class TableSelection:
    """One of a table's two row selections, behind a uniform accessor.

    The table has always exposed both -- `filtered_records` beside
    `active_records`, `filtered_coordinate_nm` beside `active_coordinate_nm` --
    but every consumer had to pick a prefix and hard-code it. That was fine
    while the renderer was the only consumer, because the renderer always wants
    the drawn rows. The exporter always wants the *selected* rows, and it needs
    the same Gaussian model computed over them, so the code that computes that
    model has to be able to take the selection as an argument rather than
    spelling it into every attribute name.

    Not a copy and not a snapshot: it reads through to the table, so it must not
    outlive a mask change.
    """

    __slots__ = ("_table", "kind")

    def __init__(self, table, kind):
        if kind not in (FILTERED, ACTIVE):
            raise ValueError(f"selection must be {FILTERED!r} or {ACTIVE!r}")
        self._table = table
        self.kind = kind

    @property
    def n(self):
        return (
            self._table.n_filtered if self.kind == FILTERED else self._table.n_active
        )

    @property
    def records(self):
        return (
            self._table.filtered_records
            if self.kind == FILTERED
            else self._table.active_records
        )

    @property
    def ids(self):
        return (
            self._table.filtered_ids if self.kind == FILTERED else self._table.active_ids
        )

    def coordinate_nm(self, axis):
        if self.kind == FILTERED:
            return self._table.filtered_coordinate_nm(axis)
        return self._table.active_coordinate_nm(axis)

    def sigma_nm(self, axis):
        """Fitted width for *axis*, in nanometres, over these rows."""
        if self.kind == FILTERED:
            return self._table.filtered_sigma_nm(axis)
        return self._table.active_sigma_nm(axis)

    def photons(self):
        """Photon counts over these rows."""
        if self.kind == FILTERED:
            return self._table.filtered_photons()
        return self._table.active_photons()

    def has_sigma_axis(self, axis):
        return self._table.has_sigma_axis(axis)

    def sigma_column_name(self, axis):
        return self._table.sigma_column_name(axis)

    def has_photons(self):
        return self._table.has_photons()

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"TableSelection({self.kind!r}, n={self.n})"


def _readonly(array):
    """Mark an array read-only before handing it out."""
    array.flags.writeable = False
    return array


class LocalizationTable:
    """Canonical localization records plus the currently active subset."""

    def __init__(
        self,
        records,
        *,
        position_columns=None,
        position_scale_nm=1.0,
        sigma_columns=None,
        sigma_scale_nm=None,
        photon_column=None,
        copy=True,
    ):
        """
        Args:
            records: a numpy record array of localizations.
            position_columns: axis name -> column name. Defaults to the
                nanometre column names in :data:`DEFAULT_POSITION_COLUMNS`.
                Axes whose column is absent from *records* are simply not
                present on this table; see :meth:`has_axis`.
            position_scale_nm: multiplier turning a position column into
                nanometres. 1.0 for a format that stores nanometres; the pixel
                size for one that stores camera pixels.
            sigma_columns: axis name -> fitted-width column name. Defaults to
                :data:`DEFAULT_SIGMA_COLUMNS`. Axes whose column is absent are
                simply not present; see :meth:`has_sigma_axis`.
            sigma_scale_nm: multiplier turning a sigma column into nanometres.
                **Defaults to following *position_scale_nm***, because a format
                that fits widths in a different unit from the one it stores
                positions in does not exist among the readers here, and one
                unit decision per dataset is the property worth keeping. Pass a
                number to state it independently.
            photon_column: photon-count column name, defaulting to
                :data:`DEFAULT_PHOTON_COLUMN`. Carries no length unit, so it
                takes no scale.
            copy: copy *records* on the way in. The default is the safe one --
                the table then owns its data outright and no caller can mutate
                it from the outside. Pass False only when the caller has just
                produced the array and will not touch it again.
        """
        self._position_columns = dict(
            DEFAULT_POSITION_COLUMNS if position_columns is None else position_columns
        )
        self._position_scale_nm = float(position_scale_nm)
        self._sigma_columns = dict(
            DEFAULT_SIGMA_COLUMNS if sigma_columns is None else sigma_columns
        )
        self._sigma_scale_nm = (
            None if sigma_scale_nm is None else float(sigma_scale_nm)
        )
        self._photon_column = (
            DEFAULT_PHOTON_COLUMN if photon_column is None else photon_column
        )
        self._records = None
        self._records_view = None
        self._filter_mask = None
        self._display_mask = None
        self._n_filtered = 0
        self._n_active = 0
        # Keyed by ``("pos", axis)``, ``("sigma", axis)`` and ``("photons",)``
        # rather than by axis alone: every derived column wants the same
        # three-stage caching, and one mechanism is cheaper to keep correct
        # than three that have to be invalidated in step.
        self._all_nm_cache = {}
        self._filtered_nm_cache = {}
        self._active_nm_cache = {}
        self._filtered_records_cache = None
        self._active_records_cache = None
        self.set_records(records, copy=copy)

    # ------------------------------------------------------------------
    # Canonical records
    # ------------------------------------------------------------------

    def set_records(self, records, *, copy=True):
        """Replace the canonical records and reset the active set to all rows."""
        if records is None:
            self._records = None
            self._filter_mask = None
            self._display_mask = None
            self._n_filtered = 0
            self._n_active = 0
        else:
            self._records = records.copy() if copy else records
            self._filter_mask = _readonly(np.ones(len(self._records), dtype=bool))
            self._display_mask = None
            self._n_filtered = len(self._records)
            self._n_active = len(self._records)
        self.invalidate_caches()

    @property
    def records(self):
        """Read-only view of the canonical table.

        Read-only rather than merely documented as such: a write here would not
        only corrupt the canonical rows, it would leave every derived
        coordinate column cached against the old values, because nothing told
        the table anything had changed. Use :meth:`adjust_column`, which is the
        one writer that also invalidates.
        """
        if self._records is None:
            return None
        if self._records_view is None:
            view = self._records[:]
            view.flags.writeable = False
            self._records_view = view
        return self._records_view

    @property
    def field_names(self):
        if self._records is None or self._records.dtype.names is None:
            return ()
        return tuple(self._records.dtype.names)

    def has_field(self, name):
        return name in self.field_names

    def has_axis(self, axis):
        """True when *axis* has a position column present in the records."""
        column = self._position_columns.get(axis)
        return column is not None and self.has_field(column)

    def __len__(self):
        return 0 if self._records is None else len(self._records)

    def column(self, name):
        """One canonical column, by field name.  Read-only, like the records."""
        if self._records is None:
            raise ValueError("this table holds no localizations")
        return _readonly(np.asarray(getattr(self._records, name)))

    def set_column(self, name, values):
        """Replace one canonical column, and drop the caches derived from it.

        Together with :meth:`adjust_column` this is the whole sanctioned write
        surface. Reaching into :attr:`records` instead is refused, because a
        write there leaves the coordinate caches holding values that no longer
        exist in the table.
        """
        values = np.asarray(values)
        if len(values) != len(self):
            raise ValueError(
                f"column {name!r} needs {len(self)} values, got {len(values)}"
            )
        setattr(self._records, name, values)
        self.invalidate_caches()

    def adjust_column(self, name, offset=0.0, scale=1.0):
        """Apply ``column * scale + offset`` in place, and drop derived caches.

        The only sanctioned writer of :attr:`records` after construction. It
        exists so that cache invalidation lives in one place rather than at
        every call site that happens to know it changed the table.
        """
        setattr(self._records, name, self.column(name) * scale + offset)
        self.invalidate_caches()

    # ------------------------------------------------------------------
    # Active subset
    # ------------------------------------------------------------------

    @property
    def filter_mask(self):
        """What the user selected: the mask everything outside the GPU sees."""
        return self._filter_mask

    @property
    def active_mask(self):
        """What gets drawn: :attr:`filter_mask` narrowed by the display limit."""
        if self._display_mask is None:
            return self._filter_mask
        return self._display_mask

    @property
    def n_filtered(self):
        return self._n_filtered

    @property
    def n_active(self):
        return self._n_active

    @property
    def is_display_limited(self):
        """True when the renderer is showing fewer rows than the user selected."""
        return self._display_mask is not None

    @property
    def filtered_ids(self):
        """Stable IDs -- canonical row indices -- of the user-selected rows."""
        if self._filter_mask is None:
            return np.empty(0, dtype=np.intp)
        return np.flatnonzero(self._filter_mask)

    @property
    def active_ids(self):
        """Stable IDs of the rows actually drawn."""
        mask = self.active_mask
        if mask is None:
            return np.empty(0, dtype=np.intp)
        return np.flatnonzero(mask)

    def set_filter_mask(self, mask):
        """Replace the user's selection.

        Clears any display limit: the budget is a function of how much data is
        selected, so a new selection invalidates the old thinning. The caller
        re-applies :meth:`limit_active_to` afterwards if it still needs to.
        """
        if self._records is None:
            raise ValueError("cannot filter a table that holds no localizations")
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (len(self._records),):
            raise ValueError(
                "filter mask must be a boolean array with one entry per "
                f"localization ({len(self._records)})"
            )
        # Copied: keeping the caller's array would let them desynchronise the
        # mask from the counts derived here simply by writing to it later.
        self._filter_mask = _readonly(mask.copy())
        self._n_filtered = int(np.count_nonzero(mask))
        self._display_mask = None
        self._n_active = self._n_filtered
        self._invalidate_selection_caches()

    def reset(self):
        """Make every localization active again, without copying any rows."""
        if self._records is None:
            return
        self.set_filter_mask(np.ones(len(self._records), dtype=bool))

    def _masked_records(self, mask, n_selected):
        if self._records is None:
            return None
        if n_selected == len(self._records):
            rows = self._records[:]  # a view; nothing is excluded
        else:
            rows = self._records[mask]
        # Read-only on purpose: a write here would either corrupt the canonical
        # table (while nothing is excluded, this is a view of it) or silently
        # diverge from it.  Both are worse than an exception.
        rows.flags.writeable = False
        return rows

    @property
    def filtered_records(self):
        """Read-only view of the rows the user selected."""
        if self._filtered_records_cache is None:
            self._filtered_records_cache = self._masked_records(
                self._filter_mask, self._n_filtered
            )
        return self._filtered_records_cache

    @property
    def active_records(self):
        """Read-only view of the rows actually drawn."""
        if self._display_mask is None:
            return self.filtered_records
        if self._active_records_cache is None:
            self._active_records_cache = self._masked_records(
                self._display_mask, self._n_active
            )
        return self._active_records_cache

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    @property
    def position_scale_nm(self):
        """Multiplier from position-column units to nanometres."""
        return self._position_scale_nm

    @position_scale_nm.setter
    def position_scale_nm(self, value):
        value = float(value)
        if value != self._position_scale_nm:
            self._position_scale_nm = value
            self.invalidate_caches()

    @property
    def sigma_scale_nm(self):
        """Multiplier from sigma-column units to nanometres.

        Follows :attr:`position_scale_nm` unless one was stated explicitly, so
        a reader that updates its pixel size updates both together rather than
        leaving widths converted with the old value.
        """
        if self._sigma_scale_nm is None:
            return self._position_scale_nm
        return self._sigma_scale_nm

    @sigma_scale_nm.setter
    def sigma_scale_nm(self, value):
        value = None if value is None else float(value)
        if value != self._sigma_scale_nm:
            self._sigma_scale_nm = value
            self.invalidate_caches()

    def position_column(self, axis):
        column = self._position_columns.get(axis)
        if column is None or not self.has_field(column):
            raise KeyError(f"this table has no {axis!r} position column")
        return column

    def sigma_column(self, axis):
        column = self._sigma_columns.get(axis)
        if column is None or not self.has_field(column):
            raise KeyError(f"this table has no {axis!r} sigma column")
        return column

    def has_sigma_axis(self, axis):
        """True when *axis* has a fitted-width column present in the records."""
        column = self._sigma_columns.get(axis)
        return column is not None and self.has_field(column)

    def sigma_column_name(self, axis):
        """The sigma column name configured for *axis*, present or not.

        For naming a column in an error message, where refusing to answer
        because the column is missing would defeat the point.
        """
        return self._sigma_columns.get(axis) or f"sigma_{axis}"

    def has_photons(self):
        """True when the photon-count column is present in the records."""
        return self.has_field(self._photon_column)

    # -- the three caching stages, shared by every derived column ------

    def _canonical(self, key):
        """Cached derived column over every canonical row."""
        cached = self._all_nm_cache.get(key)
        if cached is None:
            cached = _readonly(self._compute(key))
            self._all_nm_cache[key] = cached
        return cached

    def _filtered(self, key):
        """Cached derived column over the user-selected rows."""
        cached = self._filtered_nm_cache.get(key)
        if cached is None:
            values = self._canonical(key)
            if self._n_filtered == len(values):
                cached = values
            else:
                cached = _readonly(values[self._filter_mask])
            self._filtered_nm_cache[key] = cached
        return cached

    def _active(self, key):
        """Cached derived column over the rows actually drawn."""
        if self._display_mask is None:
            return self._filtered(key)
        cached = self._active_nm_cache.get(key)
        if cached is None:
            cached = _readonly(self._canonical(key)[self._display_mask])
            self._active_nm_cache[key] = cached
        return cached

    def _compute(self, key):
        kind = key[0]
        if kind == "photons":
            # No length unit, so no conversion -- only the dtype normalization
            # every other derived column gets.
            return np.asarray(self.column(self._photon_column), dtype=COORDINATE_DTYPE)
        if kind == "pos":
            values = self.column(self.position_column(key[1]))
            scale = self._position_scale_nm
        else:
            values = self.column(self.sigma_column(key[1]))
            scale = self.sigma_scale_nm
        if scale == 1.0:
            return np.asarray(values, dtype=COORDINATE_DTYPE)
        return np.asarray(values * scale, dtype=COORDINATE_DTYPE)

    # -- positions -----------------------------------------------------

    def coordinate_nm(self, axis):
        """Cached nanometre column for *axis*, over every canonical row."""
        return self._canonical(("pos", axis))

    def filtered_coordinate_nm(self, axis):
        """Cached nanometre column for *axis*, over the user-selected rows.

        This is the one an export reads: it ignores display thinning.
        """
        return self._filtered(("pos", axis))

    def active_coordinate_nm(self, axis):
        """Cached nanometre column for *axis*, over the rows actually drawn.

        Cached because interactive paths read the same column several times per
        gesture. Before this, a range-slider drag on a pixel-unit dataset
        multiplied the whole column by the pixel size eight times over.
        """
        return self._active(("pos", axis))

    # -- fitted widths and photon counts -------------------------------

    def sigma_nm(self, axis):
        """Cached nanometre fitted width for *axis*, over every canonical row."""
        return self._canonical(("sigma", axis))

    def filtered_sigma_nm(self, axis):
        """Fitted width for *axis* over the user-selected rows, in nanometres."""
        return self._filtered(("sigma", axis))

    def active_sigma_nm(self, axis):
        """Fitted width for *axis* over the rows actually drawn, in nanometres."""
        return self._active(("sigma", axis))

    def photons(self):
        """Photon counts over every canonical row."""
        return self._canonical(("photons",))

    def filtered_photons(self):
        """Photon counts over the user-selected rows."""
        return self._filtered(("photons",))

    def active_photons(self):
        """Photon counts over the rows actually drawn."""
        return self._active(("photons",))

    def selection(self, kind=ACTIVE):
        """A uniform accessor for one of the two row selections.

        Defaults to :data:`ACTIVE` so existing render paths keep their meaning;
        an export passes :data:`FILTERED`.
        """
        return TableSelection(self, kind)

    def invalidate_caches(self):
        """Drop every derived array.  Call after :attr:`records` changed."""
        self._records_view = None
        self._all_nm_cache = {}
        self._invalidate_selection_caches()

    def _invalidate_selection_caches(self):
        self._filtered_records_cache = None
        self._active_records_cache = None
        self._filtered_nm_cache = {}
        self._active_nm_cache = {}

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def resolve_property(self, prop, low=-np.inf, high=np.inf):
        """Map a property name and its bounds onto a canonical column.

        Bounds may be quoted in nanometres for a column that is not stored in
        nanometres; this is where that conversion happens, so no caller needs to
        know which unit a given format used.

        Returns ``(values, low, high)``.
        """
        if self.has_field(prop):
            return self.column(prop), low, high
        for axis, column in self._position_columns.items():
            if prop == f"{axis}_pos_nm" and self.has_field(column):
                scale = self._position_scale_nm
                return self.column(column), low / scale, high / scale
        # Formats that store pixels name their other columns the same way, so a
        # nanometre-quoted bound on any of them converts identically.
        if prop.endswith("_nm"):
            pixel_name = prop[: -len("_nm")] + "_pixels"
            if self.has_field(pixel_name):
                scale = self._position_scale_nm
                return self.column(pixel_name), low / scale, high / scale
        raise KeyError(f"no column {prop!r} in this table")

    def mask_for_property(self, prop, low, high):
        """Boolean mask over the canonical rows with ``low <= prop <= high``."""
        values, low, high = self.resolve_property(prop, low, high)
        # Written as the negated exclusion test rather than the more obvious
        # ``(values >= low) & (values <= high)``: the two disagree on NaN, and
        # this form keeps non-finite rows instead of dropping them.
        return np.invert((values < low) | (values > high))

    def indices_for_property(self, prop, low, high):
        return np.flatnonzero(self.mask_for_property(prop, low, high))

    def apply_filters(self, keep, remove=None):
        """Set the active mask from a keep set minus a remove set.

        Args:
            keep: a boolean mask over the canonical rows, or indices into them.
            remove: indices to deactivate afterwards, or None.

        Both contributions are written into one boolean array. The earlier form
        built the complement with ``np.delete(np.arange(n), keep)``, concatenated
        and uniquified it, then copied the record array with a second
        ``np.delete`` -- two full copies and a sort per filter change.
        """
        n_rows = len(self)
        keep = np.asarray(keep)
        if keep.dtype == np.bool_ and keep.shape == (n_rows,):
            mask = keep.copy()
        else:
            mask = np.zeros(n_rows, dtype=bool)
            keep = keep.ravel()
            if keep.size:
                mask[keep.astype(np.intp, copy=False)] = True
        if remove is not None:
            remove = np.asarray(remove).ravel()
            if remove.size:
                mask[remove.astype(np.intp, copy=False)] = False
        self.set_filter_mask(mask)

    def bandpass(self, prop, low=-np.inf, high=np.inf):
        """Narrow the selection to rows with ``low <= prop <= high``."""
        if low == -np.inf and high == np.inf:
            raise ValueError("Nothing to filter here")
        self.set_filter_mask(self._filter_mask & self.mask_for_property(prop, low, high))

    def keep_values(self, prop, values):
        """Narrow the selection to rows whose *prop* is one of *values*."""
        column, _, _ = self.resolve_property(prop)
        self.set_filter_mask(self._filter_mask & np.isin(column, np.atleast_1d(values)))

    def deactivate_positions(self, positions):
        """Deselect rows by their position *within the current selection*.

        Positions index the filter set, not the possibly-thinned display set:
        this is a user-level edit of what they picked, so it must not depend on
        how much of it the renderer happened to be showing.
        """
        positions = np.asarray(positions).ravel()
        if positions.size == 0:
            return
        mask = np.array(self._filter_mask, copy=True)
        mask[np.flatnonzero(mask)[positions.astype(np.intp, copy=False)]] = False
        self.set_filter_mask(mask)

    def restrict_by_percent(self, ranges_pc):
        """Narrow each axis to a percentage window of its current extent.

        Args:
            ranges_pc: axis name -> ``(low_percent, high_percent)``. Axes that
                are absent from the table, or mapped to None, are skipped.

        Every bound is derived from the extent *before* any of them is applied,
        so the axes cannot influence one another.
        """
        bounds = {}
        for axis, range_pc in ranges_pc.items():
            if range_pc is None or not self.has_axis(axis):
                continue
            # Measured against the selection, not the display subsample: where
            # a slider lands must not depend on how much the GPU could afford.
            coords = self.filtered_coordinate_nm(axis)
            if coords.size == 0:
                continue
            low = float(np.min(coords))
            span = float(np.max(coords)) - low
            bounds[axis] = (
                range_pc[0] / 100 * span + low,
                range_pc[1] / 100 * span + low,
            )
        for axis in AXES:
            if axis in bounds:
                self.bandpass(f"{axis}_pos_nm", *bounds[axis])

    def exclude_non_finite_positions(self):
        """Deselect rows whose position is NaN or infinite.  Returns the count.

        A NaN coordinate is not a measurement, and left selected it becomes a
        NaN vertex -- which poisons the bounding box, the camera framing and
        the mesh handed to the GPU, for every other localization too. The rows
        stay in the canonical table; they are simply not drawn and not
        exported.
        """
        if self._records is None:
            return 0
        columns = [
            self.coordinate_nm(axis) for axis in AXES if self.has_axis(axis)
        ]
        if not columns:
            return 0
        bad = non_finite_mask(*columns)
        n_bad = int(np.count_nonzero(bad & self._filter_mask))
        if n_bad:
            self.set_filter_mask(self._filter_mask & ~bad)
        return n_bad

    def limit_active_to(self, max_active):
        """Show at most *max_active* of the selected rows, evenly spaced.

        Returns the number of rows hidden. This sets the **display limit**: the
        selection is untouched, so an export or a saved dataset still sees
        everything the user asked for. Sampling is by stride, which is
        deterministic and preserves spatial coverage rather than truncating to
        one corner of the field of view.

        Passing a limit at or above the selection size clears any existing
        limit, so a budget that stops binding restores the full view.
        """
        if self._records is None or max_active is None:
            self._clear_display_limit()
            return 0
        max_active = int(max_active)
        if max_active < 1 or self._n_filtered <= max_active:
            self._clear_display_limit()
            return 0
        hidden = self._n_filtered - max_active
        stride = int(np.ceil(self._n_filtered / max_active))
        kept = self.filtered_ids[::stride][:max_active]
        mask = np.zeros(len(self._records), dtype=bool)
        mask[kept] = True
        self._display_mask = _readonly(mask)
        self._n_active = int(np.count_nonzero(mask))
        self._invalidate_selection_caches()
        return hidden

    def _clear_display_limit(self):
        if self._display_mask is None:
            return
        self._display_mask = None
        self._n_active = self._n_filtered
        self._invalidate_selection_caches()
