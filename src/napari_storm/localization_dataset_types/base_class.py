import numpy as np

from ..core import (
    DEFAULT_POSITION_COLUMNS,
    ZDIM_PRESENT,
    LocalizationTable,
    MetadataProvider,
)


class LocalizationDataBaseClass:
    """An Object which contains most basic localization data.

    The data itself lives in a :class:`~napari_storm.core.LocalizationTable`,
    which owns the canonical records, the active mask, the coordinate caches and
    every filtering operation, and which depends on nothing but numpy. This
    class is the *application* layer around it: naming, dimensionality, file
    import/export and the dialogs those need.

    The split is the Level 2 boundary from ``docs/modernization-review.md``.
    Subclasses declare *where* their coordinates live -- which columns, in which
    unit -- and inherit the mechanics rather than reimplementing them.
    """

    #: Axis name -> column name in this format's records.
    POSITION_COLUMNS = DEFAULT_POSITION_COLUMNS

    #: Class-level default so the accessors below answer sensibly on a subclass
    #: that skips __init__ before assigning locs_all -- several importers
    #: construct an empty instance and fill it in.
    _table = None

    def __init__(
        self,
        locs=None,
        name=None,
        zdim_present=False,
    ):
        self.dataset_type = "LocalizationDataBaseClass"
        self._table = None
        if locs is None:
            self.locs_dtype = None
            self.name = None
            self.zdim_present = None
        else:
            self.locs_dtype = self.init_dtype(zdim_present)

            if name is None:
                name = "Untitled"

            self.name = name
            # Copy once, here.  Callers must not be able to mutate a dataset by
            # keeping a reference to the array they passed in.
            self.locs_all = locs.copy()
            self.zdim_present = zdim_present

    # ------------------------------------------------------------------
    # The canonical table
    # ------------------------------------------------------------------

    def position_scale_nm(self):
        """Multiplier from this format's position columns to nanometres."""
        return 1.0

    @property
    def table(self):
        """The underlying :class:`LocalizationTable`, or None before loading."""
        return self._table

    @property
    def locs_all(self):
        """The canonical localization table.  Treat as read-only after load."""
        return None if self._table is None else self._table.records

    @locs_all.setter
    def locs_all(self, value):
        if value is None:
            self._table = None
            return
        # The importers above already copy; copying again would double the peak
        # memory of every load for no benefit.
        self._table = LocalizationTable(
            value,
            position_columns=self.POSITION_COLUMNS,
            position_scale_nm=self.position_scale_nm(),
            copy=False,
        )

    @property
    def locs_active(self):
        """Read-only view of the rows selected by :attr:`active_mask`."""
        return None if self._table is None else self._table.active_records

    @locs_active.setter
    def locs_active(self, value):
        raise AttributeError(
            "locs_active is derived from locs_all and the active mask; "
            "use set_filter_mask(), apply_filters() or reset_filters() instead"
        )

    @property
    def locs(self):
        return self.locs_active

    @property
    def filter_mask(self):
        """What the user selected, before any display limit."""
        return None if self._table is None else self._table.filter_mask

    @property
    def active_mask(self):
        """What is drawn: the selection narrowed by the render budget."""
        return None if self._table is None else self._table.active_mask

    def set_filter_mask(self, mask):
        self._table.set_filter_mask(mask)

    def invalidate_coordinate_cache(self):
        """Drop cached coordinate columns after ``locs_all`` changed."""
        if self._table is not None:
            self._table.invalidate_caches()

    def adjust_column(self, prop, offset=0.0, scale=1.0):
        """Apply ``column * scale + offset`` to one canonical column."""
        self._table.adjust_column(prop, offset=offset, scale=scale)

    def set_column(self, prop, values):
        """Replace one canonical column outright."""
        self._table.set_column(prop, values)

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    def all_coordinate_nm(self, axis):
        return self._table.coordinate_nm(axis)

    def active_coordinate_nm(self, axis):
        """Drawn rows.  For anything leaving the process use the filtered set."""
        return self._table.active_coordinate_nm(axis)

    def filtered_coordinate_nm(self, axis):
        """The rows the user selected, ignoring any render-budget thinning."""
        return self._table.filtered_coordinate_nm(axis)

    @property
    def x_pos_nm(self):
        return self.active_coordinate_nm("x")

    @property
    def y_pos_nm(self):
        return self.active_coordinate_nm("y")

    @property
    def z_pos_nm(self):
        return self.active_coordinate_nm("z")

    @property
    def x_pos_nm_all(self):
        return self.all_coordinate_nm("x")

    @property
    def y_pos_nm_all(self):
        return self.all_coordinate_nm("y")

    @property
    def z_pos_nm_all(self):
        return self.all_coordinate_nm("z")

    def init_dtype(self, zdim_present):
        if zdim_present:
            locs_dtype = [("x_pos_nm", "f4"), ("y_pos_nm", "f4"), ("z_pos_nm", "f4")]
        else:
            locs_dtype = [("x_pos_nm", "f4"), ("y_pos_nm", "f4")]
        return locs_dtype

    def load_ns(self, dataset):
        tmp_name = dataset.attrs["name"]
        tmp_zdim_present = dataset.attrs["zdim_present"]
        return LocalizationDataBaseClass(
            np.rec.array(dataset[...]),
            name=tmp_name,
            zdim_present=tmp_zdim_present,
        )

    def check_if_metadata_is_complete(self, metadata, metadata_provider=None):
        provider = metadata_provider or MetadataProvider()
        if "name" not in metadata:
            metadata["name"] = "Untitled"
        if "zdim_present" not in metadata:
            zdim_present = provider.ask_yes_no(ZDIM_PRESENT, "Is zdim present?")
            if zdim_present is not None:
                assert isinstance(zdim_present, bool)
                metadata["zdim_present"] = zdim_present
        return metadata

    def import_recognized_data(self, data, metadata=None, metadata_provider=None):
        data = np.rec.array(data, metadata["dataset_class_dtype"])
        metadata = LocalizationDataBaseClass().check_if_metadata_is_complete(
            metadata, metadata_provider
        )
        return LocalizationDataBaseClass(
            locs=data, name=metadata["name"], zdim_present=metadata["zdim_present"]
        )

    def save_as_npy(self, filename):
        """Write to *filename*.  Choosing it is the caller's job, not a reader's."""
        if not filename:
            raise ValueError("save_as_npy needs a filename")
        metadata = {
            "dataset_class": LocalizationDataBaseClass,
            "name": self.name,
            "zdim_present": self.zdim_present,
        }
        np.save(filename + ".npy", [self.locs, metadata])

    def load_npy(self, filename):
        if not filename:
            raise ValueError("load_npy needs a filename")
        data = np.load(filename + ".npy")
        self.locs_all = data[0].copy()
        return data[1]["dataset_class"](
            locs=data[0], name=data[1]["name"], zdim_present=data[1]["zdim_present"]
        )

    def locs_sanity_check(self):
        assert isinstance(self.locs, np.recarray), "locs should be numpy rec array"
        assert (
            self.locs.dtype == self.locs_dtype
        ), f"locs should have {self.locs_dtype} as datatype"

    # ------------------------------------------------------------------
    # Filtering -- delegated to the table
    # ------------------------------------------------------------------

    def reset_filters(self):
        """Make every localization active again, without copying any rows."""
        if self._table is not None:
            self._table.reset()

    def apply_filters(self, spatial_keep_indices, parameter_remove_indices):
        """Combine spatial render-range and parameter filters into one mask."""
        self._table.apply_filters(spatial_keep_indices, parameter_remove_indices)

    def exclude_non_finite_positions(self):
        """Deselect rows with a NaN or infinite position.  Returns the count."""
        return 0 if self._table is None else self._table.exclude_non_finite_positions()

    def limit_active_to(self, max_active):
        """Thin the active set to at most *max_active* rows, evenly spaced."""
        if self._table is None:
            return 0
        return self._table.limit_active_to(max_active)

    def get_mask_of_specified_prop_all(self, prop, l_val, u_val):
        return self._table.mask_for_property(prop, l_val, u_val)

    def get_idx_of_specified_prop_all(self, prop, l_val, u_val):
        return self._table.indices_for_property(prop, l_val, u_val)

    def restrict_locs_by_percent(
        self, x_range_pc, y_range_pc, z_range_pc=None, reset=False
    ):
        if reset:
            self.reset_filters()
        self._table.restrict_by_percent(
            {
                "x": x_range_pc,
                "y": y_range_pc,
                "z": z_range_pc if self.zdim_present else None,
            }
        )

    def restrict_locs_by_absolute(
        self, x_range_nm, y_range_nm, z_range_nm=None, reset=False
    ):
        assert not (not self.zdim_present and z_range_nm is not None), (
            "cannot use restrict in z when" " z dimension not present "
        )
        if reset:
            self.reset_filters()

        if self.zdim_present:
            self.bandpass_locs_filter_by_property(
                "z_pos_nm", z_range_nm[0], z_range_nm[1]
            )
        self.bandpass_locs_filter_by_property("x_pos_nm", x_range_nm[0], x_range_nm[1])
        self.bandpass_locs_filter_by_property("y_pos_nm", y_range_nm[0], y_range_nm[1])

    def remove_locs_by_index(self, filter_idx, reset=False):
        """Deselect rows by their position *within the current selection*."""
        if reset:
            self.reset_filters()
        self._table.deactivate_positions(filter_idx)

    def bandpass_locs_filter_by_property(self, prop, l_val=-np.inf, u_val=np.inf):
        self._table.bandpass(prop, l_val, u_val)

    def value_specific_locs_filter_by_property(self, prop, values):
        """Keep only rows whose *prop* equals one of *values*."""
        self._table.keep_values(prop, values)

    def number_of_active_entries(self):
        """How many localizations are drawn."""
        return 0 if self._table is None else self._table.n_active

    def number_of_filtered_entries(self):
        """How many localizations the user's filters left, before any thinning."""
        return 0 if self._table is None else self._table.n_filtered

    @property
    def is_display_limited(self):
        """True when the render budget is hiding rows the user selected."""
        return self._table is not None and self._table.is_display_limited

    def number_of_entries(self):
        return 0 if self._table is None else len(self._table)

    def check_if_imported_data_fits_to_datatype(self, data=None, metadata=None):
        if data is None and metadata is None:
            return -1
        if metadata is not None:
            if type(metadata["dataset_class"]).__name__ == self.__class__.__name__:
                return self
        else:
            if data.dtype == self.dataset_type:
                return self
        return False
