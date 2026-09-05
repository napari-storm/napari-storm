from dataclasses import dataclass

import numpy as np

from .colormap_factory import make_colormaps
from .core import (
    ACTIVE,
    IDENTITY,
    AppearanceChanged,
    DatasetClosed,
    DatasetTraits,
    GaussianSettings,
    MaskChanged,
    RenderPlanner,
    StoreCleared,
    TransformChanged,
)
from .core.renderer import Changed, LayerAppearance
from .CustomErrors import ParentError
from .grid_plane_renderer import GridPlaneRenderer
from .memory_budget import max_localizations_for_budget, render_bytes_for
from .napari_particles.selection import select_renderer
from .ns_constants import AXIS_VIEWS, DEFAULT_AXIS_VIEW, FLAT_DATA_Z_NM
from .scalebar_renderer import ScalebarRenderer


def look_at_plane(camera, plane=DEFAULT_AXIS_VIEW):
    """Point *camera* at the named plane of the (z, y, x) world."""
    view_direction, up_direction = AXIS_VIEWS[plane]
    camera.set_view_direction(view_direction=view_direction, up_direction=up_direction)


@dataclass
class RenderArrays:
    """The per-localization inputs the billboard renderer consumes.

    Derived from a dataset and the current render settings, and held per dataset
    id rather than in three lists indexed alongside the dataset list.
    """

    sigmas: np.ndarray
    size: float
    values: np.ndarray


class DataToLayerInterface:  # localization always with z # switch info with channel controls #
    def __init__(
        self,
        parent,
        viewer,
        render_config=None,
        surface_layer=None,
        renderer=None,
    ):

        # assert isinstance(parent, napari_storm) == True
        self._parent = parent
        self.viewer = viewer
        self.render_config = render_config
        # Callbacks for communicating back to the GUI without direct widget access
        self._on_grid_line_distance_clamped = None  # set via property
        self.on_layer_updated = None  # callable(channel_index: int)
        # callable(message: str) -- resource limits that changed what is drawn
        self.on_resource_limit_applied = None
        # Ids of datasets currently thinned/clamped, so a limit is reported when
        # it starts applying rather than on every slider tick.
        self._budgeted_datasets = set()
        self._clamped_splats = set()
        # (dataset_id, column) pairs already reported as repaired, so a bad
        # column is mentioned once rather than on every slider tick.
        self._repaired_columns = set()
        self.colormap, self.colormap_icons = make_colormaps()
        self._sbr = ScalebarRenderer(viewer, render_config)
        if surface_layer:
            self._sbr.scalebar_layer = surface_layer
            self._sbr.scalebar_exists = True
        self._gpr = GridPlaneRenderer(viewer, render_config)
        # The localization backend.  Everything on this side of it decides
        # *what* to draw; everything behind it owns GPU resources and host layer
        # objects.  Still injectable -- that seam is what let Level 3 measure
        # three implementations on the same fixtures -- but the default is now
        # the one that won that comparison, falling back when the GL backend
        # this session got cannot instance.  See `selection.py`.
        self.renderer = renderer or select_renderer(viewer)
        # A user deleting our layer in napari's own layer list means they want
        # that dataset gone.  Honour it rather than keeping half a session.
        if hasattr(self.renderer, "on_layer_removed_by_host"):
            self.renderer.on_layer_removed_by_host = self._on_layer_removed_by_host

        # Renderer inputs, keyed by stable dataset id rather than by position.
        # These were three parallel lists indexed alongside the dataset list, so
        # unloading a dataset meant popping the right index out of each of them
        # from code that had no way to know they existed (§4.2: "renderer
        # resources are keyed by stable dataset IDs, not layer names or
        # positions in parallel lists").
        self.render_state = {}
        self.render_anti_alias = 0
        # Planning -- what to draw -- is host-free and lives in core.  This
        # class is now the adapter between the Qt-configured settings and that.
        self.planner = RenderPlanner(on_repaired=self._report_repaired_column)
        self._planning_for = None

        self.render_range_x = [np.inf, -np.inf]
        self.render_range_y = [np.inf, -np.inf]
        self.render_range_z = [np.inf, -np.inf]
        self.camera = [
            self.viewer.camera.zoom,
            self.viewer.camera.center,
            self.viewer.camera.angles,
        ]

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        raise ParentError("Cannot change parent of existing Widget")

    @property
    def localization_datasets(self):
        return self.parent.localization_datasets

    @localization_datasets.setter
    def localization_datasets(self, value):
        raise ParentError("Cannot change parent's attribute from here")

    @property
    def on_grid_line_distance_clamped(self):
        return self._gpr.on_clamped

    @on_grid_line_distance_clamped.setter
    def on_grid_line_distance_clamped(self, value):
        self._on_grid_line_distance_clamped = value
        self._gpr.on_clamped = value

    @property
    def current_grid_plane_color(self):
        return self._gpr.current_grid_plane_color

    @current_grid_plane_color.setter
    def current_grid_plane_color(self, value):
        self._gpr.current_grid_plane_color = value

    def create_remove_grid_plane_state(self, enable):
        self._gpr.create_remove(
            enable, self.render_range_x, self.render_range_y, self.render_range_z
        )

    def update_grid_plane(
        self,
        z_pos=None,
        line_thickness=None,
        line_distance_nm=None,
        color=None,
        opacity=None,
    ):
        self._gpr.update(
            self.render_range_x,
            self.render_range_y,
            self.render_range_z,
            z_pos=z_pos,
            line_thickness=line_thickness,
            line_distance_nm=line_distance_nm,
            color=color,
            opacity=opacity,
        )

    def reset_render_range_and_offset(self):
        self.render_range_x = [np.inf, -np.inf]
        self.render_range_y = [np.inf, -np.inf]
        self.render_range_z = [np.inf, -np.inf]

    def on_store_event(self, event):
        """React to what the store says happened.

        Subscribed rather than called: unloading a dataset, recolouring one or
        moving one in world space no longer requires whoever did it to know
        that a renderer exists.
        """
        if isinstance(event, DatasetClosed):
            self.renderer.close(event.dataset_id)
            self.remove_dataset_state(event.dataset_id)
        elif isinstance(event, StoreCleared):
            self.renderer.close_all()
            self.clear_dataset_state()
        elif isinstance(event, AppearanceChanged):
            if self.renderer.is_open(event.dataset_id):
                self.renderer.set_appearance(event.dataset_id, event.appearance)
        elif isinstance(event, (MaskChanged, TransformChanged)):
            dataset = self._dataset_for(event.dataset_id)
            if dataset is not None:
                self.refresh_dataset(dataset)

    def _dataset_for(self, dataset_id):
        store = getattr(self.parent, "dataset_store", None)
        return None if store is None else store.get(dataset_id)

    def _channel_index_of(self, dataset):
        datasets = self.parent.localization_datasets
        return next((i for i, other in enumerate(datasets) if other is dataset), -1)

    def _report_repaired_column(self, column, n_repaired, n_total):
        """The planner found unusable uncertainty or photon values."""
        key = (self._planning_for, column)
        if key in self._repaired_columns:
            return
        self._repaired_columns.add(key)
        self._report_resource_limit(
            f"{column}: {n_repaired:,} of {n_total:,} values were zero, "
            f"negative or non-finite and were substituted with the smallest "
            f"usable value. Those localizations are drawn at the smallest "
            f"usable size, not omitted."
        )

    def _on_layer_removed_by_host(self, dataset_id):
        """The user deleted a localization layer through napari."""
        if getattr(self.parent, "_closed", False):
            # The dock is already tearing down; the viewer is emptying its own
            # layer list and there is no session left to keep consistent.
            self.remove_dataset_state(dataset_id)
            return
        store = getattr(self.parent, "dataset_store", None)
        dataset = None if store is None else store.get(dataset_id)
        if dataset is None:
            self.remove_dataset_state(dataset_id)
            return
        # Route it through the widget's own unload so the channel controls,
        # filter entries and info card go with it, exactly as if the Unload
        # button had been pressed.
        self.parent.unload_dataset(dataset)

    def close(self):
        """Release every renderer resource this interface owns."""
        self.renderer.close_all()
        detach = getattr(self.renderer, "detach", None)
        if detach is not None:
            detach()
        self.clear_dataset_state()

    def clear_dataset_state(self):
        """Release all renderer bookkeeping associated with localization data."""
        self.render_state.clear()
        self._forget_resource_limits()
        self.reset_render_range_and_offset()

    def remove_dataset_state(self, dataset_id):
        """Release the renderer inputs owned by one unloaded dataset."""
        self.render_state.pop(dataset_id, None)
        self._budgeted_datasets.discard(dataset_id)
        self._clamped_splats.discard(dataset_id)

    def _forget_resource_limits(self):
        """Allow limits to be reported again once the dataset set changes.

        The budget is shared, so unloading a dataset changes every remaining
        share; a warning that fired under the old split is worth repeating
        under the new one.
        """
        self._budgeted_datasets.clear()
        self._clamped_splats.clear()
        self._repaired_columns.clear()

    @property
    def n_layers(self):
        """How many datasets the renderer holds state for."""
        return len(self.render_state)

    def set_render_range_and_offset(self):
        self.reset_render_range_and_offset()
        for dataset in self.parent.localization_datasets:
            coords = self.get_coords_from_all_locs(dataset=dataset)
            self.set_render_range(zdim=dataset.zdim_present, coords=coords)
        self.parent.move_camera_center_to_render_range_center()

    @staticmethod
    def percent_to_absolute(axis_range, percent_pair):
        """Map a [low%, high%] slider pair onto an axis extent, in nanometres.

        Localizations are rendered in true physical coordinates, so an axis does
        not necessarily start at zero.  The general form is

            absolute = minimum + percent / 100 * (maximum - minimum)

        This previously read ``percent / 100 * maximum - offset``, which is the
        same expression specialised to a data set whose minimum has been
        translated to zero -- the job the auto-offset used to do.  Written this
        way no translation of the data is required.
        """
        low, high = axis_range[0], axis_range[1]
        if not np.isfinite(low) or not np.isfinite(high):
            return np.asarray(percent_pair, dtype=float)
        span = high - low
        return low + np.asarray(percent_pair, dtype=float) / 100.0 * span

    # ------------------------------------------------------------------
    # Resource limits (P0-04)
    # ------------------------------------------------------------------

    def _report_resource_limit(self, message):
        if self.on_resource_limit_applied is not None:
            self.on_resource_limit_applied(message)

    def _budget_share_mb(self):
        """Megabytes this dataset may spend on host-side render arrays.

        The configured budget is shared equally between the loaded datasets.
        A dataset's share is applied when that dataset is created or updated;
        loading another one deliberately does *not* re-thin the datasets that
        are already on screen, because rebuilding a settled layer is the
        broad-reconstruction behaviour P0-01 exists to remove.
        """
        budget_mb = getattr(self.render_config, "render_budget_mb", 0)
        if not budget_mb:
            return 0
        n_datasets = max(1, len(self.parent.localization_datasets))
        return float(budget_mb) / n_datasets

    def apply_memory_budget(self, dataset):
        """Thin *dataset*'s active set until it fits the render budget.

        Returns the number of localizations dropped.  Over-budget datasets are
        rendered as a uniform subsample rather than being allowed to allocate
        past the limit, which is the "warn and select a lower-cost
        representation" behaviour Level 1 asks for in place of a crash.
        """
        max_active = max_localizations_for_budget(self._budget_share_mb())
        hidden = dataset.limit_active_to(max_active)
        if hidden:
            drawn = dataset.number_of_active_entries()
            selected = dataset.number_of_filtered_entries()
            if dataset.dataset_id not in self._budgeted_datasets:
                self._budgeted_datasets.add(dataset.dataset_id)
                self._report_resource_limit(
                    f"{dataset.name}: drawing {drawn:,} of {selected:,} "
                    f"localizations to stay within the "
                    f"{self._budget_share_mb():.0f} MB render budget "
                    f"({render_bytes_for(selected) / 1e6:.0f} MB required). "
                    f"The dataset itself is untouched -- filtering and export "
                    f"still see all {selected:,}. "
                    f"Raise the budget with $NAPARI_STORM_RENDER_BUDGET_MB."
                )
        else:
            self._budgeted_datasets.discard(dataset.dataset_id)
        return hidden

    def _field_of_view_nm(self):
        """Largest finite world-space span across the current render ranges."""
        spans = []
        for axis_range in (
            self.render_range_x,
            self.render_range_y,
            self.render_range_z,
        ):
            low, high = axis_range[0], axis_range[1]
            if np.isfinite(low) and np.isfinite(high):
                spans.append(high - low)
        spans = [span for span in spans if span > 0]
        return max(spans) if spans else None

    def _splat_size_limit(self):
        """The largest billboard edge the screen-space budget allows, or None."""
        fraction = getattr(self.render_config, "max_splat_fraction_of_fov", 0)
        field_of_view = self._field_of_view_nm()
        if not fraction or field_of_view is None or not field_of_view > 0:
            return None
        return float(field_of_view) * float(fraction)

    def _note_clamped_splat(self, dataset, size_nm):
        """Say once when the screen-space cap started binding for a dataset."""
        limit = self._splat_size_limit()
        clamped = limit is not None and size_nm >= limit
        if clamped:
            if dataset.dataset_id not in self._clamped_splats:
                self._clamped_splats.add(dataset.dataset_id)
                fraction = self.render_config.max_splat_fraction_of_fov
                self._report_resource_limit(
                    f"{dataset.name}: Gaussian size clamped to {size_nm:,.0f} nm, "
                    f"{fraction:.0%} of the field of view. Larger splats cost "
                    "fragment throughput without adding visible detail."
                )
        else:
            self._clamped_splats.discard(dataset.dataset_id)

    def set_render_range(self, zdim, coords):
        """Accumulate the world-space extent of *coords* into the render ranges.

        ``coords`` columns are ordered (z, y, x), napari's own.  render_range_x
        always holds the x extent and render_range_y the y extent, for both 2-D
        and 3-D data.

        The 2-D branch used to store these swapped -- render_range_x took the
        column holding y.  Consumers then disagreed about which convention
        applied: the camera centring compensated for the swap and so was wrong
        in 3-D, while the range filtering and the preview box did not
        compensate and so were wrong in 2-D.  Both were masked while the
        auto-offset started every axis at zero and fields of view were roughly
        square.
        """
        coords = np.asarray(coords)
        if coords.size == 0:
            return
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3) in (z, y, x) order")

        self.render_range_x[1] = max(np.max(coords[:, 2]), self.render_range_x[1])
        self.render_range_y[1] = max(np.max(coords[:, 1]), self.render_range_y[1])
        self.render_range_x[0] = min(np.min(coords[:, 2]), self.render_range_x[0])
        self.render_range_y[0] = min(np.min(coords[:, 1]), self.render_range_y[0])
        if zdim:
            self.render_range_z[1] = max(np.max(coords[:, 0]), self.render_range_z[1])
            self.render_range_z[0] = min(np.min(coords[:, 0]), self.render_range_z[0])

    def create_new_layer(self, dataset, merge=False, layer_name="SMLM Data", idx=-1):
        """Creating a Particle Layer"""
        # The extent has to be known before the budget is applied, because
        # thinning changes which localizations exist but not where they are.
        self.set_render_range(
            coords=self.get_coords_from_all_locs(dataset=dataset),
            zdim=dataset.zdim_present,
        )
        if merge:
            self.set_render_range_and_offset()
            dataset.restrict_locs_by_percent(
                self.render_config.range_x_percent,
                self.render_config.range_y_percent,
                self.render_config.range_z_percent,
            )
        n_non_finite = dataset.exclude_non_finite_positions()
        if n_non_finite:
            self._report_resource_limit(
                f"{dataset.name}: {n_non_finite:,} localizations have a "
                f"non-finite position and are not drawn. A NaN coordinate is "
                f"not a measurement, and one left in the mesh would corrupt "
                f"the extent and camera framing of every other layer."
            )
        self.apply_memory_budget(dataset)
        # _render_request computes the sigma and value arrays.  Computing them
        # here as well allocated a second (N, 3) float32 sigma array and a
        # second (N,) value array per load -- 60 MB of avoidable peak on the 5M
        # fixture, on exactly the datasets most likely to run out of memory.
        dataset.name = layer_name
        # colormap[-1] is a placeholder: the channel control applies the user's
        # actual choice immediately after the layer is created.
        self.renderer.open(
            dataset.dataset_id,
            self._render_request(dataset, name=layer_name, colormap=self.colormap[-1]),
        )

        # add_layer already frames a newly inserted layer.  Camera recentering
        # after range/filter changes is handled explicitly by the widget using
        # normalized (x, y, z) render ranges.
        self.viewer.camera.perspective = 50
        look_at_plane(self.viewer.camera)
        self.camera = [
            self.viewer.camera.zoom,
            self.viewer.camera.center,
            self.viewer.camera.angles,
        ]

    def _refresh_layer(self, dataset, channel_index, extend_range=False):
        """Push the current selection and settings into an existing layer.

        The single path by which a loaded dataset's rendered content changes.
        It updates the buffers of the layer that is already there; it does not
        remove or recreate one.
        """
        if not self.renderer.is_open(dataset.dataset_id):
            return
        if dataset.number_of_active_entries() == 0:
            # Nothing left to draw.  Hide it rather than handing the renderer an
            # empty mesh, and leave the layer in place so it can come back.
            self.renderer.set_visible(dataset.dataset_id, False)
            return

        if extend_range:
            self.set_render_range(
                dataset.zdim_present, self.get_coords_from_locs(dataset)
            )
        # A filter or range change alters the selection and therefore every
        # per-localization array; an appearance-only refresh does not move any
        # localization, so the positions are unchanged.
        changed = (
            Changed.EVERYTHING if extend_range else (Changed.SIGMAS | Changed.VALUES)
        )
        self.renderer.update(
            dataset.dataset_id, self._render_request(dataset, changed=changed)
        )
        if self.on_layer_updated:
            self.on_layer_updated(channel_index)

    def refresh_dataset(self, dataset):
        """Re-apply this dataset's filters and redraw it, and only it.

        The targeted update of §4.1: a change to one dataset's selection has no
        bearing on any other, and rebuilding every layer to honour it was the
        broad signalling the plan set out to remove.
        """
        self.update_data_range(dataset)
        self._refresh_layer(dataset, self._channel_index_of(dataset), extend_range=True)

    def update_layers(self, aas=0, layer_name="SMLM Data"):
        """Re-apply the render range and parameter filters to every layer.

        P0-01: this used to close each layer, remove it from the viewer, build a
        fresh ``Particles`` and add it back -- for a filter change, a slider
        drag, or a reset. Every rebuild tore down and re-registered the layer's
        VisPy visual, shader filters and three layer-list callbacks, and on the
        5M fixture that teardown dominated the update.

        ``Particles.update_particle_data`` rebuilds the billboard geometry from
        whatever coordinates it is given, including a different number of them,
        so a filter change needs new *buffers* but not a new *layer*.
        """
        v = self.viewer
        self.camera = [v.camera.zoom, v.camera.center, v.camera.angles]
        for channel_index, dataset in enumerate(self.parent.localization_datasets):
            self.update_data_range(dataset)
            self._refresh_layer(dataset, channel_index, extend_range=True)
        v.camera.angles = self.camera[2]
        v.camera.zoom = self.camera[0]
        v.camera.center = self.camera[1]
        v.camera.update({})

    def update_layer_appearance(self):
        """Apply Gaussian/value changes while preserving each layer identity.

        Same mechanism as :meth:`update_layers`, minus the re-filtering: an
        appearance change does not move any localization, so the render range
        it contributes to is unchanged.
        """
        for channel_index, dataset in enumerate(self.parent.localization_datasets):
            self._refresh_layer(dataset, channel_index)

    def _render_request(
        self, dataset, name=None, colormap=None, changed=Changed.EVERYTHING
    ):
        """Ask the planner what this dataset should look like right now."""
        self._planning_for = dataset.dataset_id
        request = self.planner.plan(
            dataset.table,
            self.gaussian_settings(),
            self.traits_of(dataset),
            name=name if name is not None else dataset.name,
            transform=self.transform_of(dataset),
            colormap=colormap,
            antialias=self.render_anti_alias,
            changed=changed,
            size_limit=self._splat_size_limit(),
        )
        # Kept for the resource-limit reporting and for tests that inspect what
        # the renderer was handed.
        state = self._state_for(dataset)
        state.sigmas, state.size, state.values = (
            request.sigmas,
            request.size,
            request.values,
        )
        self._note_clamped_splat(dataset, request.size)
        return request

    def set_appearance(self, dataset, **fields):
        """Change how *dataset* is drawn, without touching what is drawn.

        Routed through the store, which records it on the dataset's state and
        emits AppearanceChanged; this class hears that like any other listener
        and passes it to the backend. The appearance therefore outlives the
        control that set it, which Qt widget state did not.
        """
        store = getattr(self.parent, "dataset_store", None)
        if store is not None and store.state_of(dataset) is not None:
            store.set_appearance(dataset.dataset_id, **fields)
        elif self.renderer.is_open(dataset.dataset_id):
            self.renderer.set_appearance(dataset.dataset_id, LayerAppearance(**fields))

    def appearance_of(self, dataset):
        """The appearance recorded for *dataset*, or None."""
        store = getattr(self.parent, "dataset_store", None)
        state = None if store is None else store.state_of(dataset)
        if state is not None:
            return state.appearance
        return self.renderer.appearance(dataset.dataset_id)

    def value_range_of(self, dataset):
        """The value range a contrast control should scale against."""
        value_range = self.renderer.value_range(dataset.dataset_id)
        # A backend with nothing drawn yet still has to give the control
        # something to build its slider from.
        return (0.0, 1.0) if value_range is None else value_range

    def layer_for(self, dataset):
        """The host layer drawing *dataset*, or None.

        Backend-specific and deliberately *not* part of the protocol -- a host
        layer object is exactly what the protocol exists to keep out of the
        application. Nothing in the application uses this; it is here for tests
        and inspection of the napari backend.
        """
        layer = getattr(self.renderer, "layer", None)
        return None if layer is None else layer(getattr(dataset, "dataset_id", None))

    def update_data_range(self, dataset):
        # The render ranges and the localization coordinates are both in true
        # nanometres, so the slider percentages map straight onto the axis
        # extents with no offset to undo.
        #
        # Combining per-axis boolean masks with `&` replaces the previous
        # np.where + np.intersect1d chain.  intersect1d sorts both inputs and
        # allocates several temporaries per axis; the masks are one pass and one
        # allocation of 1 B/localization each.
        axes = ("x", "y", "z") if dataset.zdim_present else ("x", "y")
        ranges = {
            "x": (self.render_range_x, self.render_config.range_x_percent),
            "y": (self.render_range_y, self.render_config.range_y_percent),
            "z": (self.render_range_z, self.render_config.range_z_percent),
        }
        render_mask = None
        for axis in axes:
            axis_range, percent = ranges[axis]
            low, high = self.percent_to_absolute(axis_range, percent)
            axis_mask = dataset.get_mask_of_specified_prop_all(
                prop=f"{axis}_pos_nm", l_val=low, u_val=high
            )
            render_mask = (
                axis_mask if render_mask is None else (render_mask & axis_mask)
            )

        param_indices = self.parent.data_filter_itf.indices_for(dataset)
        dataset.apply_filters(render_mask, param_indices)
        self.apply_memory_budget(dataset)

    def _state_for(self, dataset):
        """The render record for *dataset*, created on first use."""
        state = self.render_state.get(dataset.dataset_id)
        if state is None:
            state = RenderArrays(sigmas=None, size=0.0, values=None)
            self.render_state[dataset.dataset_id] = state
        return state

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def gaussian_settings(self):
        """The render configuration as the host-free planner wants it."""
        config = self.render_config
        return GaussianSettings(
            mode=config.gaussian_mode,
            fixed_sigma_xy_nm=config.fixed_sigma_xy_nm,
            fixed_sigma_z_nm=config.fixed_sigma_z_nm,
            var_psf_sigma_xy_nm=config.var_psf_sigma_xy_nm,
            var_psf_sigma_z_nm=config.var_psf_sigma_z_nm,
            var_sigma_min_xy_nm=config.var_sigma_min_xy_nm,
            var_sigma_min_z_nm=config.var_sigma_min_z_nm,
            z_color_encoding=bool(config.z_color_encoding),
        )

    @staticmethod
    def traits_of(dataset):
        """What this dataset's format actually recorded."""
        return DatasetTraits(
            zdim_present=bool(dataset.zdim_present),
            sigma_present=bool(getattr(dataset, "sigma_present", False)),
            photon_count_present=bool(getattr(dataset, "photon_count_present", False)),
            pixel_size_nm=float(getattr(dataset, "pixelsize_nm", None) or 1.0),
        )

    def reference_plane_z_nm(self):
        """Where a *newly imported* reference image should sit, in nanometres.

        The centre of the localizations' depth range: for a 3-D dataset that
        puts the image in the middle of the stack, and for a flat one it lands
        on the localizations' own plane -- which is what makes napari's 2-D
        display, a single slice, able to show both at once.

        With nothing loaded it falls back to the flat-data plane, so an image
        imported first still meets a 2-D dataset imported second.

        Read once, at import.  It is deliberately **not** re-applied when a
        dataset is later loaded or closed: §3.5 removed exactly that behaviour,
        and §7.4 makes "loading a dataset does not move any other layer" an
        acceptance gate.  Re-centring after the fact is a button the user
        presses, not a rule that fires behind them.
        """
        low, high = self.render_range_z
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            return FLAT_DATA_Z_NM
        return 0.5 * (float(low) + float(high))

    def transform_of(self, dataset):
        """Where this dataset sits in world space."""
        store = getattr(self.parent, "dataset_store", None)
        state = None if store is None else store.state_of(dataset)
        return IDENTITY if state is None else state.transform

    def get_coords_from_locs(self, dataset):
        """The drawn coordinates, in the renderer's (z, y, x) order."""
        return self.planner.coordinates(
            dataset.table.selection(ACTIVE),
            self.traits_of(dataset),
            self.transform_of(dataset),
        )

    def get_coords_from_all_locs(self, dataset):
        """Every localization's coordinates, in the renderer's (z, y, x) order."""
        if dataset.zdim_present:
            num_of_locs = len(dataset.x_pos_nm_all)
            coords = np.zeros([num_of_locs, 3], dtype=np.float32)
            coords[:, 0] = dataset.z_pos_nm_all
            coords[:, 1] = dataset.y_pos_nm_all
            coords[:, 2] = dataset.x_pos_nm_all

        else:
            num_of_locs = len(dataset.x_pos_nm_all)
            coords = np.zeros([num_of_locs, 3], dtype=np.float32)
            coords[:, 1] = dataset.y_pos_nm_all
            coords[:, 2] = dataset.x_pos_nm_all
            coords[:, 0] = np.ones(num_of_locs, dtype=np.float32)
        return coords

    def scalebar(self):
        """Delegate to ScalebarRenderer."""
        datasets = self.parent.localization_datasets
        if not datasets:
            # The scalebar is sized from a dataset's extent, so there is
            # nothing to update before one is loaded.  Reached from the
            # scalebar-size field's typing timer, which fires whether or not
            # anything is open.
            return
        self._sbr.update(datasets[-1])

    # active_locs_to_choords / all_locs_to_choords were removed here.  Both built
    # a three-field record array whose fields were named "_pixels" but held
    # nanometres, with x and y transposed on the way in.  The only remaining
    # caller was the Z-colour path in set_render_values, which allocated the
    # whole thing per update to read one column; it now reads the cached z
    # column directly.
