"""An instanced Gaussian billboard layer that napari can actually draw.

`instanced.py` proved the shader and the memory shape in isolation. This puts
the same idea inside napari, so the backend can be run and looked at.

The trick that makes it fit napari with no custom visual class: gloo decides to
draw instanced entirely from the buffers it is given — any attribute with a
`divisor` set turns the draw into `glDrawElementsInstanced`, with the instance
count taken from that buffer's length. So a napari `Surface` layer whose mesh is
*one quad* (four vertices, two triangles), plus filters that attach divisor-1
buffers for the per-localization centre, sigma and value, renders every
localization from four vertices.

Two things make that work rather than merely compile:

* `render_size` is already one scalar per dataset — a fixed multiple of the
  largest sigma — so the quad can carry the size and only the *shape* varies per instance.
  Had the size been per-localization the quad could not have been shared.
* `_extent_data` is overridden. napari would otherwise frame the camera on a
  four-vertex quad at the origin instead of on the data.

**Requires VisPy's `gl+` backend**, selected process-wide before any GL context
exists. See `enable_instanced_backend()` in `_napari_compat`.

The billboard filter and its quad (`InstancedBillboardsFilter`, `QUAD_XY`,
`QUAD_FACES`) live in `billboard_filter`, which needs no napari.  They are
imported here, so every old import keeps working.
"""

from __future__ import annotations

import numpy as np
from napari.layers import Surface

from ..core.footprints import (
    DEFAULT_MIN_SIZE_PX,
    FOOTPRINT_GAUSSIAN,
    validate_footprint,
    validate_min_size_px,
)
from ._napari_compat import (
    apply_footprint_blending,
    get_layer_visual,
    release_additive_blending,
)
from .billboard_filter import QUAD_FACES, QUAD_XY, InstancedBillboardsFilter
from .footprint_shaders import is_opaque
from .summed_contrast import SummedContrastPass

__all__ = ["InstancedParticles", "InstancedBillboardsFilter"]


class InstancedParticles(Surface):
    """A napari Surface whose mesh is one quad, drawn once per localization."""

    def __init__(self, coords, size=10, sigmas=(1, 1, 1), values=1, **kwargs):
        kwargs.setdefault("shading", "none")
        kwargs.setdefault("blending", "additive")

        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords should be of shape (N, 3)")
        if len(coords) == 0:
            raise ValueError("InstancedParticles requires at least one localization")

        self._instance_coords = coords
        self._instance_sigmas = np.ascontiguousarray(
            np.broadcast_to(np.asarray(sigmas, dtype=np.float32), coords.shape)
        )
        self._instance_values = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values, dtype=np.float32), (len(coords),))
        )
        self._billboard_size = float(np.max(size))

        self._billboard_filter = InstancedBillboardsFilter()
        self._footprint = FOOTPRINT_GAUSSIAN
        self._min_size_px = DEFAULT_MIN_SIZE_PX
        self._summed_contrast = True
        self._color_coded = False
        self._summed_pass = None
        self._viewer = None
        self._visual = None
        self._layer_event_connections = []

        vertices, faces, values4 = self._quad_mesh()
        super().__init__((vertices, faces, values4), **kwargs)

    # ------------------------------------------------------------------
    # Footprint
    # ------------------------------------------------------------------

    @property
    def footprint(self):
        """A name from `core.footprints.PALETTE`; see `LayerAppearance`."""
        return self._footprint

    @footprint.setter
    def footprint(self, value):
        self._footprint = validate_footprint(value)
        self._apply_footprint()

    @property
    def min_size_px(self):
        """Smallest on-screen diameter of a visualisation, in canvas pixels."""
        return self._min_size_px

    @min_size_px.setter
    def min_size_px(self, value):
        self._min_size_px = validate_min_size_px(value)
        self._apply_footprint()

    def _apply_footprint(self):
        self._billboard_filter.set_footprint(self._footprint, self._min_size_px)
        self._apply_blend_state()
        self._apply_contrast_model()

    # ------------------------------------------------------------------
    # Contrast model
    # ------------------------------------------------------------------

    @property
    def summed_contrast(self):
        """Whether an additive footprint's window applies to the sum.

        See `LayerAppearance.summed_contrast`.  Off, every localization is
        windowed and colormapped on its own.
        """
        return self._summed_contrast

    @summed_contrast.setter
    def summed_contrast(self, value):
        self._summed_contrast = bool(value)
        self._apply_contrast_model()

    @property
    def color_coded(self):
        """Whether the values are colormap positions; see `LayerAppearance`."""
        return self._color_coded

    @color_coded.setter
    def color_coded(self, value):
        self._color_coded = bool(value)
        self._apply_contrast_model()

    @property
    def contrast_is_summed(self):
        """Whether the window is being applied to the summed image right now.

        Only an additive footprint has a sum; an opaque one keeps the nearest
        marker and nothing else, so it is always windowed per localization.
        """
        return self._summed_contrast and not is_opaque(self._footprint)

    def _apply_contrast_model(self):
        """Switch the shader and the resolve pass together, never apart."""
        summed = self.contrast_is_summed
        self._billboard_filter.set_accumulate(summed, mean_colour=self._color_coded)
        if self._summed_pass is not None:
            self._summed_pass.enabled = summed
            self._summed_pass.mean_colour = self._color_coded
        if self._visual is not None:
            self._visual.update()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def _is_flat(self):
        """True when every localization sits on one z plane, i.e. 2-D data."""
        z = self._instance_coords[:, 0]
        return len(z) == 0 or float(np.ptp(z)) == 0.0

    def _quad_mesh(self):
        """The whole mesh: four vertices, whatever the localization count.

        For flat data the quad is placed *on the data's plane* rather than at
        the origin. In 3-D display that is invisible -- the vertex shader
        replaces the position outright, using only the quad's xy to recover the
        corner offset -- but in 2-D display it is the difference between
        drawing and not drawing at all: napari slices a Surface against the
        current plane, and a quad sitting at z = 0 while the data claims z = 1
        is never intersected.
        """
        vertices = np.zeros((4, 3), dtype=np.float32)
        vertices[:, 1:] = QUAD_XY * self._billboard_size
        if len(self._instance_coords):
            vertices[:, 0] = float(self._instance_coords[0, 0])
        return vertices, QUAD_FACES.copy(), np.ones(4, dtype=np.float32)

    @property
    def n_instances(self):
        return len(self._instance_coords)

    @property
    def localization_coords(self):
        """The localization centres being drawn, as an (N, 3) array in (z, x, y).

        One row per localization on both backends, which is what makes it the
        right thing for a caller to ask for. The billboard backend also keeps a
        six-vertex expansion of these; this is not that.
        """
        return self._instance_coords

    @property
    def billboard_size_nm(self):
        """Edge length of the largest splat drawn, in nanometres.

        One scalar per dataset here and one value per localization on the
        billboard backend, so the screen-space cap -- which is a statement about
        the widest splat either way -- can be checked without knowing which.
        """
        return self._billboard_size

    @property
    def n_localizations(self):
        """How many localizations this layer is drawing.

        The same question `Particles.n_localizations` answers, so that callers
        who only want the count -- tests asserting a filter took effect, mostly
        -- do not have to know that one backend stores six vertices per
        localization and the other stores one instance.
        """
        return len(self._instance_coords)

    @property
    def _extent_data(self) -> np.ndarray:
        """The extent of the localizations, not of the quad.

        Without this napari frames the camera on a four-vertex quad sitting at
        the origin, and the data is nowhere on screen.
        """
        half = 0.5 * self._billboard_size
        mins = np.min(self._instance_coords, axis=0) - half
        maxs = np.max(self._instance_coords, axis=0) + half
        if self._is_flat:
            # Do not pad the depth of data that has none. The padding exists so
            # the camera frames whole splats, which is meaningless across a
            # single plane -- and it costs 2-D display mode: napari derives the
            # slider range from this extent, so a padded z puts the slice plane
            # somewhere the data is not.
            mins[0] = maxs[0] = float(self._instance_coords[0, 0])
        return np.vstack([mins, maxs])

    def host_bytes(self):
        """One row per localization, plus a quad that amortizes to nothing."""
        return int(
            self._instance_coords.nbytes
            + self._instance_sigmas.nbytes
            + self._instance_values.nbytes
        )

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_particle_data(self, coords, size, sigmas, values):
        """Replace the per-instance rows.  The quad only changes if size does."""
        coords = np.asarray(coords, dtype=np.float32)
        if len(coords) == 0:
            raise ValueError("InstancedParticles requires at least one localization")
        self._instance_coords = coords
        self._instance_sigmas = np.ascontiguousarray(
            np.broadcast_to(np.asarray(sigmas, dtype=np.float32), coords.shape)
        )
        self._instance_values = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values, dtype=np.float32), (len(coords),))
        )
        new_size = float(np.max(size))
        if new_size != self._billboard_size:
            self._billboard_size = new_size
            self.data = self._quad_mesh()
        self._upload_instances()
        self.events.data(value=self.data)

    def _set_view_slice(self):
        super()._set_view_slice()
        self._upload_instances()

    def _upload_instances(self):
        if self._billboard_filter._attached:
            self._billboard_filter.set_billboard_size(self._billboard_size)
            self._billboard_filter.set_instances(
                self._instance_coords,
                self._instance_sigmas,
                self._instance_values,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_to_viewer(self, viewer):
        self._viewer = viewer
        viewer.add_layer(self)
        self._visual = get_layer_visual(viewer, self)
        self._visual.attach(self._billboard_filter)
        self._summed_pass = SummedContrastPass(self._visual)
        self._upload_instances()
        self._apply_colormap()
        self._apply_contrast_limits()
        self._apply_opacity()
        self.events.colormap.connect(self._apply_colormap)
        self.events.contrast_limits.connect(self._apply_contrast_limits)
        self.events.opacity.connect(self._apply_opacity)
        self._apply_blend_state()
        self._apply_contrast_model()

    def _apply_blend_state(self, event=None):
        """Keep the blend state the footprint needs, and keep it kept.

        Gaussians need true additive blending, forced.  Until P0-01 additive
        blending was repaired by accident -- every update destroyed and rebuilt
        the layer, and `add_to_viewer` set the state again on the way back in.
        Updating in place removed the rebuild and with it the repair, so it has
        to be asserted deliberately.  See `force_additive_blending` for why it
        is asserted by wrapping the setter rather than from event handlers.
        Discs need napari's opaque preset instead; see
        `apply_footprint_blending`.
        """
        apply_footprint_blending(self, self._visual, self._footprint)

    def _apply_colormap(self, event=None):
        """Hand the visual's colormap to the shader that samples it."""
        if self._visual is None:
            return
        colormap = getattr(self._visual, "cmap", None)
        if colormap is not None:
            self._billboard_filter.set_colormap(colormap)
            if self._summed_pass is not None:
                self._summed_pass.set_colormap(colormap)

    def _apply_contrast_limits(self, event=None):
        """Push the layer's contrast window into the shader that uses it."""
        limits = getattr(self, "contrast_limits", None)
        if limits is not None and len(limits) == 2:
            self._billboard_filter.set_contrast_limits(limits[0], limits[1])
            if self._summed_pass is not None:
                self._summed_pass.set_contrast_limits(limits[0], limits[1])

    def _apply_opacity(self, event=None):
        """The resolve pass applies opacity itself; the splats no longer see it."""
        if self._summed_pass is not None:
            self._summed_pass.opacity = float(self.opacity)

    def set_focus(self, focus):
        """Fade what lies outside a `core.FocusState`; None stops fading.

        The centre arrives in world coordinates and goes to the shader in this
        layer's own, so a dataset shifted in world space fades about the same
        point as the others.
        """
        if focus is None:
            self._billboard_filter.set_focus(None)
        else:
            center = np.asarray(
                self.world_to_data(np.asarray(focus.center_world, dtype=float)),
                dtype=float,
            )
            self._billboard_filter.set_focus(
                center_xyz=center[::-1],
                radius=focus.radius_nm,
                edge=focus.edge_nm,
                floor=focus.outside_opacity,
                planar=focus.planar,
            )
        if self._visual is not None:
            self._visual.update()

    def close(self):
        self._layer_event_connections = []
        for emitter, handler in (
            ("colormap", self._apply_colormap),
            ("contrast_limits", self._apply_contrast_limits),
            ("opacity", self._apply_opacity),
        ):
            try:
                getattr(self.events, emitter).disconnect(handler)
            except (ValueError, TypeError, RuntimeError):
                pass
        if self._summed_pass is not None:
            self._summed_pass.release()
            self._summed_pass = None
        if self._visual is not None:
            release_additive_blending(self._visual)
            try:
                self._visual.detach(self._billboard_filter)
            except (ValueError, AttributeError, RuntimeError):
                pass
        self._visual = None
        self._viewer = None

    detach = close
