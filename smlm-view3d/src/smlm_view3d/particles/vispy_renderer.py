"""The instanced backend on a plain VisPy scene: no napari.

`VispyInstancedRenderer` meets the renderer contract
(`core.renderer.LocalizationRenderer`) on a VisPy `SceneCanvas` of its own,
for a host that has no napari viewer.  It does what the napari backend does,
with a VisPy scene where that one has napari:

- `InstancedRenderer` (`instanced_renderer.py`): one entry per dataset id,
  open / update / close and the appearance;
- `InstancedParticles` (`instanced_layer.py`): one mesh whose mesh is one quad,
  the billboard filter on it, the instance rows, the summed-contrast pass and
  the blend state.

Each method's docstring names the napari backend's method it follows.  Traces,
pairs and focus are not drawn: `draws_traces`, `draws_pairs` and
`supports_focus` keep the contract's False.

Instancing needs VisPy's ``gl+`` GL backend (PyOpenGL), chosen before any GL
context exists.  The constructor chooses it; if instancing is still not
available, it raises RuntimeError rather than falling back to another way of
drawing.  Qt is reached only through VisPy's app: ``app`` names VisPy's app
backend, and None lets VisPy choose (the Qt binding already imported, if any).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vispy import scene
from vispy.color import get_colormap

from ..core.footprints import (
    BLEND_OPAQUE,
    DEFAULT_MIN_SIZE_PX,
    FOOTPRINT_GAUSSIAN,
    footprint_named,
    validate_footprint,
    validate_min_size_px,
)
from ..core.renderer import Changed, LayerAppearance, LocalizationRenderer
from .billboard_filter import QUAD_FACES, QUAD_XY, InstancedBillboardsFilter
from .footprint_shaders import is_opaque
from .summed_contrast import SummedContrastPass
from .vispy_compat import (
    ADDITIVE_BLEND_STATE,
    enable_instanced_backend,
    instancing_available,
)

__all__ = ["VispyInstancedRenderer"]

#: napari's ``opaque`` blending preset, which an opaque footprint gets: depth
#: test on, no blending.
OPAQUE_BLEND_STATE = {"depth_test": True, "cull_face": False, "blend": False}

#: The contrast window a dataset opens with: what napari gives the instanced
#: layer, a Surface whose four quad vertices all carry 1.0.
DEFAULT_CONTRAST_LIMITS = (0.0, 1.0)


def _vispy_colormap(colormap):
    """The VisPy colormap for *colormap*: a VisPy colormap, or its name."""
    if getattr(colormap, "glsl_map", None):
        return colormap
    return get_colormap(colormap)


class _Billboards:
    """One dataset: `InstancedParticles`, with a VisPy mesh for the napari layer.

    The mesh is one quad, drawn once per localization by
    `InstancedBillboardsFilter` from the instance rows; `SummedContrastPass`
    windows the summed image.  Positions arrive in the planner's (z, y, x)
    order; the filter's `set_instances` reverses them into VisPy's x, y, z.
    """

    def __init__(self, request, parent):
        """`InstancedParticles.__init__` and `add_to_viewer`, without napari."""
        coords = np.asarray(request.coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords should be of shape (N, 3)")
        if len(coords) == 0:
            raise ValueError("a dataset to draw needs at least one localization")
        self._set_rows(coords, request.sigmas, request.values)
        self.billboard_size = float(np.max(request.size))

        self.colormap = request.colormap
        self.opacity = 1.0
        self.contrast_limits = DEFAULT_CONTRAST_LIMITS
        self.footprint = FOOTPRINT_GAUSSIAN
        self.min_size_px = DEFAULT_MIN_SIZE_PX
        self.summed_contrast = True
        self.color_coded = False

        vertices, faces = self._quad_mesh()
        self.mesh = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            vertex_values=np.ones(len(vertices), dtype=np.float32),
            shading=None,
            parent=parent,
            name=request.name,
        )
        self.filter = InstancedBillboardsFilter()
        self.mesh.attach(self.filter)
        self.summed_pass = SummedContrastPass(self.mesh)
        self._upload_instances()
        self._apply_colormap()
        self._apply_contrast_limits()
        self._apply_opacity()
        self._apply_footprint()

    # ------------------------------------------------------------------
    # Instance rows and the quad
    # ------------------------------------------------------------------

    def _set_rows(self, coords, sigmas, values):
        """The per-instance rows, broadcast as `InstancedParticles` does."""
        self.coords = coords
        self.sigmas = np.ascontiguousarray(
            np.broadcast_to(np.asarray(sigmas, dtype=np.float32), coords.shape)
        )
        self.values = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values, dtype=np.float32), (len(coords),))
        )

    def _quad_mesh(self):
        """`InstancedParticles._quad_mesh`, in the order napari hands it to VisPy.

        The quad is built in napari's (z, y, x) order, and napari passes VisPy
        the vertex columns reversed and each face's winding reversed with them.
        The quad sits on the first localization's z, as there.
        """
        vertices = np.zeros((4, 3), dtype=np.float32)
        vertices[:, 1:] = QUAD_XY * self.billboard_size
        vertices[:, 0] = float(self.coords[0, 0])
        return vertices[:, ::-1].copy(), QUAD_FACES[:, ::-1].copy()

    def _upload_instances(self):
        """`InstancedParticles._upload_instances`."""
        self.filter.set_billboard_size(self.billboard_size)
        self.filter.set_instances(self.coords, self.sigmas, self.values)
        self.mesh.update()

    def update_particle_data(self, coords, size, sigmas, values):
        """`InstancedParticles.update_particle_data`: new rows; a new quad only
        if the size changed."""
        coords = np.asarray(coords, dtype=np.float32)
        if len(coords) == 0:
            raise ValueError("a dataset to draw needs at least one localization")
        self._set_rows(coords, sigmas, values)
        new_size = float(np.max(size))
        if new_size != self.billboard_size:
            self.billboard_size = new_size
            vertices, faces = self._quad_mesh()
            self.mesh.set_data(
                vertices=vertices,
                faces=faces,
                vertex_values=np.ones(len(vertices), dtype=np.float32),
            )
        self._upload_instances()

    @property
    def extent_xyz(self):
        """`InstancedParticles._extent_data`, in VisPy's x, y, z: the
        localizations, not the quad.

        Padded by half the billboard so whole splats are framed, except along a
        z that the data does not have.
        """
        half = 0.5 * self.billboard_size
        mins = np.min(self.coords, axis=0) - half
        maxs = np.max(self.coords, axis=0) + half
        if float(np.ptp(self.coords[:, 0])) == 0.0:
            mins[0] = maxs[0] = float(self.coords[0, 0])
        return mins[::-1], maxs[::-1]

    def host_bytes(self):
        """`InstancedParticles.host_bytes`: one row per localization."""
        return int(self.coords.nbytes + self.sigmas.nbytes + self.values.nbytes)

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    @property
    def contrast_is_summed(self):
        """`InstancedParticles.contrast_is_summed`: only an additive footprint
        has a sum."""
        return self.summed_contrast and not is_opaque(self.footprint)

    def _apply_colormap(self):
        """`InstancedParticles._apply_colormap`: one colormap for the mesh, the
        filter and the pass.

        The mesh carries it as napari's node does; that also binds the
        colormap's lookup texture (``texture2D_LUT``) on the mesh's program,
        which the filter's colormap lookup uses.
        """
        colormap = _vispy_colormap(self.colormap)
        self.mesh.cmap = colormap
        self.filter.set_colormap(colormap)
        self.summed_pass.set_colormap(colormap)
        self.mesh.update()

    def _apply_contrast_limits(self):
        """`InstancedParticles._apply_contrast_limits`, and the mesh's clim as
        napari sets it."""
        low, high = self.contrast_limits
        self.mesh.clim = (float(low), float(high))
        self.filter.set_contrast_limits(low, high)
        self.summed_pass.set_contrast_limits(low, high)
        self.mesh.update()

    def _apply_opacity(self):
        """`InstancedParticles._apply_opacity`, and the node opacity napari sets
        on its visual."""
        self.mesh.opacity = float(self.opacity)
        self.summed_pass.opacity = float(self.opacity)
        self.mesh.update()

    def _apply_footprint(self):
        """`InstancedParticles._apply_footprint`."""
        self.filter.set_footprint(self.footprint, self.min_size_px)
        self.apply_blend_state()
        self._apply_contrast_model()

    def _apply_contrast_model(self):
        """`InstancedParticles._apply_contrast_model`: the shader and the pass
        switch together."""
        summed = self.contrast_is_summed
        self.filter.set_accumulate(summed, mean_colour=self.color_coded)
        self.summed_pass.enabled = summed
        self.summed_pass.mean_colour = self.color_coded
        self.mesh.update()

    def apply_blend_state(self):
        """`_napari_compat.apply_footprint_blending`, on the mesh itself.

        An additive footprint is summed with `ADDITIVE_BLEND_STATE`; an opaque
        one gets napari's ``opaque`` preset, depth test on.  No napari
        recomputes the state here, so it is set, not forced.
        """
        if footprint_named(self.footprint).blend == BLEND_OPAQUE:
            self.mesh.set_gl_state(**OPAQUE_BLEND_STATE)
        else:
            self.mesh.set_gl_state(cull_face=False, **ADDITIVE_BLEND_STATE)
        self.mesh.update()

    def set_appearance(self, appearance):
        """`InstancedRenderer.set_appearance`: None fields are left alone."""
        if appearance.colormap is not None:
            self.colormap = appearance.colormap
            self._apply_colormap()
        if appearance.opacity is not None:
            self.opacity = float(appearance.opacity)
            self._apply_opacity()
        if appearance.contrast_limits is not None:
            low, high = appearance.contrast_limits
            self.contrast_limits = (float(low), float(high))
            self._apply_contrast_limits()
        if appearance.summed_contrast is not None:
            self.summed_contrast = bool(appearance.summed_contrast)
            self._apply_contrast_model()
        if appearance.color_coded is not None:
            self.color_coded = bool(appearance.color_coded)
            self._apply_contrast_model()
        if appearance.visible is not None:
            self.mesh.visible = bool(appearance.visible)
        if appearance.min_size_px is not None:
            self.min_size_px = validate_min_size_px(appearance.min_size_px)
            self._apply_footprint()
        if appearance.footprint is not None:
            self.footprint = validate_footprint(appearance.footprint)
            self._apply_footprint()

    def appearance(self):
        """`InstancedRenderer.appearance`, with traces and pairs off (not drawn
        here)."""
        return LayerAppearance(
            colormap=self.colormap,
            opacity=self.opacity,
            contrast_limits=tuple(self.contrast_limits),
            visible=bool(self.mesh.visible),
            footprint=self.footprint,
            min_size_px=self.min_size_px,
            summed_contrast=self.summed_contrast,
            color_coded=self.color_coded,
            traces=False,
            pairs=False,
        )

    def value_range(self):
        """The span of the instance values, for a contrast control to scale to.

        The napari backend reports the range napari keeps for its layer.  Here
        it is the values' own span, widened as napari widens a span of zero
        width (to include 0 and 1), so values that are all 1 give (0, 1), as
        the napari layer does.
        """
        low, high = float(np.min(self.values)), float(np.max(self.values))
        if low == high:
            low, high = min(low, 0.0), max(high, 1.0)
        return low, high

    def close(self):
        """`InstancedParticles.close`: the pass gives the draw back, the filter
        comes off, the mesh leaves the scene."""
        self.summed_pass.release()
        try:
            self.mesh.detach(self.filter)
        except (ValueError, AttributeError, RuntimeError):
            pass
        self.mesh.parent = None


class VispyInstancedRenderer(LocalizationRenderer):
    """The instanced backend on a VisPy scene of its own (no napari).

    ``canvas`` is a `vispy.scene.SceneCanvas`; ``widget`` is its native widget,
    for a Qt layout.  ``view`` is its one view, with a `TurntableCamera`.
    `open` returns the dataset's handle, its VisPy mesh visual.
    """

    def __init__(self, parent=None, size=(800, 600), bgcolor="black", app=None):
        """`InstancedRenderer.__init__`: instancing or nothing.

        ``gl+`` is chosen before the canvas, and so before any GL context
        exists.  Without instancing this raises RuntimeError and makes no
        canvas.  ``app`` is VisPy's app backend for the canvas (e.g.
        "pyqt6"); None lets VisPy choose.
        """
        enable_instanced_backend()
        if not instancing_available():
            raise RuntimeError(
                "instanced rendering needs VisPy's 'gl+' GL backend (PyOpenGL "
                "installed, selected before any GL context exists); it is not "
                "available in this session"
            )
        self.canvas = scene.SceneCanvas(
            app=app,
            keys=None,
            size=size,
            bgcolor=bgcolor,
            parent=parent,
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=0.0)
        self._layers: dict[Any, _Billboards] = {}

    @property
    def widget(self):
        """The canvas's native widget, to put in a layout."""
        return self.canvas.native

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------

    def layer(self, dataset_id):
        """`InstancedRenderer.layer`: the dataset's mesh visual, or None."""
        entry = self._layers.get(dataset_id)
        return None if entry is None else entry.mesh

    @property
    def layers(self):
        """`InstancedRenderer.layers`: the open datasets' mesh visuals."""
        return [entry.mesh for entry in self._layers.values()]

    def is_open(self, dataset_id):
        """`InstancedRenderer.is_open`."""
        return dataset_id in self._layers

    def reset_view(self):
        """Frame the camera on every open dataset, as napari frames its view on
        the layers' extent."""
        if not self._layers:
            return
        entries = self._layers.values()
        mins = np.min([entry.extent_xyz[0] for entry in entries], axis=0)
        maxs = np.max([entry.extent_xyz[1] for entry in entries], axis=0)
        self.view.camera.set_range(
            x=(float(mins[0]), float(maxs[0])),
            y=(float(mins[1]), float(maxs[1])),
            z=(float(mins[2]), float(maxs[2])),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, dataset_id, request):
        """`InstancedRenderer.open`: replace anything open for the id, draw it,
        frame the view."""
        self.close(dataset_id)
        entry = _Billboards(request, parent=self.view.scene)
        self._layers[dataset_id] = entry
        self.reset_view()
        return entry.mesh

    def update(self, dataset_id, request):
        """`InstancedRenderer.update`: the same visual, new instance rows."""
        entry = self._layers.get(dataset_id)
        if entry is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        if not (request.changed & ~(Changed.TRACES | Changed.PAIRS)):
            # Only the overlay differs, and no overlay is drawn here.
            return entry.mesh
        entry.update_particle_data(
            coords=request.coords,
            size=request.size,
            sigmas=request.sigmas,
            values=request.values,
        )
        entry.mesh.visible = True
        entry.apply_blend_state()
        return entry.mesh

    def set_visible(self, dataset_id, visible):
        """`InstancedRenderer.set_visible`: hide or show, keeping everything."""
        entry = self._layers.get(dataset_id)
        if entry is not None:
            entry.mesh.visible = bool(visible)

    def set_appearance(self, dataset_id, appearance):
        """`InstancedRenderer.set_appearance`."""
        entry = self._layers.get(dataset_id)
        if entry is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        entry.set_appearance(appearance)
        return entry.mesh

    def appearance(self, dataset_id):
        """`InstancedRenderer.appearance`."""
        entry = self._layers.get(dataset_id)
        return None if entry is None else entry.appearance()

    def value_range(self, dataset_id):
        """`InstancedRenderer.value_range`; see `_Billboards.value_range`."""
        entry = self._layers.get(dataset_id)
        return None if entry is None else entry.value_range()

    def contrast_is_summed(self, dataset_id):
        """`InstancedRenderer.contrast_is_summed`."""
        entry = self._layers.get(dataset_id)
        return entry is not None and entry.contrast_is_summed

    def close(self, dataset_id):
        """`InstancedRenderer.close`: release what the dataset installed and
        take it off the scene."""
        entry = self._layers.pop(dataset_id, None)
        if entry is not None:
            entry.close()

    def close_all(self):
        """`InstancedRenderer.close_all`."""
        for dataset_id in list(self._layers):
            self.close(dataset_id)

    def host_bytes(self, dataset_id):
        """`InstancedRenderer.host_bytes`: the instance rows."""
        entry = self._layers.get(dataset_id)
        return 0 if entry is None else entry.host_bytes()
