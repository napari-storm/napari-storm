"""
A billboarded particle layer with texture/shader support

"""

from collections.abc import Iterable

import numpy as np
from napari.layers import Surface
from vispy.gloo import VertexBuffer
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function, Varying

from ..core.footprints import (
    DEFAULT_MIN_SIZE_PX,
    FOOTPRINT_GAUSSIAN,
    FOOTPRINTS,
    footprint_named,
    validate_footprint,
    validate_min_size_px,
)
from ._napari_compat import (
    apply_footprint_blending,
    get_layer_visual,
    mesh_vertex_buffer,
    release_additive_blending,
)
from .filters import ShaderFilter
from .footprint_shaders import quad_scale
from .utils import generate_billboards_2d

_DEFAULT_FILTER = object()


class BillboardsFilter(Filter):
    """Billboard geometry filter (transforms vertices to always face camera)"""

    def __init__(self, antialias=0):
        vmat_inv = Function("""
            mat2 inverse(mat2 m) {
                return mat2(m[1][1],-m[0][1],-m[1][0], m[0][0]) / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
            }
        """)

        vfunc = Function("""
        varying float v_z_center;
        varying float v_scale_intensity;
        varying mat2 v_disc_inv;

        void apply(){
            // This vertex's corner of its particle's quad, e.g. [5,5] for
            // size 10: every quad sits at the origin and $vertex_center moves
            // it.  Read from the mesh's own vertex buffer, not recovered by
            // inverting the transforms, which through the perspective camera
            // falls apart far from the origin; see mesh_vertex_buffer.
            vec3 pos = $quad_corner;

            // Where this corner sits on the quad, -1 to 1, read off the
            // geometry.  It used to come from a texture-coordinate attribute,
            // which has to differ per corner and so has to agree with the
            // order the visual draws the vertices in -- and it did not: napari
            // reverses every face and VisPy expands the mesh per face corner,
            // so one triangle of every quad had its coordinates mirrored.
            // Invisible for an isotropic Gaussian, a chevron for an
            // anisotropic one.
            float half_edge = abs(pos.x);
            vec2 quad = pos.xy / half_edge;

            // The screen axes as unit vectors in world space.
            vec3 camera_right = $camera_inv(vec4(1,0,0,0)).xyz;
            vec3 camera_up    = $camera_inv(vec4(0,1,0,0)).xyz;
            float len = length(camera_right);
            camera_right = camera_right/len;
            camera_up    = camera_up/len;

            // The one-sigma ellipse and the on-screen floor, line for line as
            // InstancedBillboardsFilter draws them: the marginal of
            // diag(sigma^2) in the screen basis, in nanometres, divided by
            // the square of the drawn half-edge to put it in the quad's own
            // -1 to 1 coordinate.  It is what the Gaussian is drawn with too;
            // see footprint_shaders for the square-root covariance it
            // replaced.
            float drawn_half_edge = half_edge * $quad_scale;
            vec3 rs = camera_right * $sigmas;
            vec3 us = camera_up * $sigmas;
            mat2 disc_cov = mat2(dot(rs, rs), dot(rs, us),
                                 dot(rs, us), dot(us, us))
                          / (drawn_half_edge * drawn_half_edge);
            v_disc_inv = $inverse(disc_cov);
            float grow = 1.0;
            if ($min_half_px > 0.0) {
                vec4 c = $visual_to_canvas(vec4($vertex_center, 1.0));
                vec4 e = $visual_to_canvas(
                    vec4($vertex_center + camera_right * drawn_half_edge, 1.0));
                float det = disc_cov[0][0] * disc_cov[1][1]
                          - disc_cov[0][1] * disc_cov[1][0];
                float radius_px = length(e.xy / e.w - c.xy / c.w)
                                * sqrt(sqrt(max(det, 0.0)));
                if (radius_px > 0.0 && radius_px < $min_half_px) {
                    grow = $min_half_px / radius_px;
                }
            }

            // when particles become too small, lock texture size and apply
            // antialiasing (only used when antialias>0)
            float dist_cutoff = $antialias;
            vec4 p1 = $transform(vec4($vertex_center.xyz + camera_right*pos.x + camera_up*pos.y, 1.));
            vec4 p2 = $transform(vec4($vertex_center,1));
            float dist = length(p1.xy/p1.w-p2.xy/p2.w);

            // if antialias and far away zoomed out, keep sprite size constant
            // and shrink texture... else adjust sprite size
            if (($antialias>0) && (dist<dist_cutoff)) {
                float scale = dist_cutoff/dist;
                quad = quad*clamp(scale,1,10);
                camera_right = camera_right*scale;
                camera_up    = camera_up*scale;
                v_scale_intensity = scale;
            }
            vec3 pos_real  = $vertex_center.xyz
                           + (camera_right*pos.x + camera_up*pos.y)
                             * ($quad_scale * grow);
            gl_Position = $transform(vec4(pos_real, 1.));
            vec4 center = $transform(vec4($vertex_center,1));
            v_z_center = center.z/center.w;

            $v_texcoords = 0.5 * quad + 0.5;
        }
        """)

        ffunc = Function("""
        varying float v_scale_intensity;
        varying float v_z_center;

        void apply() {
            // Window-space depth; see InstancedBillboardsFilter for why it is
            // not v_z_center itself.
            gl_FragDepth = 0.5 * v_z_center + 0.5;
            $texcoords;
        }
        """)

        self._texcoord_varying = Varying("v_texcoord", "vec2")
        vfunc["inverse"] = vmat_inv
        vfunc["v_texcoords"] = self._texcoord_varying
        ffunc["texcoords"] = self._texcoord_varying

        vfunc["antialias"] = float(antialias)
        self._antialias = float(antialias)
        # The scientific Gaussian until `set_footprint` says otherwise.
        vfunc["quad_scale"] = 1.0
        vfunc["min_half_px"] = 0.0

        self._centercoords_buffer = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._sigmas_buffer = VertexBuffer(np.zeros((0, 3), dtype=np.float32))

        vfunc["vertex_center"] = self._centercoords_buffer
        vfunc["sigmas"] = self._sigmas_buffer

        super().__init__(vcode=vfunc, vhook="post", fcode=ffunc, fhook="post")

    def set_footprint(self, name, min_size_px=0.0):
        """Size the billboard for *name*, no smaller than *min_size_px* on screen.

        The shading itself is the layer's `ShaderFilter`.  Antialiasing is off
        for the visualisations: it keeps a far-away sprite's size by shrinking
        its texture, which is a Gaussian's answer to the problem the size floor
        answers.
        """
        footprint = footprint_named(name)
        self.vshader["quad_scale"] = quad_scale(name)
        if footprint.reconstruction:
            self.vshader["min_half_px"] = 0.0
            self.vshader["antialias"] = self._antialias
        else:
            self.vshader["min_half_px"] = 0.5 * float(min_size_px)
            self.vshader["antialias"] = 0.0

    @property
    def centercoords(self):
        """The vertex center coordinates as an (N, 3) array of floats."""
        return self._centercoords

    @centercoords.setter
    def centercoords(self, centercoords):
        self._centercoords = centercoords
        self._update_coords_buffer(centercoords)

    def _update_coords_buffer(self, centercoords):
        if self._attached and self._visual is not None:
            self._centercoords_buffer.set_data(centercoords[:, ::-1], convert=True)

    @property
    def sigmas(self):
        """The Gaussian widths per vertex, (N, 3) in nanometres, ``(z, y, x)``."""
        return self._sigmas

    @sigmas.setter
    def sigmas(self, sigmas):
        self._sigmas = sigmas
        self._update_sigmas_buffer(sigmas)

    def _update_sigmas_buffer(self, sigmas):
        if self._attached and self._visual is not None:
            self._sigmas_buffer.set_data(sigmas[:, ::-1], convert=True)

    def _attach(self, visual):

        # the full projection model view
        self.vshader["transform"] = visual.transforms.get_transform("visual", "render")
        # each vertex's own position, exactly as drawn
        self.vshader["quad_corner"] = mesh_vertex_buffer(visual)

        # Screen axes back into world space, for the billboard basis and the
        # projected covariance.  The forward transform is not needed.
        self.vshader["camera_inv"] = visual.transforms.get_transform(
            "document", "scene"
        )
        # canvas pixels, for the minimum on-screen size of a disc
        self.vshader["visual_to_canvas"] = visual.transforms.get_transform(
            "visual", "canvas"
        )
        super()._attach(visual)


class Particles(Surface):
    """Billboarded particle layer that renders camera facing quads of given size
    Can be combined with other (e.g. texture) filter to create particle systems etc
    """

    def __init__(
        self,
        coords,
        size=10,
        sigmas=(1, 1, 1),
        values=1,
        filter=_DEFAULT_FILTER,
        antialias=False,
        **kwargs,
    ):

        kwargs.setdefault("shading", "none")
        kwargs.setdefault("blending", "additive")

        # float32 throughout: the GPU consumes float32 regardless, so a float64
        # source array only doubles the size of every derived buffer.  At the
        # ~1e5 nm coordinate range used here float32 resolves to <0.01 nm, two
        # orders of magnitude below single-molecule localization precision.
        coords = np.asarray(coords, dtype=np.float32)
        sigmas = np.asarray(sigmas, dtype=np.float32)

        if np.isscalar(values):
            values = values * np.ones(len(coords), dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        values = np.broadcast_to(values, len(coords))
        size = np.broadcast_to(np.asarray(size, dtype=np.float32), len(coords))
        sigmas = np.broadcast_to(sigmas, (len(coords), 3))

        if not coords.ndim == 2:
            raise ValueError("coords should be of shape (M,D)")
        if len(coords) == 0:
            raise ValueError("Particles requires at least one localization")

        if not len(size) == len(coords) == len(sigmas):
            raise ValueError()

        # add dummy z if 2d coords
        if coords.shape[1] == 2:
            coords = np.concatenate([np.zeros((len(coords), 1)), coords], axis=-1)

        assert coords.shape[-1] == sigmas.shape[-1] == 3

        vertices, faces = generate_billboards_2d(coords, size=size)

        # The generator expands every centre to six vertices.
        vpp = 6
        centercoords = np.repeat(coords, vpp, axis=0)
        sigmas = np.repeat(sigmas, vpp, axis=0)
        values = np.repeat(values, vpp, axis=0)

        self._coords = coords
        self._centercoords = centercoords
        self._sigmas = sigmas
        self._size = size
        self._billboard_filter = BillboardsFilter(antialias=antialias)
        if filter is _DEFAULT_FILTER:
            filter = ShaderFilter("gaussian")
            shader_name = "gaussian"
        else:
            shader_name = None
        self.filter = filter
        self._viewer = None
        self._visual = None
        self._shader_name = shader_name
        self._footprint = FOOTPRINT_GAUSSIAN
        self._min_size_px = DEFAULT_MIN_SIZE_PX
        # Names of layer-list events we connected to, so close() can undo them.
        self._layer_event_connections = []
        super().__init__((vertices, faces, values), **kwargs)

    def update_particle_data(self, coords, size, sigmas, values):
        """Update billboard geometry and attributes without replacing the layer.

        Appearance changes still need fresh quad geometry when their maximum
        Gaussian size changes, but they do not need a new napari Layer, VisPy
        visual, shader filter, or set of viewer callbacks.
        """
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2:
            raise ValueError("coords should be of shape (M,D)")
        if len(coords) == 0:
            raise ValueError("Particles requires at least one localization")
        if coords.shape[1] == 2:
            coords = np.concatenate(
                [np.zeros((len(coords), 1), dtype=np.float32), coords], axis=-1
            )

        size = np.broadcast_to(np.asarray(size, dtype=np.float32), len(coords))
        sigmas = np.broadcast_to(np.asarray(sigmas, dtype=np.float32), (len(coords), 3))
        values = np.broadcast_to(np.asarray(values, dtype=np.float32), len(coords))

        vertices, faces = generate_billboards_2d(coords, size=size)
        vertices_per_particle = 6
        self._coords = coords
        self._size = size
        self._centercoords = np.repeat(coords, vertices_per_particle, axis=0)
        self._sigmas = np.repeat(sigmas, vertices_per_particle, axis=0)
        vertex_values = np.repeat(values, vertices_per_particle, axis=0)

        # Surface.data emits napari's normal data event, updating the existing
        # VisPy visual.  Our attributes are assigned first because the ensuing
        # slice may immediately ask the billboard filter for matching buffers.
        self.data = (vertices, faces, vertex_values)
        self._update_billboard_filter()

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        super()._set_view_slice()
        self._update_billboard_filter()

    def _update_billboard_filter(self):
        """Upload the per-vertex attributes.

        Each is repeated six times per localization, so every vertex of a
        quad carries the same values and the order in which the visual
        consumes them cannot matter.  The quad coordinate itself is not
        uploaded: the shader derives it from the vertex position.  Until 3.1
        texture coordinates were uploaded, and since they differ per corner
        they had to agree with a vertex order that napari and VisPy do not
        keep; one triangle of every quad had its coordinates mirrored.
        """
        if self._billboard_filter._attached:
            if self._centercoords is not None:
                self._billboard_filter.centercoords = self._centercoords[:, -3:]
            self._billboard_filter.sigmas = self._sigmas[:, -3:]

    @property
    def localization_coords(self):
        """The localization centres being drawn, as an (N, 3) array in (z, x, y).

        One row per localization on both Gaussian backends, which is what makes
        it the right thing for a caller to ask for. This layer also keeps a
        six-vertex expansion of these in `data`; this is not that.
        """
        return self._coords

    @property
    def n_localizations(self):
        """How many localizations this layer is drawing."""
        return len(self._coords)

    @property
    def billboard_size_nm(self):
        """Edge length of the largest splat drawn, in nanometres.

        One value per localization here and one scalar per dataset on the
        instanced backend, so the screen-space cap -- a statement about the
        widest splat either way -- can be checked without knowing which.
        """
        return float(np.max(self._size))

    @property
    def filter(self):
        """The filter property."""
        return self._filter

    @filter.setter
    def filter(self, value):
        if value is None:
            value = ()
        elif not isinstance(value, Iterable):
            value = (value,)
        self._filter = tuple(value)

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.
        Returns
        -------
        extent_data : array, shape (2, D)
        """
        if len(self._coords) == 0:
            extrema = np.full((2, self.ndim), np.nan)
        else:
            size = np.repeat(self._size[:, np.newaxis], self.ndim, axis=-1)
            size[:, :-2] *= 0
            maxs = np.max(self._coords + 0.5 * size, axis=0)
            mins = np.min(self._coords - 0.5 * size, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema

    @property  # LR
    def coords(self):
        return self._coords

    @coords.setter  # LR
    def coords(self, coords):
        self._coords = coords

    @property
    def shader(self):
        """Name of the fragment shader used to draw each billboard.

        Deliberately *not* called ``shading``.  napari's Surface layer owns a
        property of that name and forwards its value straight to VisPy, which
        accepts only ``None``, ``'flat'`` and ``'smooth'``::

            _on_shading_change -> self.node.shading = self.layer.shading
            MeshVisual.shading -> assert shading in (None, 'flat', 'smooth')

        Overriding it meant napari pushed ``'gaussian'`` into VisPy and raised
        AssertionError as soon as anything re-sliced the layer -- which napari
        does to every layer whenever the scene extent changes, i.e. as soon as a
        second dataset covering a different area is loaded.  Our Gaussian
        shading is implemented by a shader filter, so napari's own ``shading``
        stays at ``'none'`` and is left alone.
        """
        return self._shader_name

    @shader.setter
    def shader(self, name):
        self._shader_name = name
        self._detach_filter()
        self.filter = ShaderFilter(name)
        self._attach_filter()

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
        # A layer built with a filter of its own keeps it while it stays a
        # Gaussian one; any palette footprint replaces a palette shader.
        if self._shader_name != self._footprint and (
            self._footprint != FOOTPRINT_GAUSSIAN or self._shader_name in FOOTPRINTS
        ):
            self.shader = self._footprint
        self._apply_blend_state()
        if self._visual is not None:
            self._visual.update()

    def _detach_filter(self):
        if self._visual is None:
            return
        for f in self.filter:
            self._visual.detach(f)

    def _attach_filter(self):
        if self._visual is None:
            return
        for f in self.filter:
            self._visual.attach(f)

    def get_visual(self, viewer):
        return get_layer_visual(viewer, self)

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

    def add_to_viewer(self, viewer):
        self._viewer = viewer
        self._viewer.add_layer(self)

        # Get the vispy visual and attach our billboard filter
        self._visual = self.get_visual(viewer)
        self._visual.attach(self._billboard_filter)

        # Now that the filter is attached, its buffers can be populated.
        self._update_billboard_filter()

        # Attach any other shader filters (e.g. gaussian)
        self._attach_filter()

        self._apply_blend_state()

        # napari's own shading combo box is deliberately left alone.  We used to
        # clear it and refill it with the names in _shader_functions, but that
        # combo is wired to QtSurfaceControls.changeShading, which assigns
        # straight to napari's `shading` property:
        #
        #     self.layer.shading = self.shadingComboBox.currentData()
        #
        # so clearing it made the next signal assign None, and selecting one of
        # our entries assigned 'gaussian' -- both rejected by the Shading enum
        # with ValueError.  Our shader is selected through Particles.shader; it
        # is not one of napari's shading modes and does not belong in that
        # widget.  A selector for it belongs in the plugin's own controls.

    def close(self):
        """Release every resource this layer owns.

        Restores the blend setter, detaches the shader filters from the VisPy
        visual, and drops the viewer/visual references.  Safe to call more than
        once, and safe to call on a layer that was never added.
        """
        self._layer_event_connections = []

        if self._visual is not None:
            visual = self._visual
            release_additive_blending(visual)
            for shader_filter in self.filter:
                try:
                    visual.detach(shader_filter)
                except (ValueError, AttributeError, RuntimeError):
                    # It may already have been removed with the canvas.
                    pass
            try:
                visual.detach(self._billboard_filter)
            except (ValueError, AttributeError, RuntimeError):
                # The visual may already have been torn down with the canvas.
                pass

        self._visual = None
        self._viewer = None

    # Alias: "detach" reads better at renderer call sites, "close" matches the
    # lifecycle vocabulary used elsewhere in the plan.
    detach = close
