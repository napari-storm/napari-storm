"""
A billboarded particle layer with texture/shader support

"""

from collections.abc import Iterable

import numpy as np
from napari.layers import Surface
from vispy.gloo import VertexBuffer
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function, Varying

from ._napari_compat import (
    force_additive_blending,
    get_layer_visual,
    release_additive_blending,
)
from .filters import ShaderFilter
from .utils import generate_billboards_2d


_DEFAULT_FILTER = object()


class BillboardsFilter(Filter):
    """Billboard geometry filter (transforms vertices to always face camera)"""

    def __init__(self, antialias=0):
        vmat_inv = Function(
            """
            mat2 inverse(mat2 m) {
                return mat2(m[1][1],-m[0][1],-m[1][0], m[0][0]) / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
            }
        """
        )

        vfunc = Function(
            """
        varying float v_z_center;
        varying float v_scale_intensity;
        varying mat2 covariance_inv;

        void apply(){
            // original world coordinates of the (constant) particle squad, e.g. [5,5] for size 5
            vec4 pos = $transform_inv(gl_Position);

            pos.z *= pos.w;

            vec2 tex = $texcoords;

            mat4 cov = mat4(1.0);

            cov[0][0] = sqrt($sigmas[0]);
            cov[1][1] = sqrt($sigmas[1]);
            cov[2][2] = sqrt($sigmas[2]);

            // get new inverse covariance matrix (for rotating a gaussian)
            vec4 ex = vec4(1,0,0,0);
            vec4 ey = vec4(0,1,0,0);
            vec4 ez = vec4(0,0,1,0);
            vec3 ex2 = $camera(cov*$camera_inv(ex)).xyz;
            vec3 ey2 = $camera(cov*$camera_inv(ey)).xyz;
            vec3 ez2 = $camera(cov*$camera_inv(ez)).xyz;
            mat3 Rmat = mat3(ex2, ey2, ez2);
            covariance_inv = mat2(transpose(Rmat)*mat3(cov)*Rmat);
            covariance_inv = $inverse(covariance_inv);


            // get first and second column of view (which is the inverse of the camera)
            vec3 camera_right = $camera_inv(vec4(1,0,0,0)).xyz;
            vec3 camera_up    = $camera_inv(vec4(0,1,0,0)).xyz;

            // when particles become too small, lock texture size and apply antialiasing (only used when antialias=1)
            // decrease this value to increase antialiasing
            //float dist_cutoff = .2 * max(abs(pos.x), abs(pos.y));

            // increase this value to increase antialiasing
            float dist_cutoff = $antialias;

            float len = length(camera_right);

            //camera_right = normalize(camera_right);
            //camera_up    = normalize(camera_up);

            camera_right = camera_right/len;
            camera_up    = camera_up/len;

            vec4 p1 = $transform(vec4($vertex_center.xyz + camera_right*pos.x + camera_up*pos.y, 1.));
            vec4 p2 = $transform(vec4($vertex_center,1));
            float dist = length(p1.xy/p1.w-p2.xy/p2.w);


            // if antialias and far away zoomed out, keep sprite size constant and shrink texture...
            // else adjust sprite size
            if (($antialias>0) && (dist<dist_cutoff)) {

                float scale = dist_cutoff/dist;
                tex = .5+(tex-.5)*clamp(scale,1,10);

                camera_right = camera_right*scale;
                camera_up    = camera_up*scale;
                v_scale_intensity = scale;

            }
            vec3 pos_real  = $vertex_center.xyz + camera_right*pos.x + camera_up*pos.y;
            gl_Position = $transform(vec4(pos_real, 1.));
            vec4 center = $transform(vec4($vertex_center,1));
            v_z_center = center.z/center.w;

            // flip tex coords neccessary since 0.4.13 and vispy bump
            // TODO: investigate

            $v_texcoords = vec2(tex.y, tex.x);
            }
        """
        )

        ffunc = Function(
            """
        varying float v_scale_intensity;
        varying float v_z_center;

        void apply() {
            gl_FragDepth = v_z_center;
            $texcoords;
        }
        """
        )

        self._texcoord_varying = Varying("v_texcoord", "vec2")
        vfunc["inverse"] = vmat_inv
        vfunc["v_texcoords"] = self._texcoord_varying
        ffunc["texcoords"] = self._texcoord_varying

        self._texcoords_buffer = VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        vfunc["texcoords"] = self._texcoords_buffer
        vfunc["antialias"] = float(antialias)

        self._centercoords_buffer = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._sigmas_buffer = VertexBuffer(np.zeros((0, 3), dtype=np.float32))

        vfunc["vertex_center"] = self._centercoords_buffer
        vfunc["sigmas"] = self._sigmas_buffer

        super().__init__(vcode=vfunc, vhook="post", fcode=ffunc, fhook="post")

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
        """The vertex center coordinates as an (N, 3) array of floats."""
        return self._sigmas

    @sigmas.setter
    def sigmas(self, sigmas):
        self._sigmas = sigmas
        self._update_sigmas_buffer(sigmas)

    def _update_sigmas_buffer(self, sigmas):
        if self._attached and self._visual is not None:
            self._sigmas_buffer.set_data(sigmas[:, ::-1], convert=True)

    @property
    def texcoords(self):
        """The texture coordinates as an (N, 2) array of floats."""
        return self._texcoords

    @texcoords.setter
    def texcoords(self, texcoords):
        self._texcoords = texcoords
        self._update_texcoords_buffer(texcoords)

    def _update_texcoords_buffer(self, texcoords):
        if self._attached and self._visual is not None:
            self._texcoords_buffer.set_data(texcoords[:, ::-1], convert=True)

    def _attach(self, visual):

        # the full projection model view
        self.vshader["transform"] = visual.transforms.get_transform("visual", "render")
        # the inverse of it
        self.vshader["transform_inv"] = visual.transforms.get_transform(
            "render", "visual"
        )

        # the modelview
        self.vshader["camera_inv"] = visual.transforms.get_transform(
            "document", "scene"
        )
        # inverse of it
        self.vshader["camera"] = visual.transforms.get_transform("scene", "document")
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

        vertices, faces, texcoords = generate_billboards_2d(coords, size=size)

        # The generator expands every centre to six vertices.
        vpp = 6
        centercoords = np.repeat(coords, vpp, axis=0)
        sigmas = np.repeat(sigmas, vpp, axis=0)
        values = np.repeat(values, vpp, axis=0)

        self._coords = coords
        self._centercoords = centercoords
        self._sigmas = sigmas
        self._size = size
        self._texcoords = texcoords
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
        # Names of layer-list events we connected to, so close() can undo them.
        self._layer_event_connections = []
        super().__init__((vertices, faces, values), texcoords=texcoords, **kwargs)

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

        size = np.broadcast_to(
            np.asarray(size, dtype=np.float32), len(coords)
        )
        sigmas = np.broadcast_to(
            np.asarray(sigmas, dtype=np.float32), (len(coords), 3)
        )
        values = np.broadcast_to(
            np.asarray(values, dtype=np.float32), len(coords)
        )

        vertices, faces, texcoords = generate_billboards_2d(coords, size=size)
        vertices_per_particle = 6
        self._coords = coords
        self._size = size
        self._centercoords = np.repeat(coords, vertices_per_particle, axis=0)
        self._sigmas = np.repeat(sigmas, vertices_per_particle, axis=0)
        self._texcoords = texcoords
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
        """Upload attributes in the same vertex order as the Surface visual.

        ``Surface._view_faces`` is an index buffer; it does not reorder the
        visual's vertex buffer.  Indexing these attributes by flattened faces
        therefore assigned the second triangle's texture coordinates to the
        wrong vertices after every in-place update, visually splitting each
        Gaussian along the quad diagonal.
        """
        if self._billboard_filter._attached:
            if self._texcoords is not None:
                self._billboard_filter.texcoords = self._texcoords
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
        """Force true additive blending, and keep it forced.

        Until P0-01 additive blending was repaired by accident -- every update
        destroyed and rebuilt the layer, and `add_to_viewer` set the state again
        on the way back in.  Updating in place removed the rebuild and with it
        the repair, so it has to be asserted deliberately.  See
        `force_additive_blending` for why it is asserted by wrapping the setter
        rather than from event handlers.
        """
        if self._visual is None:
            return
        force_additive_blending(self._visual)

    def add_to_viewer(self, viewer):
        self._viewer = viewer
        self._viewer.add_layer(self)

        # Get the vispy visual and attach our billboard filter
        self._visual = self.get_visual(viewer)
        self._visual.attach(self._billboard_filter)

        # Populate filter buffers if we already have texcoords
        if self._texcoords is not None:
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
