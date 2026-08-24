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

* `render_size` is already one scalar per dataset — five times the largest sigma
  — so the quad can carry the size and only the *shape* varies per instance.
  Had the size been per-localization the quad could not have been shared.
* `_extent_data` is overridden. napari would otherwise frame the camera on a
  four-vertex quad at the origin instead of on the data.

**Requires VisPy's `gl+` backend**, selected process-wide before any GL context
exists. See `enable_instanced_backend()` in `_napari_compat`.
"""
from __future__ import annotations

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

__all__ = ["InstancedParticles", "InstancedBillboardsFilter"]

#: The quad, in units of the billboard edge.  Four vertices, for any N.
QUAD_XY = np.array(
    [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32
)
QUAD_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

# There is deliberately no texture-coordinate buffer here.  A per-vertex
# attribute has to be in the order the vertices are *drawn*, and that is not the
# order this file writes them: napari reverses the axis order of the vertices
# and reverses each face's winding to match, so faces [[0,1,2],[0,2,3]] reach
# the GPU as the vertex sequence 2,1,0,3,2,0.  A texcoord buffer written in the
# obvious order pairs the wrong coordinate with four of the six vertices; the
# two triangles then interpolate different maps, and every Gaussian is split
# along the quad diagonal.
#
# The shader derives the quad coordinate from the vertex position instead.
# Geometry cannot disagree with itself, so no ordering assumption survives.


class InstancedBillboardsFilter(Filter):
    """Camera-facing expansion and Gaussian falloff, from per-instance buffers.

    The same geometry as the non-instanced `BillboardsFilter`: recover the
    quad-local offset, rebuild it against the camera basis, and shade a
    Gaussian whose covariance follows the camera. What differs is where the
    per-localization values come from — divisor-1 buffers rather than the same
    number repeated across six vertices.
    """

    def __init__(self):
        vmat_inv = Function(
            """
            mat2 inverse(mat2 m) {
                return mat2(m[1][1],-m[0][1],-m[1][0], m[0][0])
                       / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
            }
        """
        )
        vfunc = Function(
            """
        varying float v_z_center;
        varying float v_instance_value;
        varying mat2 covariance_inv;

        void apply(){
            // The quad corner, in world units, recovered the same way the
            // non-instanced filter does it.
            vec4 pos = $transform_inv(gl_Position);
            pos.z *= pos.w;

            mat4 cov = mat4(1.0);
            cov[0][0] = sqrt($sigmas[0]);
            cov[1][1] = sqrt($sigmas[1]);
            cov[2][2] = sqrt($sigmas[2]);

            vec4 ex = vec4(1,0,0,0);
            vec4 ey = vec4(0,1,0,0);
            vec4 ez = vec4(0,0,1,0);
            vec3 ex2 = $camera(cov*$camera_inv(ex)).xyz;
            vec3 ey2 = $camera(cov*$camera_inv(ey)).xyz;
            vec3 ez2 = $camera(cov*$camera_inv(ez)).xyz;
            mat3 Rmat = mat3(ex2, ey2, ez2);
            covariance_inv = mat2(transpose(Rmat)*mat3(cov)*Rmat);
            covariance_inv = $inverse(covariance_inv);

            vec3 camera_right = $camera_inv(vec4(1,0,0,0)).xyz;
            vec3 camera_up    = $camera_inv(vec4(0,1,0,0)).xyz;
            float len = length(camera_right);
            camera_right = camera_right/len;
            camera_up    = camera_up/len;

            vec3 pos_real = $vertex_center.xyz
                          + camera_right*pos.x + camera_up*pos.y;
            gl_Position = $transform(vec4(pos_real, 1.));

            vec4 center = $transform(vec4($vertex_center,1));
            v_z_center = center.z/center.w;
            v_instance_value = $instance_value;

            // Where this corner sits on the quad, in [-0.5, 0.5], taken from
            // the geometry rather than from a parallel attribute that would
            // have to agree with the drawn vertex order.
            $v_texcoords = pos.xy / $billboard_size + 0.5;
        }
        """
        )
        ffunc = Function(
            """
        varying float v_z_center;
        varying float v_instance_value;
        varying mat2 covariance_inv;

        void apply() {
            gl_FragDepth = v_z_center;
            vec2 x = 2.0*($texcoords - 0.5);
            float gaussian = exp(-2.0*dot(x, covariance_inv*x));
            // napari puts the layer's opacity in the incoming alpha, and under
            // additive blending alpha is the *only* thing scaling what this
            // layer contributes.  Overwriting it -- which this shader used to
            // do -- silently disabled both the per-channel opacity slider and
            // the show/hide checkbox, which hides a channel by setting opacity
            // to zero.  Multiply into it instead.
            float layer_alpha = gl_FragColor.a;
            // The colormap is sampled *here*, with this instance's own value.
            // It cannot come from the incoming colour: napari looks the
            // colormap up per mesh vertex, and the mesh is one quad whose four
            // vertices all carry 1.0, so every localization came back the same
            // hue and only its brightness varied.  Z-colour-coding rendered as
            // a red-to-pink ramp instead of a rainbow because of it.
            //
            // The contrast window is applied first, exactly as vispy does it
            // for a mesh: napari hands the visual `node.clim` and the value is
            // normalised into that window before the lookup.  Sampling the
            // colormap ourselves means applying it ourselves too -- without
            // this the colormap-range slider and the cutoff/factor boxes in
            // ChannelControls change the layer state and nothing else.
            float t = clamp(
                (v_instance_value - $clim_low) / $clim_range, 0.0, 1.0
            );
            vec4 mapped = $cmap(t);
            // The falloff goes in alpha as well as colour so additive blending
            // squares it, matching the shader this replaces.
            gl_FragColor = mapped * gaussian;
            gl_FragColor.a = gaussian * layer_alpha;
        }
        """
        )

        # A grey ramp until the real colormap is available; `set_colormap`
        # swaps in the visual's own GLSL so the two never disagree.
        self._cmap_function = Function(
            "vec4 napari_storm_default_cmap(float t) { return vec4(t, t, t, 1.0); }"
        )
        ffunc["cmap"] = self._cmap_function
        # Identity window until the layer says otherwise.
        ffunc["clim_low"] = 0.0
        ffunc["clim_range"] = 1.0

        self._texcoord_varying = Varying("v_texcoord", "vec2")
        vfunc["inverse"] = vmat_inv
        vfunc["v_texcoords"] = self._texcoord_varying
        ffunc["texcoords"] = self._texcoord_varying

        # One scalar for the whole dataset: the billboard edge, used to turn a
        # vertex position back into a quad coordinate.
        vfunc["billboard_size"] = 1.0

        # Per-instance: one row per localization.  divisor=1 is what turns the
        # draw into glDrawElementsInstanced -- gloo infers the instance count
        # from these buffers' length.
        self._centers_buffer = VertexBuffer(
            np.zeros((1, 3), dtype=np.float32), divisor=1
        )
        self._sigmas_buffer = VertexBuffer(
            np.ones((1, 3), dtype=np.float32), divisor=1
        )
        self._values_buffer = VertexBuffer(
            np.ones((1,), dtype=np.float32), divisor=1
        )
        vfunc["vertex_center"] = self._centers_buffer
        vfunc["sigmas"] = self._sigmas_buffer
        vfunc["instance_value"] = self._values_buffer

        super().__init__(vcode=vfunc, vhook="post", fcode=ffunc, fhook="post")

    def set_colormap(self, colormap):
        """Use *colormap*'s own GLSL, so our lookup matches napari's exactly."""
        glsl = getattr(colormap, "glsl_map", None)
        if not glsl:
            return
        self._cmap_function = Function(glsl)
        self.fshader["cmap"] = self._cmap_function

    def set_contrast_limits(self, low, high):
        """Normalise instance values into ``[low, high]`` before the colormap.

        A zero-width window would divide by zero in the shader; napari allows
        one transiently while a slider is dragged, so it is floored here rather
        than trusted.
        """
        low, high = float(low), float(high)
        self.fshader["clim_low"] = low
        self.fshader["clim_range"] = max(high - low, 1e-8)

    def set_billboard_size(self, size):
        """The quad edge, needed to recover the quad coordinate in the shader."""
        self.vshader["billboard_size"] = float(size)

    def set_instances(self, centers, sigmas, values):
        """Upload one row per localization."""
        self._centers_buffer.set_data(
            np.ascontiguousarray(centers[:, ::-1], dtype=np.float32), convert=True
        )
        self._sigmas_buffer.set_data(
            np.ascontiguousarray(sigmas[:, ::-1], dtype=np.float32), convert=True
        )
        self._values_buffer.set_data(
            np.ascontiguousarray(values, dtype=np.float32), convert=True
        )

    def _attach(self, visual):
        self.vshader["transform"] = visual.transforms.get_transform(
            "visual", "render"
        )
        self.vshader["transform_inv"] = visual.transforms.get_transform(
            "render", "visual"
        )
        self.vshader["camera_inv"] = visual.transforms.get_transform(
            "document", "scene"
        )
        self.vshader["camera"] = visual.transforms.get_transform(
            "scene", "document"
        )
        super()._attach(visual)


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
        self._viewer = None
        self._visual = None
        self._layer_event_connections = []

        vertices, faces, values4 = self._quad_mesh()
        super().__init__((vertices, faces, values4), **kwargs)

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
        self._upload_instances()
        self._apply_colormap()
        self._apply_contrast_limits()
        self.events.colormap.connect(self._apply_colormap)
        self.events.contrast_limits.connect(self._apply_contrast_limits)
        self._apply_blend_state()

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

    def _apply_colormap(self, event=None):
        """Hand the visual's colormap to the shader that samples it."""
        if self._visual is None:
            return
        colormap = getattr(self._visual, "cmap", None)
        if colormap is not None:
            self._billboard_filter.set_colormap(colormap)

    def _apply_contrast_limits(self, event=None):
        """Push the layer's contrast window into the shader that uses it."""
        limits = getattr(self, "contrast_limits", None)
        if limits is not None and len(limits) == 2:
            self._billboard_filter.set_contrast_limits(limits[0], limits[1])

    def close(self):
        self._layer_event_connections = []
        for emitter, handler in (
            ("colormap", self._apply_colormap),
            ("contrast_limits", self._apply_contrast_limits),
        ):
            try:
                getattr(self.events, emitter).disconnect(handler)
            except (ValueError, TypeError, RuntimeError):
                pass
        if self._visual is not None:
            release_additive_blending(self._visual)
            try:
                self._visual.detach(self._billboard_filter)
            except (ValueError, AttributeError, RuntimeError):
                pass
        self._visual = None
        self._viewer = None

    detach = close
