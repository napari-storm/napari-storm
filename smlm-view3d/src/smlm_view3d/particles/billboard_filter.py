"""The instanced billboard filter, without napari.

The camera-facing billboards of the instanced backend: a VisPy `Filter` that
attaches to a mesh visual whose mesh is one quad (`QUAD_XY`, `QUAD_FACES`) and
draws that quad once per localization from divisor-1 instance buffers.  Nothing
here imports napari, so a plain VisPy scene can host it as well as napari's
`Surface` layer (`instanced_layer.InstancedParticles`).

Moved unchanged from `instanced_layer.py` (`QUAD_XY`, `QUAD_FACES`,
`InstancedBillboardsFilter`) and `_napari_compat.py` (`NapariInternalsChanged`,
`mesh_vertex_buffer`).  Both import them from here, so every old import keeps
working.  The moved comments still speak of napari, the host they were
written for.
"""

from __future__ import annotations

import numpy as np
from vispy.gloo import VertexBuffer
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function, Varying

from ..core.footprints import FOOTPRINT_GAUSSIAN, footprint_named
from .footprint_shaders import is_opaque, quad_scale, shape_function


class NapariInternalsChanged(RuntimeError):
    """Raised when a required napari internal is missing on this version."""


def mesh_vertex_buffer(visual):
    """The VertexBuffer a mesh visual draws its vertices from, in drawn order.

    A shader that needs each vertex's own position reads it from here instead
    of recovering it by inverting the transforms.  Through a perspective
    projection that inversion loses a vertex far from the camera entirely:
    VisPy puts the near plane at 1/3162 of the eye distance, so for anything
    much farther away the depth row subtracts two all-but-equal float32
    numbers, and the error is multiplied by the camera's distance from the
    vertex.  It is VisPy's private ``MeshVisual._vertices``, filled with the
    face-expanded vertices that ``position`` is bound to, so it cannot
    disagree with what is drawn.
    """
    buffer = getattr(visual, "_vertices", None)
    if buffer is None:
        raise NapariInternalsChanged(
            "the mesh visual no longer exposes its vertex buffer as _vertices"
        )
    return buffer


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
        vmat_inv = Function("""
            mat2 inverse(mat2 m) {
                return mat2(m[1][1],-m[0][1],-m[1][0], m[0][0])
                       / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
            }
        """)
        vfunc = Function("""
        varying float v_z_center;
        varying float v_instance_value;
        varying mat2 v_disc_inv;
        varying float v_focus;

        void apply(){
            // The quad corner, in visual units, read from the mesh's own
            // vertex buffer.  It used to be recovered by inverting the
            // transforms on gl_Position, which through the perspective camera
            // came apart for a quad far from the camera -- and this quad sits
            // near the origin, wherever the data is: zoomed in and tilted
            // away from the origin, every splat flickered and was cut off
            // from changing sides.  See mesh_vertex_buffer.
            vec3 pos = $quad_corner;

            // The screen axes as unit vectors in world space.
            vec3 camera_right = $camera_inv(vec4(1,0,0,0)).xyz;
            vec3 camera_up    = $camera_inv(vec4(0,1,0,0)).xyz;
            float len = length(camera_right);
            camera_right = camera_right/len;
            camera_up    = camera_up/len;

            // The localization's one-sigma ellipse on screen: what the
            // Gaussian is drawn with, and what the footprints that draw an
            // outline measure against.  Its covariance in the screen basis is
            // the marginal of the axis-aligned diag(sigma^2) -- what an
            // orthographic projection of the 3-D Gaussian looks like.
            // $sigmas are in nanometres and the drawn square reaches
            // half_edge nanometres to either side, so dividing by its square
            // puts the ellipse in the square's own -1 to 1 coordinate.
            //
            // Until 3.1 the Gaussian was drawn from a second covariance, built
            // of the square roots of the sigmas and sandwiched between two
            // rotations, which drew widths going as sigma to the power 0.75 --
            // and its falloff was squared by the blend on top.
            vec3 rs = camera_right * $sigmas;
            vec3 us = camera_up * $sigmas;
            float half_edge = 0.5 * $billboard_size * $quad_scale;
            mat2 disc_cov = mat2(dot(rs, rs), dot(rs, us),
                                 dot(rs, us), dot(us, us))
                          / (half_edge * half_edge);
            v_disc_inv = $inverse(disc_cov);

            // A one-sigma outline smaller on screen than $min_half_px (a
            // radius, in canvas pixels) is enlarged to it, so a visualisation
            // stays visible when zoomed out.  Measured on the ellipse's
            // equal-area radius, the square root of its semi-axes' product.
            // Zero for the scientific Gaussian, which is never enlarged: its
            // summed intensity is the measurement.
            float grow = 1.0;
            if ($min_half_px > 0.0) {
                vec4 c = $visual_to_canvas(vec4($vertex_center, 1.0));
                vec4 e = $visual_to_canvas(
                    vec4($vertex_center + camera_right * half_edge, 1.0));
                float det = disc_cov[0][0] * disc_cov[1][1]
                          - disc_cov[0][1] * disc_cov[1][0];
                float radius_px = length(e.xy / e.w - c.xy / c.w)
                                * sqrt(sqrt(max(det, 0.0)));
                if (radius_px > 0.0 && radius_px < $min_half_px) {
                    grow = $min_half_px / radius_px;
                }
            }

            // $quad_scale shrinks the billboard to the square the footprint
            // needs; the quad coordinate below is taken before scaling, so it
            // still runs -1 to 1 across whatever is drawn.
            vec3 pos_real = $vertex_center.xyz
                          + (camera_right*pos.x + camera_up*pos.y)
                            * ($quad_scale * grow);
            gl_Position = $transform(vec4(pos_real, 1.));

            vec4 center = $transform(vec4($vertex_center,1));
            v_z_center = center.z/center.w;
            v_instance_value = $instance_value;

            // Focus mode: what this localization keeps of its opacity, 1
            // within the radius of the focus centre and easing down to the
            // outside opacity across the edge -- core.focus.focus_weight, on
            // the GPU, so the region follows the camera without a redraw of
            // the data.  $focus_axes zeroes the line of sight in a 2-D view.
            v_focus = 1.0;
            if ($focus_on > 0.5) {
                vec3 offset = ($vertex_center - $focus_center) * $focus_axes;
                float over = (length(offset) - $focus_radius) / $focus_edge;
                v_focus = mix(1.0, $focus_floor,
                              smoothstep(0.0, 1.0, clamp(over, 0.0, 1.0)));
            }

            // Where this corner sits on the quad, in [-0.5, 0.5], taken from
            // the geometry rather than from a parallel attribute that would
            // have to agree with the drawn vertex order.
            $v_texcoords = pos.xy / $billboard_size + 0.5;
        }
        """)
        ffunc = Function("""
        varying float v_z_center;
        varying float v_instance_value;
        varying mat2 v_disc_inv;
        varying float v_focus;

        void apply() {
            // Faded out completely by focus mode: nothing to draw.
            if (v_focus <= 0.0) {
                discard;
            }
            // Window-space depth, which is what the depth test compares: NDC
            // mapped onto the default depth range [0, 1], which neither VisPy
            // nor napari changes.  This used to write v_z_center itself -- NDC,
            // -1 to 1 -- and it went unnoticed only because Gaussians are
            // drawn with the depth test off.  Under one, everything nearer
            // than the middle of the depth range clamped to the same depth and
            // occlusion followed draw order.
            gl_FragDepth = 0.5 * v_z_center + 0.5;
            vec2 x = 2.0*($texcoords - 0.5);
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
            // Squared distance from the centre in sigmas of this localization.
            float q = dot(x, v_disc_inv*x);
            // What the footprint puts on the square: see footprint_shaders.
            // An opaque one is drawn at alpha 1 -- napari blends the
            // bottom-most layer with the canvas even under its "opaque"
            // preset, at whatever alpha arrives -- and not at all at opacity
            // 0, which is how a channel is switched off.  The Gaussian's
            // falloff goes into alpha alone, so additive blending applies it
            // once.
            if ($opaque > 0.5 && layer_alpha <= 0.0) {
                discard;
            }
            if ($accumulate > 0.5 && $mean_colour > 0.5) {
                // Colour coding through the sum (see summed_contrast.py): add
                // this localization's colour times its footprint weight w to
                // RGB, and w to alpha, so the resolve can divide the two into
                // the weighted mean colour.  Both go out scaled by sqrt(w),
                // because the additive blend (src_alpha, one) multiplies every
                // channel by the alpha written: RGB adds colour * w and alpha
                // adds w, with no change of blend state.  The value is already
                // a position on the colormap; the window acts on the weight.
                // Focus mode scales the weight: fainter, same colour.
                vec4 weight = $shape(x, q, 0.25 * q, vec4(1.0));
                float root = sqrt(max(weight.r * weight.a * v_focus, 0.0));
                vec3 colour = $cmap(clamp(v_instance_value, 0.0, 1.0)).rgb;
                gl_FragColor = vec4(colour * root, root);
            } else if ($accumulate > 0.5) {
                // Summed contrast (see summed_contrast.py): write only what
                // this localization adds here -- its value times what the
                // footprint puts on the square, colour times alpha exactly as
                // additive blending combines them below -- at alpha 1.  The
                // window, colormap and opacity are applied once, to the sum.
                vec4 weight = $shape(x, q, 0.25 * q, vec4(1.0));
                gl_FragColor = vec4(
                    v_instance_value * weight.r * weight.a * v_focus,
                    0.0, 0.0, 1.0
                );
            } else {
                vec4 drawn = $shape(x, q, 0.25 * q,
                                    vec4(mapped.rgb, layer_alpha));
                if ($opaque > 0.5) {
                    drawn.a = 1.0;
                }
                drawn.a *= v_focus;
                gl_FragColor = drawn;
            }
        }
        """)

        # A grey ramp until the real colormap is available; `set_colormap`
        # swaps in the visual's own GLSL so the two never disagree.
        self._cmap_function = Function(
            "vec4 napari_storm_default_cmap(float t) { return vec4(t, t, t, 1.0); }"
        )
        ffunc["cmap"] = self._cmap_function
        # Identity window until the layer says otherwise.
        ffunc["clim_low"] = 0.0
        ffunc["clim_range"] = 1.0
        # The scientific Gaussian until `set_footprint` says otherwise.
        ffunc["shape"] = shape_function(FOOTPRINT_GAUSSIAN)
        ffunc["opaque"] = 0.0
        # Colours, until the layer hands the sum to a SummedContrastPass.
        ffunc["accumulate"] = 0.0
        ffunc["mean_colour"] = 0.0
        vfunc["quad_scale"] = 1.0
        vfunc["min_half_px"] = 0.0
        # Focus mode off until `set_focus` says otherwise.
        vfunc["focus_on"] = 0.0
        vfunc["focus_center"] = (0.0, 0.0, 0.0)
        vfunc["focus_axes"] = (1.0, 1.0, 1.0)
        vfunc["focus_radius"] = 1.0
        vfunc["focus_edge"] = 1.0
        vfunc["focus_floor"] = 1.0

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
        self._sigmas_buffer = VertexBuffer(np.ones((1, 3), dtype=np.float32), divisor=1)
        self._values_buffer = VertexBuffer(np.ones((1,), dtype=np.float32), divisor=1)
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

    def set_accumulate(self, accumulate, mean_colour=False):
        """Write summed weights (True) or windowed colours (False).

        *mean_colour* accumulates each localization's colour beside its weight
        instead, for colour coding; see `LayerAppearance.color_coded`.
        """
        self.fshader["accumulate"] = 1.0 if accumulate else 0.0
        self.fshader["mean_colour"] = 1.0 if mean_colour else 0.0

    def set_billboard_size(self, size):
        """The quad edge, needed to recover the quad coordinate in the shader."""
        self.vshader["billboard_size"] = float(size)

    def set_footprint(self, name, min_size_px=0.0):
        """Draw *name* from the palette, no smaller than *min_size_px* on screen.

        The floor never applies to the scientific Gaussian.
        """
        footprint = footprint_named(name)
        self.fshader["shape"] = shape_function(name)
        self.fshader["opaque"] = 1.0 if is_opaque(name) else 0.0
        self.vshader["quad_scale"] = quad_scale(name)
        self.vshader["min_half_px"] = (
            0.0 if footprint.reconstruction else 0.5 * float(min_size_px)
        )

    def set_focus(self, center_xyz=None, radius=1.0, edge=1.0, floor=1.0, planar=False):
        """Fade instances away from *center_xyz*, or stop fading with None.

        In the order the instance buffers hold positions: x, y, z, in the
        layer's own data coordinates.
        """
        if center_xyz is None:
            self.vshader["focus_on"] = 0.0
            return
        self.vshader["focus_on"] = 1.0
        self.vshader["focus_center"] = tuple(float(v) for v in center_xyz)
        self.vshader["focus_axes"] = (1.0, 1.0, 0.0) if planar else (1.0, 1.0, 1.0)
        self.vshader["focus_radius"] = float(radius)
        self.vshader["focus_edge"] = max(float(edge), 1e-6)
        self.vshader["focus_floor"] = float(floor)

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
        self.vshader["transform"] = visual.transforms.get_transform("visual", "render")
        self.vshader["quad_corner"] = mesh_vertex_buffer(visual)
        # Screen axes back into world space, for the billboard basis and the
        # projected covariance.  The forward transform is not needed.
        self.vshader["camera_inv"] = visual.transforms.get_transform(
            "document", "scene"
        )
        # Canvas pixels, for the minimum on-screen size of a disc: logical
        # pixels, so the floor reads the same on a high-density display.
        self.vshader["visual_to_canvas"] = visual.transforms.get_transform(
            "visual", "canvas"
        )
        super()._attach(visual)
