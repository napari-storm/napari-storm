"""The GLSL for every footprint in `core.footprints.PALETTE`.

One table, read by both shader backends, so a footprint cannot look one way on
the instanced backend and another on the billboard fallback.  Each entry is a
function

    vec4 shape(vec2 x, float q, float g2, vec4 colour)

* ``x`` -- where the fragment sits on the square being drawn, -1 to 1 across.
* ``q`` -- its squared distance from the centre in sigmas, measured on the
  localization's own one-sigma ellipse: ``q == 1`` is the outline.
* ``g2`` -- the quadratic form the scientific Gaussian has always been drawn
  with, kept so that footprint stays bit for bit what it was.
* ``colour`` -- the colormapped colour, with the layer's opacity in alpha.

and returns the fragment's colour, or discards it.  The caller forces alpha to
1 for an opaque footprint and discards everything while the layer's opacity is
0, so a hidden channel stays hidden whatever its footprint.

The entries after the markers are napari-particles' sprites (Martin Weigert,
BSD-3-Clause; see NOTICE), with their formulas unchanged.  There each returned
a factor the incoming colour was multiplied by; here each multiplies the colour
itself, which is the same arithmetic.
"""

from vispy.visuals.shaders import Function

from ..core.footprints import BLEND_OPAQUE, footprint_named
from ..core.render_planner import SIGMA_TO_SIZE_FACTOR

__all__ = ["SHAPES", "LEGACY_NAMES", "quad_scale", "shape_function", "is_opaque"]

SHAPES = {
    # -- the reconstruction
    "gaussian": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float g = exp(-2.0 * g2);
            return vec4(colour.rgb * g, colour.a * g);
        }""",
    # -- markers, drawn at the one-sigma outline
    "disc": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (q > 1.0) discard;
            return colour;
        }""",
    "outlined": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (q > 1.0) discard;
            float rim = smoothstep(0.6, 0.85, q);
            return vec4(mix(colour.rgb, vec3(0.05), rim), colour.a);
        }""",
    "ring": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (q > 1.0 || q < 0.62) discard;
            return colour;
        }""",
    "sphere": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (q > 1.0) discard;
            float h = sqrt(1.0 - q);
            vec3 normal = normalize(vec3(x.x, -x.y, h));
            vec3 light = normalize(vec3(-0.5, 0.6, 0.7));
            float diffuse = max(dot(normal, light), 0.0);
            float shine = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 30.0);
            return vec4(colour.rgb * (0.25 + 0.75 * diffuse) + 0.4 * shine, colour.a);
        }""",
    "cross": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            bool bar = (abs(x.x) < 0.22 && abs(x.y) < 0.95)
                    || (abs(x.y) < 0.22 && abs(x.x) < 0.95);
            if (!bar) discard;
            return colour;
        }""",
    "square": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (max(abs(x.x), abs(x.y)) > 0.8) discard;
            return colour;
        }""",
    "diamond": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            if (abs(x.x) + abs(x.y) > 0.95) discard;
            return colour;
        }""",
    # -- napari-particles' sprites, on the whole billboard
    "dome": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = length(x);
            float r0 = 0.8;
            if (r >= r0) discard;
            float val = sqrt(0.001 + r0 * r0 - r * r);
            return vec4(colour.rgb * val, colour.a);
        }""",
    "glow": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = length(x);
            float val = 0.05 / ((max(r, 0.01) - 0.01) + 0.05);
            return colour * val;
        }""",
    "bubble": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = length(x);
            float r1 = 0.8;
            float r2 = 0.9;
            float val = 0.0;
            if (r < r1)
                val = (sqrt(r2 * r2 - r * r) - sqrt(r1 * r1 - r * r))
                    / sqrt(r2 * r2 - r1 * r1);
            if (r < r2)
                val = sqrt(r2 * r2 - r * r) / sqrt(r2 * r2 - r1 * r1);
            else
                discard;
            return colour * val;
        }""",
    "bubble_thin": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = length(x);
            float r0 = 0.9;
            float val = exp(-400.0 * (r - r0) * (r - r0));
            if (r < r0) {
                val = max(val, r * r / r0 / r0);
            } else {
                discard;
            }
            return colour * val;
        }""",
    "airy": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = 8.0 * length(x);
            float val = abs(sin(r) / (1e-8 + r));
            return colour * val;
        }""",
    "fresnel": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float r = length(x);
            float d = 0.7;
            float val = 1.0;
            if (r > d) {
                val = exp(-4.0 * (r - d));
                val *= cos(1000.0 * (r - d) * (r - d));
            }
            return colour * val;
        }""",
    "fractal": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            vec2 c = vec2(-0.4, 0.6);
            const float r = 2.0;
            const int n = 100;
            int res = 0;
            for (int i = 0; i < n; i++) {
                res += int(length(x) < r);
                x = vec2(x.x * x.x - x.y * x.y, 2.0 * x.x * x.y);
                x = x + c;
            }
            float val = float(res) / float(n);
            return colour * val;
        }""",
    "gaussian_poly": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            float y = -2.0 * g2;
            float val = (120.0 + y * (120.0 + y * (60.0 + y * (20.0 + y * (5.0 + y)))))
                      * 0.0083333333;
            val = clamp(val, 0.0, 1.0);
            return colour * val;
        }""",
    "tile": """
        vec4 shape(vec2 x, float q, float g2, vec4 colour) {
            return colour;
        }""",
}

#: napari-particles' own names for its sprites, so code written against
#: `Particles.shader` keeps working.  Its "sphere" is not here: the palette's
#: "sphere" is the lit one, and napari-particles' is the palette's "dome".
LEGACY_NAMES = {
    "gaussian2": "gaussian_poly",
    "particle": "glow",
    "none": "tile",
    "bubble2": "bubble_thin",
}


def quad_scale(name):
    """The drawn square as a fraction of the Gaussian billboard, for *name*.

    The billboard is `SIGMA_TO_SIZE_FACTOR` widest sigmas across -- room for a
    Gaussian's tails -- so a footprint needing ``extent_sigmas`` either side of
    the centre needs this much of it.
    """
    return footprint_named(name).extent_sigmas / (0.5 * SIGMA_TO_SIZE_FACTOR)


def is_opaque(name):
    return footprint_named(name).blend == BLEND_OPAQUE


def shape_function(name):
    """A fresh VisPy Function for *name*'s shape."""
    footprint_named(name)  # a name the palette does not have is a ValueError
    return Function(SHAPES[name])
