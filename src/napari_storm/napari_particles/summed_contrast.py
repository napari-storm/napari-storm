"""Contrast applied to the summed image, not to each localization.

Additive footprints are summed by blending, and until now every one of them was
windowed and colormapped *before* the sum: each splat carried
``cmap(clamp((value - low) / (high - low)))`` into an 8-bit canvas that clipped
the total at 1.  The sum itself never existed anywhere, so the contrast window
could not act on it.  The lower limit could dim every splat alike, but it could
never hide sparse regions while keeping dense ones, which is what a lower limit
means on any image.

This draws such a layer in two passes instead:

1. the splats into an offscreen 32-bit float target, each contributing only its
   weight (its value times its footprint), with nothing clipped;
2. one full-screen pass that windows that sum, colormaps it and adds it to
   the canvas, at the layer's opacity.

The window's limits are therefore in units of summed weight.  With fixed-size
Gaussians every weight peaks at 1, so they count overlapping localizations.

**Colour coding** goes through the same two passes with a four-channel target.
Each splat adds its colour -- the colormap at its value, a property's position
on it -- times its weight, and the weight on its own; the resolve divides the
two into the weighted mean colour and scales it by the windowed weight.  So the
property sets the hue and density sets the brightness.  Adding colours straight
onto the canvas, as colour coding did before, saturated every colour with a
light component to white wherever localizations overlapped: grey, a diverging
map's pale middle, most of viridis.

It hooks the layer's VisPy node rather than napari: the node's ``draw`` is
shadowed on the instance, and everything else is left as napari drew it.  The
target is sized to whatever framebuffer is current, so screenshots rendered
into napari's own offscreen buffer at another size come out right, and the
viewport is left untouched, so the splats land on the same pixels they would
have hit on the canvas.  The resolve pass reads them back by ``gl_FragCoord``.
"""

from __future__ import annotations

import numpy as np
from vispy import gloo
from vispy.visuals.shaders import Function, ModularProgram

__all__ = ["SummedContrastPass"]

_RESOLVE_VERT = """
void main() {
    gl_Position = vec4($position, 0.0, 1.0);
}
"""

_RESOLVE_FRAG = """
void main() {
    float summed = texture2D($summed, gl_FragCoord.xy / $target_size).r;
    // Below the lower limit nothing is drawn -- not cmap(0).  Colormaps that
    // start at black would add nothing there anyway; one that does not would
    // otherwise tint the whole canvas, which is wrong for a layer that is
    // added to the others.
    if (summed <= $low) {
        discard;
    }
    float t = clamp((summed - $low) / $range, 0.0, 1.0);
    vec3 colour = $cmap(t).rgb;
    // The colormap's floor colour fades in from black just above the lower
    // limit.  A colormap that does not start at black -- viridis, turbo, hsv,
    // a host's own -- otherwise paints that colour on every pixel a splat
    // touches, however faint, out to the edge of its quad and then cuts to
    // black: every splat a hard-edged square.  Only cmap(0) is faded, so a
    // colormap that starts at black is left exactly as it was.
    float fade = smoothstep(0.0, $fade, summed - $low);
    colour = max(colour - $cmap(0.0).rgb * (1.0 - fade), 0.0);
    // The colormap's own alpha is ignored, as the per-localization path has
    // always ignored it.  napari-storm's channel colormaps ramp alpha with
    // colour from (0, 0, 0, 0); under (src_alpha, one) that alpha would
    // multiply the colour by itself and square every faint region away.
    gl_FragColor = vec4(colour, $opacity);
}
"""

_RESOLVE_MEAN_COLOUR_FRAG = """
void main() {
    vec4 summed = texture2D($summed, gl_FragCoord.xy / $target_size);
    float weight = summed.a;
    if (weight <= $low) {
        discard;
    }
    // The mean of the colours, weighted as the footprints overlap here, at the
    // brightness the window gives the summed weight.  It starts from black at
    // the lower limit, so no floor colour needs fading in.
    vec3 mean = summed.rgb / max(weight, 1e-8);
    float brightness = clamp((weight - $low) / $range, 0.0, 1.0);
    gl_FragColor = vec4(mean * brightness, $opacity);
}
"""

#: Summed weight above the lower limit over which the floor colour fades in.
#: In weight, not as a share of the window, because what it has to hide is
#: fixed in weight: a Gaussian's quad ends where it is still exp(-4), 1.8% of
#: its peak.  At 0.2 the floor colour there is down to 1.5% of itself.
FADE_WEIGHT = 0.2

#: The viewport, corner to corner, as a triangle strip.
_FULL_VIEWPORT = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)

#: True additive blending for the resolve: what every additive layer gets.
_RESOLVE_STATE = {
    "blend": True,
    "depth_test": False,
    "cull_face": False,
    "blend_func": ("src_alpha", "one"),
}


class SummedContrastPass:
    """Draws *node* through a float target and windows the sum.

    Disabled it is invisible: the node draws exactly as it did.  Enabled, the
    node's shader has to be writing weights rather than colours -- the layer
    switches both together.
    """

    def __init__(self, node):
        self._node = node
        self._original_draw = node.draw
        self._target = None
        self._framebuffer = None
        self._target_size = None
        self._target_channels = None

        self._program = ModularProgram(_RESOLVE_VERT, _RESOLVE_FRAG)
        self._program.frag["cmap"] = Function(
            "vec4 napari_storm_grey(float t) { return vec4(t, t, t, 1.0); }"
        )
        self._program.frag["fade"] = FADE_WEIGHT
        # The colour-coding resolve; the colormap was applied per splat.
        self._mean_program = ModularProgram(_RESOLVE_VERT, _RESOLVE_MEAN_COLOUR_FRAG)
        for program in (self._program, self._mean_program):
            program.vert["position"] = gloo.VertexBuffer(_FULL_VIEWPORT)
            program.frag["low"] = 0.0
            program.frag["range"] = 1.0
            program.frag["opacity"] = 1.0
            program.frag["target_size"] = (1.0, 1.0)

        self.enabled = False
        #: Resolve the weighted mean colour rather than colormapping the sum.
        self.mean_colour = False
        self.opacity = 1.0
        # Shadow the class's draw on this instance only.  VisPy calls
        # ``node.draw()`` for every node it draws, so this is the whole hook.
        node.draw = self.draw

    # ------------------------------------------------------------------
    # What the layer tells it
    # ------------------------------------------------------------------

    def set_colormap(self, colormap):
        """Colour the sum with *colormap*'s own GLSL and lookup table."""
        glsl = getattr(colormap, "glsl_map", None)
        if not glsl:
            return
        self._program.frag["cmap"] = Function(glsl)
        lut = getattr(colormap, "texture_lut", None)
        if lut is not None:
            self._program["texture2D_LUT"] = lut()

    def set_contrast_limits(self, low, high):
        """The window, in units of summed weight.  Floored like the splat one."""
        low, high = float(low), float(high)
        for program in (self._program, self._mean_program):
            program.frag["low"] = low
            program.frag["range"] = max(high - low, 1e-8)

    def release(self):
        """Give the node its own draw back."""
        if self._node.__dict__.get("draw") == self.draw:
            del self._node.draw
        self._target = self._framebuffer = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self):
        node = self._node
        canvas = node.canvas
        if not self.enabled or canvas is None or getattr(node, "picking", False):
            return self._original_draw()
        if self.opacity <= 0.0:
            return None  # a hidden channel adds nothing; skip both passes

        size = self._current_framebuffer_size(canvas)
        self._ensure_target(size, 4 if self.mean_colour else 1)
        program = self._mean_program if self.mean_colour else self._program

        # Pass 1: the weights, summed without clipping.  The node keeps its
        # own forced additive state; its shader writes alpha 1 in this mode,
        # so (src_alpha, one) adds the weight unscaled.
        self._framebuffer.activate()
        try:
            canvas.context.clear(color=(0.0, 0.0, 0.0, 0.0), depth=False)
            self._original_draw()
        finally:
            self._framebuffer.deactivate()

        # Pass 2: window, colormap, add to whatever was current before.
        program.frag["opacity"] = float(self.opacity)
        canvas.context.set_state(**_RESOLVE_STATE)
        program.draw("triangle_strip")
        return None

    @staticmethod
    def _current_framebuffer_size(canvas):
        """(width, height) of the framebuffer being drawn into, in pixels."""
        framebuffer, _origin, _canvas_size = canvas._current_framebuffer()
        if framebuffer is None:
            width, height = canvas.physical_size
            return int(width), int(height)
        height, width = framebuffer.color_buffer.shape[:2]
        return int(width), int(height)

    def _ensure_target(self, size, channels=1):
        """A float target of *size*: one channel for a sum, four for colours.

        Four only while colour coding, because the target is four times the
        memory and the fill bandwidth of the one-channel sum.
        """
        if (
            size == self._target_size
            and channels == self._target_channels
            and self._framebuffer is not None
        ):
            return
        width, height = size
        self._target = gloo.Texture2D(
            shape=(height, width, channels),
            internalformat="r32f" if channels == 1 else "rgba32f",
            interpolation="nearest",
        )
        self._framebuffer = gloo.FrameBuffer(color=self._target)
        for program in (self._program, self._mean_program):
            program.frag["summed"] = self._target
            program.frag["target_size"] = (float(width), float(height))
        self._target_size = size
        self._target_channels = channels
