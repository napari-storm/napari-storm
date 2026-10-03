""" """

import numpy as np
from vispy.gloo import Texture2D
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function

from .footprint_shaders import LEGACY_NAMES, SHAPES, is_opaque, shape_function


# TODO: Add mipmapping
class TextureFilter(Filter):
    def __init__(self, texture, **kwargs):
        kwargs.setdefault("fhook", "post")

        self._fcode = Function("""
        void apply() {
            gl_FragColor *= texture2D($u_texture, v_texcoord);
        }
        """)
        texture = np.asarray(texture).astype(np.float32)
        if not texture.ndim == 3:
            raise ValueError("texure needs to be array of size (M,N,1)")
        self.texture = texture
        super().__init__(fcode=self._fcode, **kwargs)

    @property
    def texture(self):
        """The texture image."""
        return self._texture

    @texture.setter
    def texture(self, texture):
        self._texture = texture
        self._fcode["u_texture"] = Texture2D(texture)


# ShaderFilters
#
# What a billboard shows is a footprint from `core.footprints.PALETTE`.  The
# GLSL for each lives in `footprint_shaders`, shared with the instanced
# backend; napari-particles' own sprite table used to sit here.


class ShaderFilter(Filter):
    """Shades every billboard with one footprint from the palette.

    *mode* is a palette name, one of napari-particles' own names for its sprites
    (see `footprint_shaders.LEGACY_NAMES`), or raw GLSL for a filter of one's
    own.  *distance_intensity_increase* is accepted and ignored: the template it
    fed computed a value nothing used.
    """

    def __init__(self, mode="gaussian", distance_intensity_increase=1, **kwargs):
        kwargs.setdefault("fhook", "post")
        name = LEGACY_NAMES.get(mode, mode)
        if name in SHAPES:
            fcode = Function("""
            varying mat2 v_disc_inv;

            void apply() {
                // normalize texcoords to (-1,1)
                vec2 x = 2.0*(v_texcoord - 0.5);
                // See InstancedBillboardsFilter: an opaque footprint is drawn
                // at alpha 1, and not at all while the layer's opacity is 0.
                if ($opaque > 0.5 && gl_FragColor.a <= 0.0) {
                    discard;
                }
                float q = dot(x, v_disc_inv*x);
                vec4 drawn = $shape(x, q, 0.25 * q, gl_FragColor);
                if ($opaque > 0.5) {
                    drawn.a = 1.0;
                }
                gl_FragColor = drawn;
            }""")
            fcode["shape"] = shape_function(name)
            fcode["opaque"] = 1.0 if is_opaque(name) else 0.0
        else:
            fcode = mode

        super().__init__(fcode=fcode, **kwargs)
