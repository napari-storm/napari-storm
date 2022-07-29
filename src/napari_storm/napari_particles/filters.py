"""
"""

from abc import ABC

import numpy as np
from vispy.gloo import Texture2D, VertexBuffer
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function, Varying


# TODO: Add mipmapping
class TextureFilter(Filter):
    def __init__(self, texture, **kwargs):
        kwargs.setdefault("fhook", "post")

        self._fcode = Function(
            """
        void apply() {
            gl_FragColor *= texture2D($u_texture, v_texcoord);
        }
        """
        )
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

_shader_functions = {
    "gaussian": """
            varying mat2 covariance_inv;

            vec4 func(vec2 x){
                float val = exp(-2*dot(x,covariance_inv*x));
                //val = 0*val+x.x
                return val*vec4(1,1,1,1);
            }
            """,
    "gaussian2": """
            varying mat2 covariance_inv;
            vec4 func(vec2 x){
                float u = dot(x,covariance_inv*x);
                float y = -2*u;

                float val = (120+y*(120+y*(60+y*(20+y*(5+y)))))*0.0083333333f;
                val = clamp(val,0.f,1.f);
                return val*vec4(1,1,1,1);
            }
            """,
    "particle": """
            vec4 func(vec2 x){
                float r = length(x);
                float val = .05/((max(r,.01)-0.01)+0.05);


                return val*vec4(1,1,1,1);
            }
            """,
    "airy": """
            vec4 func(vec2 x){
                float r = 8*length(x);
                float val = abs(sin(r)/(1e-8+r));
                return val*vec4(1,1,1,1);
            }
            """,
    "fresnel": """
            vec4 func(vec2 x){
                float r = length(x);
                float d = .7;
                float val = 1.;
                if (r>d){
                    val = exp(-4*(r-d));
                    val *= cos(1000*(r-d)*(r-d));

                }
                return val*vec4(1,1,1,1);
            }
            """,
    "sphere": """
            vec4 func(vec2 x){

                float r = length(x);
                float r0 = .8;
                float val= 0;
                if (r<r0)
                    val= sqrt(0.001+r0*r0-r*r);
                else
                    discard;
                return vec4(val,val,val,1);
            }
            """,
    "none": """
            vec4 func(vec2 x){
                return vec4(1,1,1,1);
            }
            """,
    "bubble": """
            vec4 func(vec2 x){
                float r = length(x);
                float r1 = .8;
                float r2 = .9;
                float val = 0;
                if (r<r1)
                    val = (sqrt(r2*r2-r*r)-sqrt(r1*r1-r*r))/sqrt(r2*r2-r1*r1);
                if (r<r2)
                    val = sqrt(r2*r2-r*r)/sqrt(r2*r2-r1*r1);
                else
                    discard;
                return val*vec4(1,1,1,1);
            }
            """,
    "bubble2": """
            vec4 func(vec2 x){
                float r = length(x);
                float r0 = .9;
                float val = exp(-400*(r-r0)*(r-r0));
                if (r<r0){
                    val = max(val,r*r/r0/r0);
                }
                else{
                    discard;
                }
                return val*vec4(1,1,1,1);
            }
            """,
    "fractal": """
            vec4 func(vec2 x){
                vec2 c = vec2(-.4,.6);
                const float r = 2;
                const int n = 100;
                int res=0;
                for (int i=0;i<n;i++){
                    res += int(length(x)<r);
                    x = vec2(x.x*x.x-x.y*x.y,2*x.x*x.y);
                    x = x+c;
                }
                float val= float(res)/n;
                return val*vec4(1,1,1,1);
            }
            """,
}


class ShaderFilter(Filter):
    def __init__(self, mode="gaussian", distance_intensity_increase=1, **kwargs):
        kwargs.setdefault("fhook", "post")

        fcode = Function(
            """


        void apply() {
            // normalize texcoords to (-1,1)
            vec4 val = $func(2*(v_texcoord-.5));

            // if particle is far away, ramp up intensity
            float infinity_raise = $distance_intensity_increase*length(fwidth(v_texcoord));

            //gl_FragColor *= val*(1+infinity_raise);

            //val.w *= (exp(-.5*max(v_scale_intensity-1,0))+.05)/1.05;

            //gl_FragColor *= val/sqrt(max(1, v_scale_intensity));

            gl_FragColor *= val;




        }"""
        )

        if mode in _shader_functions:
            fcode["func"] = Function(_shader_functions[mode])
            fcode["distance_intensity_increase"] = 10 * distance_intensity_increase
        else:
            fcode = mode

        super().__init__(fcode=fcode, **kwargs)
