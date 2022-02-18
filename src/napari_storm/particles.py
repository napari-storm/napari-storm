"""
A billboarded particle layer with texture/shader support

"""

import numpy as np
from collections.abc import Iterable
from napari.layers import Surface
from vispy.visuals.filters import Filter
from vispy.visuals.shaders import Function, Varying
from vispy.gloo import VertexBuffer

from .utils import generate_billboards_2d
from .filters import ShaderFilter, _shader_functions


class BillboardsFilter(Filter):
    """ Billboard geometry filter (transforms vertices to always face camera) 
    """
    def __init__(self, antialias=0):
        vmat_inv = Function(""" 
            mat2 inverse(mat2 m) {
                return mat2(m[1][1],-m[0][1],-m[1][0], m[0][0]) / (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
            }
        """)

        vfunc = Function("""
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
        """)

        ffunc = Function("""
        varying float v_scale_intensity;
        varying float v_z_center;
        
        void apply() {   
            gl_FragDepth = v_z_center;
            $texcoords;
        }
        """)

        self._texcoord_varying = Varying('v_texcoord', 'vec2')
        vfunc['inverse'] = vmat_inv
        vfunc['v_texcoords'] = self._texcoord_varying
        ffunc['texcoords'] = self._texcoord_varying
        
        self._texcoords_buffer = VertexBuffer(
            np.zeros((0, 2), dtype=np.float32)
        )
        vfunc['texcoords'] = self._texcoords_buffer
        vfunc['antialias'] = float(antialias)

        
        self._centercoords_buffer = VertexBuffer(
            np.zeros((0, 3), dtype=np.float32))
        self._sigmas_buffer = VertexBuffer(
            np.zeros((0, 3), dtype=np.float32))

        vfunc['vertex_center'] = self._centercoords_buffer
        vfunc['sigmas']        = self._sigmas_buffer

        super().__init__(vcode=vfunc, vhook='post',fcode=ffunc, fhook='post')

        

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
            self._centercoords_buffer.set_data(centercoords[:,::-1], convert=True)

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
            self._sigmas_buffer.set_data(sigmas[:,::-1], convert=True)

    @property
    def texcoords(self):
        """The texture coordinates as an (N, 2) array of floats."""
        return self._texcoords

    @texcoords.setter
    def texcoords(self, texcoords):
        self._texcoords = texcoords
        self._update_texcoords_buffer(texcoords)

    def _update_texcoords_buffer(self, texcoords):
        if self._attached or self._visual is not None:
            self._texcoords_buffer.set_data(texcoords[:,::-1], convert=True)

    def _attach(self, visual):

        # the full projection model view
        self.vshader['transform'] = visual.transforms.get_transform('visual', 'render')
        # the inverse of it
        self.vshader['transform_inv'] = visual.transforms.get_transform('render', 'visual')

        # the modelview
        self.vshader['camera_inv'] = visual.transforms.get_transform('document', 'scene')
        # inverse of it
        self.vshader['camera'] = visual.transforms.get_transform('scene', 'document')
        super()._attach(visual)
        

class Particles(Surface):
    """ Billboarded particle layer that renders camera facing quads of given size
        Can be combined with other (e.g. texture) filter to create particle systems etc 
    """
    def __init__(self, coords, size=10, sigmas=(1,1,1), values=1, filter=ShaderFilter('gaussian'), antialias=False, **kwargs):

        kwargs.setdefault('shading', 'none')
        kwargs.setdefault('blending', 'additive')
        
        coords = np.asarray(coords)   
        sigmas = np.asarray(sigmas, dtype=np.float32) 

        if np.isscalar(values):
            values = values * np.ones(len(coords))

        values = np.broadcast_to(values, len(coords))
        size   = np.broadcast_to(size, len(coords))
        sigmas = np.broadcast_to(sigmas, (len(coords),3))
        
        if not coords.ndim == 2 :
            raise ValueError(f'coords should be of shape (M,D)')

        if not len(size)==len(coords)==len(sigmas):
            raise ValueError()
    
        # add dummy z if 2d coords 
        if coords.shape[1] == 2:
            coords = np.concatenate([np.zeros((len(coords),1)), coords], axis=-1)

        assert coords.shape[-1]==sigmas.shape[-1]==3

        vertices, faces, texcoords = generate_billboards_2d(coords, size=size)    


        # repeat values for each 4 vertices
        centercoords = np.repeat(coords, 4, axis=0)
        sigmas       = np.repeat(sigmas, 4, axis=0)
        values       = np.repeat(values, 4, axis=0)
        self._coords = coords
        self._centercoords = centercoords
        self._sigmas  = sigmas
        self._size = size
        self._texcoords = texcoords 
        self._billboard_filter = BillboardsFilter(antialias=antialias)
        self.filter = filter
        self._viewer = None 
        super().__init__((vertices, faces, values), **kwargs)

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""
        super()._set_view_slice()
        self._update_billboard_filter()

    def _update_billboard_filter(self):
        faces = self._view_faces.flatten()
        if self._billboard_filter._attached and len(faces)>0:
                self._billboard_filter.texcoords    = self._texcoords[faces]
                self._billboard_filter.centercoords = self._centercoords[faces][:,-3:]
                self._billboard_filter.sigmas       = self._sigmas[faces][:,-3:]

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
            size = np.repeat(self._size[:,np.newaxis], self.ndim, axis=-1)
            size[:,:-2] *=0
            maxs = np.max(self._coords+.5*size, axis=0)
            mins = np.min(self._coords-.5*size, axis=0)
            extrema = np.vstack([mins, maxs])
        return extrema

    @property #LR
    def coords(self):
        return self._coords

    @coords.setter #LR
    def coords(self,coords):
        print("we're here")
        self._coords=coords


    @property
    def shading(self):
        return str(self._shading)

    @shading.setter
    def shading(self, shading):
        self._shading = shading
        self._detach_filter()
        self.filter=ShaderFilter(shading)
        self._attach_filter()

    def _detach_filter(self):
        for f in self.filter:
            self._visual.detach(f)

    def _attach_filter(self):
        for f in self.filter:
            self._visual.attach(f)

    def get_visual(self, viewer):
        import warnings
        warnings.filterwarnings("ignore")
        #depreciated soon
        #return viewer.layer_to_visual[self].node
        return viewer.window.qt_viewer.layer_to_visual[self].node

    def add_to_viewer(self, viewer):
        self._viewer = viewer 
        self._viewer.add_layer(self)
        self._visual = self.get_visual(viewer)
        
        
        self._visual.attach(self._billboard_filter)
        self._update_billboard_filter()
        self._attach_filter()

        #""" Martins Hack into the layer Controlls -> replaced later
        #update combobox
        import warnings
        warnings.filterwarnings("ignore")
        shading_ctrl = self._viewer.window.qt_viewer.controls.widgets[self]
        combo = shading_ctrl.shadingComboBox

        combo.clear()
        for k in _shader_functions.keys():
            combo.addItem(k, k)
        #"""



        





