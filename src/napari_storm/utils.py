from typing import Tuple, Union
import numpy as np

def generate_billboards_2d(coords: np.ndarray, size: Union[float, np.ndarray] =20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (vertices, faces, texture coordinates) of a <n> standard 2D billboards of given size(s) 
    """
    verts0 = np.array([[-0.5, -0.5],
                       [0.5,  -0.5],
                       [0.5,   0.5],
                       [-0.5,  0.5]]).astype(np.float32)

    n = len(coords) 

    if np.isscalar(size):
        size = np.ones(n)*size
    else:
        size = np.asarray(size)

    assert len(size)==n and size.ndim==1

    verts = size[:,np.newaxis, np.newaxis]*verts0[np.newaxis]
    verts = verts.reshape((-1,verts.shape[-1]))

    # add time/z dimensions if present
    
    if coords.shape[1]>2:
        coords = np.repeat(coords, 4, axis=0) 
        verts = np.concatenate([coords[:,:-2], verts], axis=-1)
        

    texcoords0 = np.array([[0, 0],
                           [1, 0],
                           [1, 1],
                           [0, 1]]).astype(np.float32)
    
    texcoords = np.tile(texcoords0,(n,1))
    
    faces = np.tile(np.array([[0,1,2],[0,3,2]]),(n,1))
    faces = faces+np.repeat(np.repeat(4*np.arange(n)[:,np.newaxis],3,axis=-1),2 ,axis=0)
    return verts, faces, texcoords


if __name__ == "__main__":
    #coords = np.random.uniform(0,1,(4,3))
    coords=np.asarray([[0,0,0],[1,0,0],[0,0,1],[1,0,1],[0,1,0],[1,1,0],[0,1,1],[1,1,1]])
    verts, faces, texc = generate_billboards_2d(coords, size=1)

    print(f"coords={coords},\nverts={verts},\nfaces={faces},\ntexc={texc}")
    