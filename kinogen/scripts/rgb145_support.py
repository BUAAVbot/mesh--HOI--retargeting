"""Pure geometry queries for the rgb145 terrain proxies (Z-up metres)."""
from pathlib import Path
import json
import numpy as np

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def inside_convex_polygon(xy,polygon,tolerance=1e-8):
    xy=np.asarray(xy,float);poly=np.asarray(polygon,float);edges=np.roll(poly,-1,axis=0)-poly;delta=xy[...,None,:]-poly
    cross=edges[:,0]*delta[...,1]-edges[:,1]*delta[...,0]
    return np.all(cross>=-tolerance,axis=-1)|np.all(cross<=tolerance,axis=-1)
def height_at_xy(spec,xy,return_names=False):
    xy=np.asarray(xy,float);shape=xy.shape[:-1];flat=xy.reshape(-1,2);g=np.asarray(spec['ground']['z_eq_ax_by_c']);heights=flat@g[:2]+g[2];names=np.full(len(flat),spec['ground']['name'],dtype=object)
    for item in spec['supports']:
        inside=inside_convex_polygon(flat,item['xy_polygon_m']);b=np.asarray(item['top_z_eq_ax_by_c']);z=flat@b[:2]+b[2];replace=inside&(z>heights)
        heights[replace]=z[replace];names[replace]=item['name']
    if return_names:return heights.reshape(shape),names.reshape(shape)
    return heights.reshape(shape)
