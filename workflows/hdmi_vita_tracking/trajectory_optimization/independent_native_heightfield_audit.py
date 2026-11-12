"""Independently verify compiled hfield array, frame and vertical rays."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco as mj
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();r=a.folder
    hp=r/'heightfield.npz';mp=r/'compiled_model.mjb';z=np.load(hp,allow_pickle=False)
    m=mj.MjModel.from_binary_path(str(mp));d=mj.MjData(m);mj.mj_forward(m,d)
    gid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'calibrated_scene_surface')
    if gid<0:raise ValueError('Compiled calibrated surface missing')
    hid=int(m.geom_dataid[gid]);shape=(int(m.hfield_nrow[hid]),int(m.hfield_ncol[hid]))
    adr=int(m.hfield_adr[hid]);data=np.asarray(m.hfield_data[adr:adr+np.prod(shape)]).reshape(shape)
    size=m.hfield_size[hid];position=d.geom_xpos[gid]
    reconstructed=data*size[2]+position[2]
    height=np.asarray(z['height_m']);array_error=float(np.max(np.abs(reconstructed-height))) if shape==height.shape else None
    rows,cols=np.where(z['observed_mask']);valid=(rows>0)&(rows<shape[0]-1)&(cols>0)&(cols<shape[1]-1)
    rows,cols=rows[valid],cols[valid];rng=np.random.default_rng(20261030)
    picked=rng.choice(len(rows),min(150,len(rows)),replace=False)
    samples=[]
    for n in picked:
        iy,ix=int(rows[n]),int(cols[n]);point=np.array([z['x'][ix],z['y'][iy],1.],dtype=np.float64)
        distance=mj.mj_rayHfield(m,d,gid,point,np.array([0.,0.,-1.]))
        measured=1.-distance if distance>=0 else None
        samples.append(dict(ix=ix,iy=iy,x=float(point[0]),y=float(point[1]),
                            expected_z_m=float(height[iy,ix]),ray_z_m=measured,
                            absolute_error_m=float(abs(measured-height[iy,ix])) if measured is not None else None))
    errors=np.array([q['absolute_error_m'] for q in samples if q['absolute_error_m'] is not None])
    center=np.array([(z['x'][0]+z['x'][-1])/2,(z['y'][0]+z['y'][-1])/2])
    half=np.array([(z['x'][-1]-z['x'][0])/2,(z['y'][-1]-z['y'][0])/2])
    checks=dict(shape=shape==height.shape,array_matches=array_error is not None and array_error<1e-6,
        center=np.max(np.abs(position[:2]-center))<1e-6,half_size=np.max(np.abs(size[:2]-half))<1e-6,
        no_row_column_flip=len(errors)==len(samples) and float(errors.max())<1e-5,
        geom_axes_identity=np.max(np.abs(d.geom_xmat[gid].reshape(3,3)-np.eye(3)))<1e-8)
    report=dict(schema='hdmi.independent_native_heightfield.v1',passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        model_sha256=sha(mp),heightfield_sha256=sha(hp),compiled_shape_yx=shape,
        compiled_center_xyz=position.tolist(),compiled_size=size.tolist(),array_reconstruction_max_error_m=array_error,
        ray_samples=len(samples),ray_absolute_error_max_m=float(errors.max()),ray_absolute_error_median_m=float(np.median(errors)),samples=samples)
    (r/'independent_native_heightfield_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
