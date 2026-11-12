"""Attach a calibrated mesh-derived heightfield to the static terrain body.

The mjlab scene callback runs after the existing terrain/sensors are created.
Reusing its static body avoids fixed-entity mocap wrapping and preserves a
literal terrain-only contact sensor target. No shared task file is modified.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def configure_ground_scene(cfg, scene_file):
    scene_file=Path(scene_file)
    scene=json.loads(scene_file.read_text())
    descriptor=scene.get('ground_heightfield')
    if not descriptor:
        return {'calibrated_heightfield':False,'support_bodies':['terrain']+
                [f'seat_{i}/seat_{i}' for i in range(len(scene.get('boxes',[])))]}
    path=Path(descriptor['path'])
    if not path.is_absolute():path=scene_file.parent/path
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=descriptor['sha256']:raise ValueError('Heightfield SHA differs from calibrated scene')
    if descriptor.get('coordinate_frame')!='hdmi_world_z_up_m':raise ValueError('Unsupported heightfield frame')
    if scene.get('boxes'):raise ValueError('Calibrated field must explicitly replace proxy boxes')
    with np.load(path,allow_pickle=False) as z:
        height=np.asarray(z['height_m'],float);x=np.asarray(z['x'],float);y=np.asarray(z['y'],float)
        unknown=np.asarray(z['unknown_mask'],bool)
    if height.shape!=(len(y),len(x)) or not np.isfinite(height).all():raise ValueError('Invalid height grid')
    if len(x)<2 or len(y)<2 or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):raise ValueError('Invalid field axes')
    if not np.allclose(np.diff(x),np.diff(x)[0],atol=1e-7) or not np.allclose(np.diff(y),np.diff(y)[0],atol=1e-7):
        raise ValueError('MuJoCo heightfield requires a regular XY grid')
    low=float(height.min());span=max(float(height.max()-low),1e-5)
    xhalf=float((x[-1]-x[0])/2);yhalf=float((y[-1]-y[0])/2)
    data=((height-low)/span).ravel().tolist()
    original_callback=cfg.scene.spec_fn

    def install(spec):
        if original_callback:original_callback(spec)
        terrain=spec.body('terrain')
        plane=spec.geom('terrain')
        friction=plane.friction.copy()
        plane.pos=(0.,0.,low-.15)
        field=spec.add_hfield(name='calibrated_scene_heightfield',nrow=len(y),ncol=len(x),
                              size=(xhalf,yhalf,span,.1),userdata=data)
        terrain.add_geom(name='calibrated_scene_surface',type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,pos=(float((x[0]+x[-1])/2),float((y[0]+y[-1])/2),low),
            friction=friction,rgba=(.65,.68,.70,1))
    cfg.scene.spec_fn=install
    return dict(calibrated_heightfield=True,path=str(path),sha256=digest,
                coordinate_frame='hdmi_world_z_up_m',support_bodies=['terrain'],
                grid_shape=list(height.shape),unknown_grid_fraction=float(unknown.mean()),
                geom_name='calibrated_scene_surface',body_name='terrain',
                center_xyz=[float((x[0]+x[-1])/2),float((y[0]+y[-1])/2),low],
                extent_halfsize=[xhalf,yhalf,span,.1],
                low_m=low,high_m=float(height.max()),fallback_plane_height_m=low-.15)
