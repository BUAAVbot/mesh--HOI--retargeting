"""Rasterize observed mesh triangles without broadening raised terrain into boxes."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resolution',type=float,default=.025)
    a=p.parse_args();base=a.base;out=a.output
    out.mkdir(parents=True,exist_ok=False)
    inp=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3'
    source=inp/'mjlab_motion.npz';meta=json.loads(source.with_suffix('.json').read_text())
    scene=json.loads((inp/'scene.json').read_text())
    refp=base/'exo_compare_20261030/videomimic_gallery_08_v2/person_01/omni/reference.npz'
    meshp=Path('/data/yihang.bai/incoming/aliyun_hoi_result/exo/videomimic_gallery_08_v2/scene/mesh.npz')
    modelp=base/'outputs/gallery08_p01_trajopt_20261030_2205/baseline_instrumented/compiled_model.mjb'
    z=np.load(source,allow_pickle=False);mesh=np.load(meshp,allow_pickle=False);r=np.load(refp,allow_pickle=False)
    rot=np.asarray(r['world_from_source_rotation'],float);floor=float(np.asarray(r['floor_offset']).reshape(-1)[0])
    verts=np.asarray(mesh['vertices'],float)@rot.T;verts[:,:2]-=scene['xy_offset'];verts[:,2]-=floor
    faces=np.asarray(mesh['faces'],int)
    m=mj.MjModel.from_binary_path(str(modelp));d=mj.MjData(m)
    jnames=meta['joint_names'];rootid=meta['body_names'].index('base_link')
    qadr=[m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'robot/'+n)] for n in jnames]
    gn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)]
    gids=[i for i,n in enumerate(gn) if 'collision_foot_' in n]
    signs=np.array([[xx,yy,zz] for xx in (-1,1) for yy in (-1,1) for zz in (-1,1)])
    feet=[]
    for k in range(len(z['joint_pos'])):
        mj.mj_resetData(m,d);d.qpos[:3]=z['body_pos_w'][k,rootid];d.qpos[3:7]=z['body_quat_w'][k,rootid];d.qpos[qadr]=z['joint_pos'][k]
        mj.mj_forward(m,d)
        for gid in gids:
            corners=(signs*m.geom_size[gid])@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
            feet.extend(corners[np.argsort(corners[:,2])[:4],:2])
    feet=np.asarray(feet)
    step=a.resolution;lower=np.floor((feet.min(0)-.5)/step)*step;upper=np.ceil((feet.max(0)+.5)/step)*step
    nx,ny=(np.rint((upper-lower)/step).astype(int)+1).tolist()
    x=lower[0]+np.arange(nx)*step;y=lower[1]+np.arange(ny)*step
    heights=np.full((ny,nx),-np.inf,dtype=np.float64)
    tri=verts[faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(cross,axis=1)
    # Ground support observations only: exclude walls and overhead scene fragments.
    normals_ok=np.abs(cross[:,2])> .7*np.maximum(norm,1e-20)
    bounding_ok=np.all(tri[:,:,:2].max(1)>=lower,axis=1)&np.all(tri[:,:,:2].min(1)<=upper,axis=1)
    z_ok=(tri[:,:,2].min(1)>=-.06)&(tri[:,:,2].max(1)<=.5)
    chosen=np.flatnonzero(normals_ok&bounding_ok&z_ok)
    for idx in chosen:
        t=tri[idx];start=np.maximum(np.ceil((t[:,:2].min(0)-lower)/step).astype(int),0)
        end=np.minimum(np.floor((t[:,:2].max(0)-lower)/step).astype(int),[nx-1,ny-1])
        if np.any(end<start):continue
        ix=np.arange(start[0],end[0]+1);iy=np.arange(start[1],end[1]+1)
        xx,yy=np.meshgrid(x[ix],y[iy]);q=np.stack((xx-t[0,0],yy-t[0,1]),axis=-1)
        one=t[1,:2]-t[0,:2];two=t[2,:2]-t[0,:2];det=one[0]*two[1]-one[1]*two[0]
        if abs(det)<1e-15:continue
        u=(q[:,:,0]*two[1]-q[:,:,1]*two[0])/det
        w=(one[0]*q[:,:,1]-one[1]*q[:,:,0])/det
        inside=(u>=-1e-7)&(w>=-1e-7)&(u+w<=1+1e-7)
        zh=t[0,2]+u*(t[1,2]-t[0,2])+w*(t[2,2]-t[0,2])
        sub=heights[start[1]:end[1]+1,start[0]:end[0]+1]
        np.maximum(sub,np.where(inside,zh,-np.inf),out=sub)
    observed=np.isfinite(heights)
    if not observed.any():raise ValueError('No actual triangle samples inside terrain region')
    distances,nearest=distance_transform_edt(~observed,sampling=(step,step),return_indices=True)
    labels,nlabels=label(~observed);filled=np.zeros_like(observed)
    for lid in range(1,nlabels+1):
        rows,cols=np.where(labels==lid)
        boundary=np.any((rows==0)|(cols==0)|(rows==ny-1)|(cols==nx-1))
        if not boundary and len(rows)<=25 and distances[rows,cols].max()<=.0500001:
            filled[rows,cols]=True
    heights[filled]=heights[nearest[0][filled],nearest[1][filled]]
    unknown=~(observed|filled)
    # A dense MuJoCo heightfield needs finite samples. Unknown space retains
    # the old z=0 plane, is explicitly masked and is never called reconstructed.
    heights[unknown]=0.
    fi=np.clip(np.rint((feet[:,0]-lower[0])/step).astype(int),0,nx-1)
    fj=np.clip(np.rint((feet[:,1]-lower[1])/step).astype(int),0,ny-1)
    foot_unknown=unknown[fj,fi]
    unknown_by_frame=foot_unknown.reshape(len(z['joint_pos']),-1).sum(1)
    transform=dict(rotation=rot.tolist(),floor_offset=floor,xy_offset=scene['xy_offset'])
    transform_sha=hashlib.sha256(json.dumps(transform,sort_keys=True).encode()).hexdigest()
    npz=out/'heightfield.npz'
    np.savez_compressed(npz,height_m=heights.astype(np.float32),x=x,y=y,spacing_xy_m=np.array([step,step]),
        observed_mask=observed,small_hole_filled_mask=filled,unknown_mask=unknown,
        foot_samples_xy=feet,foot_samples_unknown=foot_unknown,
        foot_unknown_samples_per_frame=unknown_by_frame,
        source_mesh_sha256=np.array(sha(meshp)),source_reference_sha256=np.array(sha(refp)),
        source_motion_sha256=np.array(sha(source)),transform_sha256=np.array(transform_sha))
    new_scene=dict(scene);new_scene['boxes']=[]
    new_scene['ground_heightfield']=dict(path=str(npz.resolve()),sha256=sha(npz),array_key='height_m',
        coordinate_frame='hdmi_world_z_up_m',nrow=ny,ncol=nx,spacing_xy_m=[step,step],
        bounds_xy=[lower.tolist(),upper.tolist()],center_xy=((lower+upper)/2).tolist(),
        half_size_xy=((upper-lower)/2).tolist(),height_min_m=float(heights.min()),height_max_m=float(heights.max()),
        source_mesh=str(meshp),source_mesh_sha256=sha(meshp),reference_transform_sha256=transform_sha,
        original_reference_sha256=sha(refp),original_motion_sha256=sha(source),
        unknown_policy='original_z0_plane_fallback; unknown_mask retained; not observed reconstruction')
    new_scene['collision_proxy_method']='observed_mesh_triangle_heightfield_25mm_no_broad_tile_boxes'
    new_scene['original_collision_scene_sha256']=sha(inp/'scene.json')
    new_scene['physics_validated']=False
    (out/'scene.json').write_text(json.dumps(new_scene,indent=2),encoding='utf-8')
    report=dict(schema='hdmi.mesh_calibrated_heightfield.v1',physics_validated=False,
        source_mesh_sha256=sha(meshp),source_reference_sha256=sha(refp),source_motion_sha256=sha(source),
        model_sha256=sha(modelp),heightfield_sha256=sha(npz),scene_sha256=sha(out/'scene.json'),transform=transform,
        transform_sha256=transform_sha,resolution_m=step,shape_yx=[ny,nx],triangles_used=len(chosen),
        vertex_height_filter_m=[-.06,.5],normal_z_abs_min=.7,bounds_xy=[lower.tolist(),upper.tolist()],
        observed_grid_fraction=float(observed.mean()),small_hole_filled_fraction=float(filled.mean()),
        unknown_grid_fraction=float(unknown.mean()),foot_samples=len(feet),foot_unknown_samples=int(foot_unknown.sum()),
        foot_observed_or_small_hole_fraction=float(np.mean(~foot_unknown)),
        foot_frames_with_any_unknown=int(np.sum(unknown_by_frame>0)),
        first_foot_unknown_frame=int(np.flatnonzero(unknown_by_frame>0)[0]) if np.any(unknown_by_frame) else None,
        foot_unknown_frame_indices=np.flatnonzero(unknown_by_frame>0).tolist(),
        height_range_m=[float(heights.min()),float(heights.max())],
        limits=['Heightfield approximates vertical steps over one grid interval.',
                'Large mesh holes are explicit unknown areas with original plane fallback, not reconstructed support.',
                'Foot route and later physical rollout must avoid unknown support regions.'],
        prior_proxy_audit=str(base/'outputs/gallery08_p01_trajopt_20261030_2205/independent_mesh_ground_audit.json'))
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
