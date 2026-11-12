"""Native geometry audit of source reference feet against actual scene surfaces."""
import json
from pathlib import Path
import mujoco
import numpy as np

base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
source=np.load(motion,allow_pickle=False)
m=mujoco.MjModel.from_binary_path(str(out/'baseline_instrumented/compiled_model.mjb'))
d=mujoco.MjData(m)
geom_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
scene_ids=[i for i,name in enumerate(geom_names) if name=='terrain' or 'seat_' in name]
scene_names=[geom_names[i] for i in scene_ids]
feet=[[i for i,name in enumerate(geom_names) if name.startswith('robot/') and
       ('foot_' in name or 'toe_' in name) and ('_l_' in name if side=='left' else '_r_' in name)]
      for side in ('left','right')]
if any(len(ids)!=5 for ids in feet):raise ValueError(f'Unexpected foot geoms {feet}')
distance=np.zeros((len(source['joint_pos']),2,len(scene_ids)))
foot_pos=np.zeros((len(distance),2,3))
source_depth=[]
contact_rows=[]
for t in range(len(distance)):
    d.qpos[:3]=source['body_pos_w'][t,0]
    d.qpos[3:7]=source['body_quat_w'][t,0]
    d.qpos[7:]=source['joint_pos'][t]
    mujoco.mj_forward(m,d)
    for side,ids in enumerate(feet):
        foot_pos[t,side]=d.geom_xpos[ids].mean(0)
        for k,scene_id in enumerate(scene_ids):
            pairs=[mujoco.mj_geomDistance(m,d,i,scene_id,10.,None) for i in ids]
            distance[t,side,k]=min(pairs)
    source_depth.append(max([max(-c.dist,0) for c in d.contact[:d.ncon]] or [0]))
    for c in d.contact[:d.ncon]:
        if c.dist < -.001:
            g1,g2=geom_names[int(c.geom1)],geom_names[int(c.geom2)]
            contact_rows.append(dict(frame=t,depth_m=float(-c.dist),geom1=g1,geom2=g2,
                                     kind='self' if g1.startswith('robot/') and g2.startswith('robot/') else 'scene'))
speed=np.linalg.norm(np.gradient(foot_pos,.02,axis=0),axis=-1)
nearest=distance.argmin(-1)
gap=distance.min(-1)
audit=dict(source=str(motion),frames=len(distance),fps=50.,scene_geometries=scene_names,
    foot_geometries=[[geom_names[i] for i in ids] for ids in feet],
    frame0=dict(gaps_by_surface_m=distance[0].tolist(),nearest_surface=[scene_names[i] for i in nearest[0]],
                nearest_gap_m=gap[0].tolist(),foot_speed_m_s=speed[0].tolist(),foot_center_m=foot_pos[0].tolist()),
    gap_ranges_m=[dict(min=float(gap[:,i].min()),max=float(gap[:,i].max()),median=float(np.median(gap[:,i]))) for i in range(2)],
    both_gap_over_2cm_frames=int(np.sum((gap>.02).all(-1))),
    min_gap_over_2cm_frames=int(np.sum(gap.min(-1)>.02)),
    max_source_contact_penetration_m=float(max(source_depth)),
    max_source_scene_penetration_m=max([r['depth_m'] for r in contact_rows if r['kind']=='scene'] or [0]),
    max_source_self_penetration_m=max([r['depth_m'] for r in contact_rows if r['kind']=='self'] or [0]),
    deepest_contact_pairs=sorted(contact_rows,key=lambda r:r['depth_m'],reverse=True)[:20],
    nearest_counts=[{scene_names[k]:int(np.sum(nearest[:,i]==k)) for k in range(len(scene_ids))} for i in range(2)])
np.savez_compressed(out/'source_support_geometry.npz',distance_m=distance,nearest_surface=nearest,gap_m=gap,
                    foot_center_m=foot_pos,foot_speed_m_s=speed,scene_names=np.asarray(scene_names),
                    source_contact_penetration_m=np.asarray(source_depth))
(out/'source_support_geometry_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
