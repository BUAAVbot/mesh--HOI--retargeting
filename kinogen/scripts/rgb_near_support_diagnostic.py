"""Read-only support geometry, solver forces and COM projection for each actor."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull

p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
m=mj.MjModel.from_binary_path(str(a.trial/'model.mjb'));z=np.load(a.trial/'trace.npz',allow_pickle=False);c=np.load(a.trial/'contacts.npz',allow_pickle=False)
d=mj.MjData(m);r=mj.MjData(m);bn=list(map(str,c['body_names']));gn=list(map(str,c['geom_names']))
def bottom(data,g):
    R=data.geom_xmat[g].reshape(3,3);typ=int(m.geom_type[g]);pos=data.geom_xpos[g]
    if typ==int(mj.mjtGeom.mjGEOM_CAPSULE):return float(pos[2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(R[2,2]))
    if typ==int(mj.mjtGeom.mjGEOM_BOX):return float(pos[2]-np.abs(R[2])@m.geom_size[g])
    if typ==int(mj.mjtGeom.mjGEOM_MESH):
        mid=int(m.geom_dataid[g]);v=m.mesh_vert[int(m.mesh_vertadr[mid]):int(m.mesh_vertadr[mid])+int(m.mesh_vertnum[mid])]
        return float(np.min(v@R[2])+pos[2])
    return float(pos[2]-(np.abs(R[2])@m.geom_aabb[g,3:]))
out=[]
for requested in [0.,.1,.5,1.,min(2.,float(z['time'][-1]))]:
    step=int(np.clip(np.searchsorted(z['time'],requested,side='right')-1,0,len(z['ctrl'])-1));t=float(z['time'][step])
    d.qpos[:]=z['qpos'][step];d.qvel[:]=z['qvel'][step];d.ctrl[:]=z['ctrl'][step]
    if m.neq:m.eq_data[:]=z['eq_data'][step];d.eq_active[:]=z['eq_active'][step]
    mj.mj_forward(m,d);r.qpos[:]=d.qpos
    for actor in range(2):
        q=np.asarray([np.interp(t,z['reference_time_s'],z['reference_qpos'][:,actor,j])for j in range(59)])
        rq=int(z['robot_root_qadr'][actor]);r.qpos[rq:rq+7]=q[:7];r.qpos[z['joint_qadr'][actor]]=q[7:]
    r.qpos[int(z['object_qadr']):int(z['object_qadr'])+7]=np.asarray([np.interp(t,z['reference_time_s'],z['reference_object_target_pose'][:,j])for j in range(7)])
    mj.mj_forward(m,r);actors=[];ids=np.flatnonzero(c['interval_index']==step)
    for actor,prefix in enumerate(['p1_','p2_']):
        bids=[b for b,n in enumerate(bn)if n.startswith(prefix)];mass=float(np.sum(m.body_mass[bids]));com=np.sum(d.xipos[bids]*m.body_mass[bids,None],axis=0)/mass
        soles={};goals={}
        for side in ['left','right']:
            gs=[g for g in range(m.ngeom)if m.geom_contype[g]and bn[int(m.geom_bodyid[g])].startswith(prefix+side+'_ankle')]
            soles[side]=min(bottom(d,g)for g in gs);goals[side]=min(bottom(r,g)for g in gs)
        ground={'left':0.,'right':0.,'shoe':0.};seat=0.;points=[]
        for row in ids:
            gids=c['geom_ids'][row];bodies=c['body_ids'][row];geoms=[gn[int(g)]for g in gids];bodiesnames=[bn[int(b)]for b in bodies]
            for k,name in enumerate(bodiesnames):
                relevant=name.startswith(prefix)or(actor==0 and name=='shoe')
                if not relevant:continue
                f=c['force_world_on_geom2_N'][row]*(1 if k==1 else-1)
                if f[2]<=1.:continue
                isground=any('ground'in g or'floor'in g for g in geoms)
                isseat=any('bench_seat'in g for g in geoms)
                if isground:
                    if name=='shoe':ground['shoe']+=float(f[2])
                    elif'left_ankle'in name:ground['left']+=float(f[2])
                    elif'right_ankle'in name:ground['right']+=float(f[2])
                    else:continue
                    points.append(c['position_world_m'][row,:2].tolist())
                if isseat:seat+=float(f[2]);points.append(c['position_world_m'][row,:2].tolist())
        inside=None
        if len(points)>=3:
            try:
                hull=ConvexHull(np.asarray(points));inside=bool(np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2])<=.002)
            except Exception:pass
        shoegeoms=[g for g in range(m.ngeom)if m.geom_contype[g]and bn[int(m.geom_bodyid[g])]=='shoe']
        actors.append({'actor':actor,'mass_kg':mass,'com_world_m':com.tolist(),'reference_foot_bottom_m':goals,'actual_foot_bottom_m':soles,
                       'shoe_bottom_actual_m':min(bottom(d,g)for g in shoegeoms)if actor==0 else None,
                       'shoe_bottom_reference_m':min(bottom(r,g)for g in shoegeoms)if actor==0 else None,
                       'actual_ground_vertical_force_N':ground,'actual_seat_vertical_force_N':seat,'support_contact_points_world_xy':points,'COM_inside_instantaneous_support_polygon':inside})
    out.append({'time_s':t,'actors':actors})
report={'schema':'rgb_near.support_diagnostic.v1','trial':str(a.trial),'samples':out,'note':'Forces come from original recorded solver intervals; source-vs-actual collision sole geometry recomputed read-only.'}
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
