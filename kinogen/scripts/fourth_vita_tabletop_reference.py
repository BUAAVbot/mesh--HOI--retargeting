"""Build disclosed tabletop contact adaptation, retaining the original source."""
from pathlib import Path
import json
import argparse
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

R=Path('/data/KinoGen')
p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/fourth_vita_demo/retarget_reference_tabletop_v7.npz');p.add_argument('--offset',type=float,default=.10);p.add_argument('--height',type=float,default=.035);p.add_argument('--travel',type=float,default=.06);p.add_argument('--seconds',type=float,default=2.0);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(R/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic_ankle100.xml'));d=mj.MjData(m)
seed=np.load(R/'outputs/fourth_vita_demo/retarget_reference_palm_v6.npz',allow_pickle=False)
source=np.load(R/'data/fourth_sim_handoff/objects/poses_sim.npz',allow_pickle=False)
names=seed['dof_names'];q=seed['qpos'][0].copy()
selected=[j for j in range(m.njnt)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(('waist_','left_shoulder','left_elbow','left_wrist'))]
qa=np.array([m.jnt_qposadr[j]for j in selected]);lo=m.jnt_range[selected,0];hi=m.jnt_range[selected,1]
hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'lh_hand_base_link')
table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support')
walls=[g for g in range(m.ngeom)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('target_object_05_')and int(m.geom_contype[g])!=0]
arm=[g for g in range(m.ngeom)if int(m.geom_contype[g])!=0 and any(s in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'')for s in ['lh_','left_wrist','left_elbow'])]
base=source['position_m'][600,4].astype(float).copy();base[2]=m.geom_pos[table,2]+m.geom_size[table,2]+.0005
obj=np.r_[base,[1.,0.,0.,0.]]
toward=q[:3]-base;toward[2]=0;toward/=np.linalg.norm(toward)
normal=-toward;up=np.array([0.,0.,1.]);desired=np.column_stack((normal,np.cross(up,normal),up))
def distances():
    return np.array([mj.mj_geomDistance(m,d,g,w,.4,np.zeros(6))for g in arm for w in [table]+walls])
def fit(target,previous):
    def err(v):
        d.qpos[:59]=q;d.qpos[qa]=v;d.qpos[59:66]=obj;mj.mj_forward(m,d)
        rot=Rotation.from_matrix(desired@d.xmat[hand].reshape(3,3).T).as_rotvec()
        return np.r_[8*(d.xpos[hand]-target),.5*rot,.015*(v-previous),20*np.minimum(distances()+.001,0)]
    fit=least_squares(err,np.clip(previous,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=120,diff_step=1e-4,ftol=1e-7)
    d.qpos[:59]=q;d.qpos[qa]=fit.x;d.qpos[59:66]=obj;mj.mj_forward(m,d)
    result=d.qpos[:59].copy();return result,float(np.linalg.norm(d.xpos[hand]-target)),float(np.min(distances()))
initial=base+toward*a.offset+np.array([0.,0.,a.height])
q0,e,dist=fit(initial,q[qa]);print('INITIAL',e,dist,q0[qa].tolist(),flush=True)
qs=[];targets=[];err=[];clear=[];times=np.linspace(0,a.seconds,int(a.seconds*30)+1)
prev=q0[qa]
for t in times:
    # Slow precontact, 0.5s push, then hold away from table; all synthetic.
    progress=np.clip((t-.5)/.8,0,1);progress=progress*progress*(3-2*progress)
    target=initial-toward*(a.travel*progress)
    fitted,e,dist=fit(target,prev);prev=fitted[qa];qs.append(fitted);targets.append(target);err.append(e);clear.append(dist)
qs=np.asarray(qs);out=R/a.output;out.parent.mkdir(parents=True,exist_ok=True)
payload={k:seed[k]for k in seed.files if len(seed[k].shape)==0}
payload.update(qpos=qs,time_s=times,frame_ids=np.full(len(qs),600,dtype=np.int32),dof_names=names,
               object_target_pose=np.tile(obj,(len(qs),1)),source_object_target_pose_raw=source['position_m'][590:621,4],
               source_object_orientation_raw=source['quaternion_wxyz'][590:621,4],source_frame_ids=np.arange(590,621),
               task_adaptation=np.asarray('Synthetic tabletop push adaptation from source frame600; upright bowl settled on table; source poses retained; not exact source tracking'),
               synthetic_tabletop_adaptation=np.asarray(True),synthetic_static_diagnostic=np.asarray(False),
               adapted_hand_target_world=np.asarray(targets),adapted_hand_ik_error_m=np.asarray(err),adapted_collision_min_distance_m=np.asarray(clear))
np.savez_compressed(out,**payload)
report={'output':str(out),'initial_target':initial.tolist(),'object_pose':obj.tolist(),'min_clearance_m':float(min(clear)),'max_ik_error_m':float(max(err)),'times':len(times),'adaptation':str(payload['task_adaptation'])}
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
