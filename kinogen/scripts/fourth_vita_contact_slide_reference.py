"""Small contact-conditioned arm slide on source 590+ bowl trajectory."""
from pathlib import Path
import json,argparse
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
R=Path('/data/KinoGen');p=argparse.ArgumentParser();p.add_argument('--travel',type=float,default=.02);p.add_argument('--start-away',type=float,default=0.);p.add_argument('--lift',type=float,default=0.);p.add_argument('--source-direction',action='store_true');p.add_argument('--output',default='outputs/fourth_vita_demo/retarget_source_slide_v9.npz');a=p.parse_args()
z=np.load(R/'outputs/fourth_vita_demo/retarget_source_contact_v8.npz',allow_pickle=False)
o=np.load(R/'data/fourth_sim_handoff/objects/poses_sim.npz',allow_pickle=False);tl=np.load(R/'data/fourth_sim_handoff/timeline.npz',allow_pickle=False)
m=mj.MjModel.from_xml_path(str(R/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic_ankle100.xml'));d=mj.MjData(m)
q=z['qpos'][-1].copy();qsource=q.copy();obj=np.r_[o['position_m'][590,4],o['quaternion_wxyz'][590,4]]
selected=[j for j in range(m.njnt)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(('left_shoulder','left_elbow','left_wrist'))]
qa=np.array([m.jnt_qposadr[j]for j in selected]);lo=m.jnt_range[selected,0];hi=m.jnt_range[selected,1]
hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'lh_hand_base_link');table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support')
handgeoms=[g for g in range(m.ngeom)if int(m.geom_contype[g])and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith(('lh_','left_wrist'))]
d.qpos[:59]=q;d.qpos[59:]=obj;mj.mj_forward(m,d);initial=d.xpos[hand].copy();initialrot=d.xmat[hand].reshape(3,3).copy()
direction=obj[:3]-initial;direction[2]=0;direction/=np.linalg.norm(direction)
initial=initial-direction*a.start_away+np.asarray([0.,0.,a.lift])
if a.source_direction:
    direction=o['position_m'][620,4]-o['position_m'][590,4];direction[2]=0;direction/=np.linalg.norm(direction)
ids=np.arange(590,621);times=tl['pts_seconds'][ids]-tl['pts_seconds'][590];poses=np.c_[o['position_m'][ids,4],o['quaternion_wxyz'][ids,4]]
qs=[];errs=[];prev=q[qa].copy()
for t,pose in zip(times,poses):
    progress=np.clip((t-.1)/.8,0,1);progress=progress*progress*(3-2*progress);target=initial+direction*a.travel*progress
    def residual(v):
        d.qpos[:59]=q;d.qpos[qa]=v;d.qpos[59:]=pose;mj.mj_forward(m,d)
        return np.r_[10*(d.xpos[hand]-target),.015*(v-prev),20*np.minimum(np.array([mj.mj_geomDistance(m,d,g,table,.4,np.zeros(6))for g in handgeoms])-.002,0)]
    fit=least_squares(residual,np.clip(prev,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=60,diff_step=1e-4)
    d.qpos[:59]=q;d.qpos[qa]=fit.x;d.qpos[59:]=pose;mj.mj_forward(m,d);qs.append(d.qpos[:59].copy());prev=fit.x;errs.append(float(np.linalg.norm(d.xpos[hand]-target)))
payload={k:z[k]for k in z.files if z[k].ndim==0};payload.update(qpos=np.asarray(qs),time_s=times,dof_names=z['dof_names'],frame_ids=ids,object_id=np.asarray(5),object_target_pose=poses,raw_source_object_pose_xyz_wxyz=poses.copy(),raw_source_frame_ids=ids.copy(),raw_source_time_s=times.copy(),precontact_robot_seed_qpos=np.tile(qsource,(len(qs),1)),synthetic_task_adaptation=np.asarray(False),adaptation_description=np.asarray(f'Original source590-620 object pose and timing; synthetic precontact IK plus {a.travel:g}m hand slide, initial hand away {a.start_away:g}m/lift {a.lift:g}m; source object unaltered'),synthetic_hand_slide_m=np.asarray(a.travel),synthetic_hand_slide_direction_world=direction,synthetic_initial_hand_away_m=np.asarray(a.start_away),synthetic_initial_hand_lift_m=np.asarray(a.lift),hand_slide_target_error_m=np.asarray(errs))
h=np.load(R/'data/fourth_sim_handoff/human/smplx_params_sim.npz',allow_pickle=False)
for k in h.files:
    if h[k].ndim and h[k].shape[0] in (838,923):payload['raw_source_smplx_'+k]=h[k][ids]
    elif h[k].ndim==0:payload['raw_source_smplx_'+k]=h[k]
payload['source_bundle_sha256']=np.asarray('5d3b56568a041c2934af1ec62a83db58b59388dc80f7bb126bdcc4f0fa3cda6c')
payload['source_bundle_filename']=np.asarray('fourth_sim_handoff_bundle.tar.gz')
payload['synthetic_reference_hand_adjustment']=np.asarray(True)
out=R/a.output;np.savez_compressed(out,**payload);report={'output':str(out),'source_start':590,'source_stop':621,'object_trajectory_unchanged':True,'synthetic_hand_slide_m':a.travel,'direction':direction.tolist(),'ik_error_max':max(errs),'initial_hand':initial.tolist(),'object_initial':obj.tolist()};out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
