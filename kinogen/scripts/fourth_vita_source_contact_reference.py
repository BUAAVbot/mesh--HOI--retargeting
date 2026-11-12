"""Collision-aware contact refinement, preserving raw source-frame trajectories."""
from pathlib import Path
import argparse,json
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
R=Path('/data/KinoGen');p=argparse.ArgumentParser();p.add_argument('--offset',type=float,default=.002);p.add_argument('--output',default='outputs/fourth_vita_demo/retarget_source_contact_v8.npz');a=p.parse_args()
z=np.load(R/'outputs/fourth_vita_demo/retarget_diagnostic_status1_581.npz',allow_pickle=False)
m=mj.MjModel.from_xml_path(str(R/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic_ankle100.xml'));d=mj.MjData(m)
selected=[j for j in range(m.njnt)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(('left_shoulder','left_elbow','left_wrist','lh_thumb','lh_index','lh_middle'))]
qa=np.array([m.jnt_qposadr[j]for j in selected]);lo=m.jnt_range[selected,0];hi=m.jnt_range[selected,1]
table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support')
walls=[g for g in range(m.ngeom)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('target_object_05_')and int(m.geom_contype[g])!=0]
hand=[g for g in range(m.ngeom)if int(m.geom_contype[g])!=0 and (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith(('lh_','left_wrist'))]
fingers=[g for g in hand if any(x in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'')for x in ['thumb_distal','index_distal','middle_distal'])]
def dist(g,w):return mj.mj_geomDistance(m,d,g,w,.4,np.zeros(6))
def mind(g):return min(dist(g,w)for w in walls)
qs=[];rows=[];prev=None
for i in range(len(z['qpos'])):
    source=z['qpos'][i].copy();obj=z['object_target_pose'][i]
    initial=source[qa] if prev is None else prev
    def err(v):
        d.qpos[:59]=source;d.qpos[qa]=v;d.qpos[59:66]=obj;mj.mj_forward(m,d)
        ds=np.asarray([mind(g)for g in hand]);dt=np.asarray([dist(g,table)for g in hand]);df=np.asarray([mind(g)for g in fingers])
        return np.r_[.015*(v-source[qa]),.025*(v-initial),8*(df-a.offset),30*np.minimum(ds+.001,0),30*np.minimum(dt-.003,0)]
    fit=least_squares(err,np.clip(initial,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=100,diff_step=1e-4,ftol=1e-7)
    d.qpos[:59]=source;d.qpos[qa]=fit.x;d.qpos[59:66]=obj;mj.mj_forward(m,d)
    q=d.qpos[:59].copy();qs.append(q);prev=fit.x
    row={'frame':int(z['frame_ids'][i]),'finger_dist':[float(mind(g))for g in fingers], 'table_min':float(min(dist(g,table)for g in hand)),'bowl_min':float(min(mind(g)for g in hand)),'joint_delta_max':float(np.max(np.abs(fit.x-source[qa])))};rows.append(row)
    print(json.dumps(row),flush=True)
payload={k:z[k]for k in z.files};payload['qpos']=np.asarray(qs)
payload['raw_source_object_pose_xyz_wxyz']=z['object_target_pose'].copy();payload['raw_source_frame_ids']=z['frame_ids'].copy();payload['raw_source_time_s']=z['time_s'].copy();payload['raw_source_robot_qpos']=z['qpos'].copy()
payload['synthetic_task_adaptation']=np.asarray(False);payload['adaptation_description']=np.asarray('Source object trajectory and timing unchanged; collision-aware finger precontact IK assumption; right arm safe-static from prior v6')
sourcehuman=np.load(R/'data/fourth_sim_handoff/human/smplx_params_sim.npz',allow_pickle=False)
for k in sourcehuman.files:
    if sourcehuman[k].ndim and sourcehuman[k].shape[0] in (838,923):payload['raw_source_smplx_'+k]=sourcehuman[k][z['frame_ids']]
out=R/a.output;np.savez_compressed(out,**payload);out.with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n');print('OUTPUT',out,flush=True)
