"""Independent full control replay with unchanged raw and separate20ms support proofs."""
from pathlib import Path
import argparse,hashlib,json
from collections import deque
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert(a.trial/'report.json').is_file();m=mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'));b=mj.MjModel.from_xml_path(str(a.base.resolve()));z=dict(np.load(a.trial/'physical_trace.npz'));dt=float(m.opt.timestep);times=z['time_s'];assert len(times)==len(z['ctrl'])+1;assert np.all(np.diff(times)>0)
def shapes(model,prefixes):
 answer={}
 for g in range(model.ngeom):
  name=model.body(int(model.geom_bodyid[g])).name or''
  if not name.startswith(prefixes):continue
  value={k:np.asarray(getattr(model,k)[g]).tolist()for k in['geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_margin','geom_gap']}
  if model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
   mi=int(model.geom_dataid[g]);vs=int(model.mesh_vertadr[mi]);fs=int(model.mesh_faceadr[mi]);value['mesh_sha']=hashlib.sha256(model.mesh_vert[vs:vs+model.mesh_vertnum[mi]].tobytes()+model.mesh_face[fs:fs+model.mesh_facenum[mi]].tobytes()).hexdigest()
  answer[model.geom(g).name]=value
 return answer
hand_same=shapes(m,('rh_','lh_'))==shapes(b,('rh_','lh_'));objects_same=shapes(m,('target_object_',))==shapes(b,('target_object_',));d=mj.MjData(m);kin=mj.MjData(m);mj.mj_setState(m,d,z['initial_state'],int(z['state_spec']));mj.mj_forward(m,d);obj=m.body('target_object_01').id;oj=int(m.body_jntadr[obj]);va=int(m.jnt_dofadr[oj]);oq=int(m.jnt_qposadr[oj]);plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);plane/=np.linalg.norm(plane[:3]);normal=plane[:3];table={g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('table_support')};robot=m.body('base_link').id;pieces=[]
for g in range(m.ngeom):
 if m.geom_bodyid[g]!=obj or not m.geom_contype[g]:continue
 if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
  mi=int(m.geom_dataid[g]);v=int(m.mesh_vertadr[mi]);pieces.append((g,m.mesh_vert[v:v+int(m.mesh_vertnum[mi])].copy()))
 else:assert m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER;pieces.append((g,None))
def lowest():
 result=[]
 for g,v in pieces:
  G=kin.geom_xmat[g].reshape(3,3);c=kin.geom_xpos[g]
  if v is not None:result.append(float(np.min(v@G.T@normal+c@normal+plane[3])))
  else:
   cosine=abs(float(G[:,2]@normal));result.append(float(c@normal+plane[3]-m.geom_size[g,1]*cosine-m.geom_size[g,0]*np.sqrt(max(0,1-cosine*cosine))))
 return min(result)
err=np.zeros(3);mimicmax=0.;rows=[];handnames=[m.body(j).name or''for j in range(m.nbody)];mass=float(m.body_mass[obj]);weight=mass*9.81;relations=[(int(m.jnt_qposadr[int(m.eq_obj1id[e])]),int(m.jnt_qposadr[int(m.eq_obj2id[e])]),float(m.eq_data[e,1]))for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT];hand_aids=[i for i in range(m.nu)if(m.actuator(i).name or'').startswith(('motor_rh_','motor_lh_'))and np.any(m.actuator_gear[i])];passive=np.flatnonzero(np.all(m.actuator_gear==0,axis=1));maxhand=0.;maxpassive=0.
for step,ctrl in enumerate(z['ctrl']):
 d.ctrl[:]=ctrl;mj.mj_step(m,d);err=np.maximum(err,[np.max(abs(d.qpos-z['qpos'][step+1])),np.max(abs(d.qvel-z['qvel'][step+1])),np.max(abs(d.actuator_force-z['actuator_force'][step]))]);handup=tablen=othern=0.;count=0
 for ci,c in enumerate(d.contact[:d.ncon]):
  gs=list(map(int,c.geom));bs=list(map(int,m.geom_bodyid[gs]));
  if obj not in bs:continue
  oi=bs.index(obj);ii=1-oi;F=np.zeros(6);mj.mj_contactForce(m,d,ci,F);force=(1 if oi==1 else-1)*np.asarray(c.frame).reshape(3,3).T@F[:3]
  if handnames[bs[ii]].startswith(('rh_','lh_')):handup+=float(force[2]);count+=int(F[0]>.03)
  elif gs[ii]in table:tablen+=max(float(F[0]),0.)
  else:othern+=max(float(F[0]),0.)
 kin.qpos[:]=d.qpos;mj.mj_kinematics(m,kin);height=lowest();up=float(kin.xmat[robot].reshape(3,3)[2,2]);speed=float(np.linalg.norm(d.qvel[va:va+3]));ang=float(np.linalg.norm(d.qvel[va+3:va+6]));geometric=height>=.03 and up>=.8 and speed<.05 and ang<.3;raw=geometric and tablen<.15 and tablen+othern<.15 and count>0 and handup>=.7*weight;mimic=max(abs(d.qpos[c]-r*d.qpos[p])for c,p,r in relations);mimicmax=max(mimicmax,mimic);maxhand=max(maxhand,float(np.max(abs(d.actuator_force[hand_aids]))));maxpassive=max(maxpassive,float(np.max(abs(d.actuator_force[passive]))));rows.append([float(times[step+1]),height,handup,tablen,othern,count,up,speed,ang,int(geometric),int(raw),float(d.qpos[oq+2]),mimic])
 if step%5000==0:print(json.dumps({'t':float(times[step+1]),'replay_errors':err.tolist(),'height':height,'hand_up_N':handup,'table_N':tablen}),flush=True)
arr=np.asarray(rows);window=round(.020/dt);avg=np.full((len(arr),3),np.nan)
for i,col in enumerate([2,3,4]):avg[window-1:,i]=np.convolve(arr[:,col],np.ones(window)/window,mode='valid')
average_good=(arr[:,9]>0)&(avg[:,0]>=.7*weight)&(avg[:,1]<.15)&((avg[:,1]+avg[:,2])<.15)&(arr[:,5]>0)
def longest(flags):
 best=[0.,None,None];start=None
 for i,flag in enumerate(flags):
  if flag:
   if start is None:start=float(times[i])
   length=float(times[i+1]-start)
   if length>best[0]:best=[length,start,float(times[i+1])]
  else:start=None
 return{'duration_s':best[0],'start_s':best[1],'end_s':best[2]}
rawrun=longest(arr[:,10]>0);avgrun=longest(average_good);root=m.joint('vita_demo_free_root').id;free=all(m.jnt_type[int(m.body_jntadr[m.body('target_object_%02d'%i).id])]==mj.mjtJoint.mjJNT_FREE for i in range(1,7));external=not np.any(z['xfrc_applied'])and not np.any(z['qfrc_applied']);invariants=bool(hand_same and objects_same and free and m.jnt_type[root]==mj.mjtJoint.mjJNT_FREE and not np.any(m.eq_type==mj.mjtEq.mjEQ_WELD)and external and np.max(err)<1e-10 and mimicmax<=.03 and maxhand<=.4+1e-9 and maxpassive==0);result={'scope':'Independent full1ms control replay and unchanged original raw success rule.20ms trailing-average support is a separately named diagnostic, never replaces raw proof.','trial':str(a.trial),'model_sha256':hashlib.sha256((a.trial/'effective_model.mjb').read_bytes()).hexdigest(),'replay_max_qpos_qvel_actuatorforce_errors':err.tolist(),'original_hand_geometry_fingerprint_unchanged':hand_same,'all_six_original_object_shapes_unchanged':objects_same,'all_six_objects_free':bool(free),'root_free':bool(m.jnt_type[root]==mj.mjtJoint.mjJNT_FREE),'no_welds':bool(not np.any(m.eq_type==mj.mjtEq.mjEQ_WELD)),'zero_external_forces':bool(external),'hand_torque_max_Nm':maxhand,'passive_force_max_Nm':maxpassive,'mimic_max_rad':mimicmax,'invariants_pass':invariants,'raw_rule_longest_3cm_hold':rawrun,'raw_success':bool(invariants and rawrun['duration_s']>=2.-1e-9),'separate20ms_support_diagnostic_longest_3cm_hold':avgrun,'separate20ms_support_diagnostic_pass':bool(invariants and avgrun['duration_s']>=2.-1e-9),'raw_qualified_fraction_above3cm':float(np.mean(arr[arr[:,1]>=.03,10]))if np.any(arr[:,1]>=.03)else 0.,'maximum_lowest_collider_clearance_m':float(arr[:,1].max()),'final_row':arr[-1].tolist()};np.savez_compressed(a.output.with_suffix('.metrics.npz'),values=arr,mean20ms=avg,mean20ms_qualified=average_good,columns=np.asarray(['time_s','lowest_collider_table_clearance_m','hand_up_N','table_normal_N','nonhand_other_normal_N','active_hand_contacts','robot_upright','object_linear_speed_m_s','object_angular_speed_rad_s','height_and_speed_qualified','raw_proof_qualified','object_origin_world_z','mimic_rad']));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
