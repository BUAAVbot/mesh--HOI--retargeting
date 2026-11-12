"""Private original-joint opening at fixed final grasp; no model geometry edits."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import root_scalar
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle
R=Path(__file__).resolve().parents[1];source=R/'reports/single_plate_explicit_edge_reset_benchmark_v3.json';s=json.loads(source.read_text());mfile=R/'outputs/fourth_full_sequence/physics_o6_C6_upperbodyhold_v1/effective_model.mjb';m=mj.MjModel.from_binary_path(str(mfile));d=mj.MjData(m);h=ArmObjectFeedback(m,dict(np.load(R/'outputs/fourth_full_sequence/physics_single_plate_groundaware_stand_v1/source_reference.npz')));o=ConvexSeparationOracle(m);q=np.asarray(s['best']['planning_grasp_qpos']).copy();closed=q.copy();digits=['thumb','index','middle','ring','pinky'];master=[m.joint('rh_'+n).id for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];relation=[(int(m.eq_obj1id[e]),int(m.eq_obj2id[e]),float(m.eq_data[e,1]))for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT];hg={v:[g for g in h.hand_geoms[1]if(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+v)]for v in digits};og=h.object_geoms[0]
def update():
 for ch,pa,ratio in relation:q[m.jnt_qposadr[ch]]=ratio*q[m.jnt_qposadr[pa]]
 d.qpos[:]=q;d.qvel[:]=0.;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def dg(v):return min(float(o.query(d,g,t)[0])for g in hg[v]for t in og)
rows=[]
for v,j in zip(digits,master[1:]):
 qa=int(m.jnt_qposadr[j]);x0=float(q[qa]);low,high=m.jnt_range[j].copy()
 for ch,pa,ratio in relation:
  if pa==j:low=max(low,m.jnt_range[ch,0]/ratio);high=min(high,m.jnt_range[ch,1]/ratio)
 low=max(low+1e-6,x0-.12);high=min(high-1e-6,x0+.12);cache=[]
 def f(x):
  q[qa]=x;update();g=dg(v);cache.append((float(x),float(g)));return g-.0045
 f0=f(x0);fp=f(min(high,x0+.02));slope=(fp-f0)/(min(high,x0+.02)-x0);estimate=float(np.clip(x0-f0/slope,low,high))if abs(slope)>1e-8 else x0;f(estimate)
 points=sorted(set([low,high,x0,estimate]));values=[(x,f(x))for x in points];brackets=[(x,y)for(x,fx),(y,fy)in zip(values,values[1:])if fx*fy<=0]
 if brackets:
  aa,bb=min(brackets,key=lambda p:abs((p[0]+p[1])/2-x0));result=root_scalar(f,bracket=(aa,bb),xtol=1e-7,maxiter=12);x=float(result.root)
 else:x=min(cache,key=lambda p:abs(p[1]-.0045))[0]
 q[qa]=x;update();row={'digit':v,'master':m.joint(j).name,'closed_command_rad':x0,'open_command_rad':x,'delta_rad':x-x0,'object_gap_m':dg(v),'samples':cache};rows.append(row);print(json.dumps(row),flush=True)
update();mj.mj_forward(m,d);gap={};
for v in digits:
 sat,g,t=min((float(o.query(d,g,t)[0]),g,t)for g in hg[v]for t in og);gap[v]={'SAT_gap_m':sat,'native_gap_m':min(float(mj.mj_geomDistance(m,d,g,t,.1,None))for g in hg[v]for t in og),'closest_pair':[m.geom(g).name,m.geom(t).name]}
tablegap=min(float(o.query(d,g,t)[0])for g in h.hand_geoms[1]for t in h.table_geoms);othergap=min(float(o.query(d,g,t)[0])for g in h.hand_geoms[1]for group in h.object_geoms[1:]for t in group);initial=np.asarray(s['best']['initial_qpos']).copy();names=[m.joint(j).name or''for j in range(m.njnt)];targets={names[j]:float(q[m.jnt_qposadr[j]])for j in range(m.njnt)if names[j].startswith('rh_')}
for j,n in enumerate(names):
 if n in targets:initial[m.jnt_qposadr[j]]=targets[n]
nativecontact=[{'geoms':[m.geom(int(g)).name for g in c.geom],'dist_m':float(c.dist)}for c in d.contact[:d.ncon]if any(int(g)in h.hand_geoms[1]for g in c.geom)];finalsame=all(float(closed[m.jnt_qposadr[m.joint(n).id]])==float(v)for n,v in s['best']['template']['finger_joint_targets'].items());out={'scope':'Private FK open-preshape planning only. All final-grasp arm/root/body/objects held exactly at original v3 planned grasp; only six original masters and original mimic relations vary. No hand/model/object geometry or physical state edited. Initial_qpos_open is an explicitly declared reset alternative; after reset execute actuator controls only. Final closed template unchanged.','source_setup':str(source),'model':str(mfile),'model_sha256':hashlib.sha256(mfile.read_bytes()).hexdigest(),'final_closed_template':s['best']['template'],'closed_finger_targets_exactly_preserved':finalsame,'open_finger_joint_targets':targets,'open_master_targets':{names[j]:float(q[m.jnt_qposadr[j]])for j in master},'initial_qpos_open':initial.tolist(),'planning_open_grasp_qpos':q.tolist(),'open_grasp_digit_gaps':gap,'open_grasp_minimum_hand_table_gap_m':tablegap,'open_grasp_minimum_hand_other_objects_gap_m':othergap,'open_grasp_native_hand_contacts':nativecontact,'all_digit_gap_4_to_5mm_and_table_2mm':bool(all(.004<=v['SAT_gap_m']<=.005 for v in gap.values())and tablegap>=.002),'joint_adjustments':rows};dest=R/'reports/single_plate_open_preshape_v1.json';dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in['initial_qpos_open','planning_open_grasp_qpos','joint_adjustments','final_closed_template']}),flush=True)
