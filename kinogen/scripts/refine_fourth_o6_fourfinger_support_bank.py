"""58 local MCP probes for ring/pinky support-bank preshaping, no physics edits.

The root, waist, arms, thumb, index, middle and all six actual object poses are
fixed to the supplied candidate. Only two O6 native MCPs and their passive DIP
ratios are changed in private geometry, with exact finite-table SAT guards.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns,wrench_lp
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=json.loads(a.candidate.read_text());m=mj.MjModel.from_xml_path(src['model']);d=mj.MjData(m);q0=np.asarray(src['best']['qpos']);h=ArmObjectFeedback(m,dict(np.load(a.reference)));oracle=ConvexSeparationOracle(m)
objects=sum(h.object_geoms,[]);obj=h.object_bids[0];bottom=m.geom('target_object_01_bottom_collision').id;rows={};chosen={};qbest=q0.copy()
def forward(q):
 d.qpos[:]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def bounds(g):
 R=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+R@m.geom_aabb[g,:3];e=abs(R)@m.geom_aabb[g,3:];return c-e,c+e
def allgap(hand):
 vals=[]
 for g in hand:
  lo,hi=bounds(g)
  for o in objects:
   ol,oh=bounds(o);lower=float(np.linalg.norm(np.maximum(np.maximum(lo-oh,ol-hi),0)))
   vals.append(oracle.query(d,g,o)[0]if lower<.006 else lower)
 return min(vals)
for digit in ['ring','pinky']:
 master=m.joint('rh_'+digit+'_mcp_pitch').id;child=m.joint('rh_'+digit+'_dip').id;qa=int(m.jnt_qposadr[master]);cq=int(m.jnt_qposadr[child]);ratio=next(float(m.eq_data[e,1])for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj1id[e])==child)
 geoms=[g for g in h.hand_geoms[1]if(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+digit)];local=[]
 grid=np.linspace(max(.20,m.jnt_range[master,0]),min(1.60,m.jnt_range[master,1],m.jnt_range[child,1]/ratio),29);grid[int(np.argmin(abs(grid-q0[qa])))]=q0[qa]
 for angle in grid:
  q=q0.copy();q[qa]=angle;q[cq]=angle*ratio;forward(q)
  table=min(oracle.query(d,g,h.table)[0]for g in geoms);objectgap=allgap(geoms);bottomgap=min(oracle.query(d,g,bottom)[0]for g in geoms)
  feasible=table>=.002 and objectgap>=.0004
  row={'mcp_rad':float(angle),'dip_rad':float(angle*ratio),'finite_table_gap_m':table,'minimum_all_six_object_gap_m':objectgap,'bottom_disc_gap_m':bottomgap,'geometry_safe':feasible};local.append(row)
 safe=[r for r in local if r['geometry_safe']]
 best=min(safe,key=lambda r:(abs(r['bottom_disc_gap_m']-.001),abs(r['mcp_rad']-q0[qa])))if safe else min(local,key=lambda r:(max(0,.002-r['finite_table_gap_m'])+max(0,.0004-r['minimum_all_six_object_gap_m']),abs(r['bottom_disc_gap_m']-.001)))
 chosen[digit]=best;rows[digit]=local
 if best['geometry_safe']:qbest[qa]=best['mcp_rad'];qbest[cq]=best['dip_rad']
 print(json.dumps({'digit':digit,'chosen':best,'safe_samples':len(safe),'total_samples':len(local)}),flush=True)
forward(qbest);mj.mj_forward(m,d);names=[m.joint(j).name or''for j in range(m.njnt)];masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for k,j in enumerate(masters):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e])in masters:P[m.jnt_dofadr[int(m.eq_obj1id[e])],masters.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
margin=m.geom_margin.copy();m.geom_margin[h.hand_geoms[1]]=.0015;mj.mj_forward(m,d);cols=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle','rh_ring','rh_pinky'),projection=P,max_gap_m=.003)
lp=wrench_lp(cols['columns'],cols['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=None,motor_torque_matrix=cols['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.35);m.geom_margin[:]=margin
template=dict(src['best']['template']);template['finger_joint_targets']={name:float(qbest[m.jnt_qposadr[j]])for j,name in enumerate(names)if name.startswith('rh_')};template['controlled_digits']=['thumb','index','middle'];template['required_contact_digits']=['thumb','index','middle'];template['ring_pinky_safe_bank_preshape_only']=True
changed=np.flatnonzero(qbest!=q0).tolist();allowed=[int(m.jnt_qposadr[m.joint('rh_'+di+'_'+part).id])for di in ['ring','pinky']for part in ['mcp_pitch','dip']];assert set(changed)<=set(allowed)
result={'schema':'fourth.o6.four_finger_support_bank_local.v1','scope':'Private geometry and static wrench only; only ring/pinky MCP and coupled DIP change, all actual object poses/root/waist/arms/three main contact digits fixed. No physical motion claimed.','source_candidate':str(a.candidate),'source_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'model':src['model'],'model_sha256':src['model_sha256'],'evaluations':58,'rows':rows,'chosen':chosen,'changed_qpos_indices':changed,'best':{'qpos':qbest.tolist(),'template':template,'wrench_budget035':lp,'prospective_contacts':cols['patches']},'all_other_qpos_exactly_preserved':True}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'changed_qpos_indices':changed,'wrench':lp,'prospective_contacts':cols['patches']}),flush=True)
