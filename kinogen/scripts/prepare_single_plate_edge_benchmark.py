"""Explicit single-plate-at-edge RESET benchmark; not raw-source reproduction."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_contact_wrench import contact_columns,wrench_lp

R=Path(__file__).resolve().parents[1];modelpath=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_calibrated_mimic_v3.xml';m=mj.MjModel.from_xml_path(str(modelpath));d=mj.MjData(m)
ref=dict(np.load(R/'outputs/fourth_full_sequence/reference_o6_support_posture_stand_v2.npz'));h=ArmObjectFeedback(m,ref,{'initial_geometric_seed':'actual_state'});src=R/'reports/fourth_o6_deepgrasp_fourfinger_bank_v2.json';receipt=json.loads(src.read_text());best=receipt['best'];template=best['template'];names=[m.joint(j).name or''for j in range(m.njnt)]
raw=np.load(R/'outputs/fourth_full_sequence/physics_sonic15225_o6_support_absolute_hold_v1_5s/physical_trace.npz')['qpos'][0].copy();settled=np.load(R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_source_table_v1_settled_2s.npz')['final_object_pose_world_xyz_wxyz'];base=raw.copy();base[:59]=ref['qpos'][0];bank=np.asarray(best['qpos']);waist=[j for j,n in enumerate(names)if n.startswith('waist')]
for j in waist:base[int(m.jnt_qposadr[j])]=bank[int(m.jnt_qposadr[j])]
for j,n in enumerate(names):
 if n.startswith('rh_'):base[int(m.jnt_qposadr[j])]=bank[int(m.jnt_qposadr[j])]
relations=[(int(m.eq_obj1id[e]),int(m.eq_obj2id[e]),float(m.eq_data[e,1]))for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
def project(q):
 for child,parent,ratio in relations:q[int(m.jnt_qposadr[child])]=ratio*q[int(m.jnt_qposadr[parent])]
out=np.array([.35216658756772096,.9359373347617386,0.]);out[2]=-(.010271048830894367*out[0]+.02292500590934548*out[1])/.999684425036206;out/=np.linalg.norm(out);delta=.05*out;obj=h.object_bids[0];oq=h.object_qadr[0];radial=raw[:2]-settled[0,:2]-delta[:2];radial/=np.linalg.norm(radial)
chain=h.chains[1];qa=chain['qa'];va=chain['va'];jids=chain['joint_ids'];site=m.site('rh_palm_control_site').id;JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);lo=m.jnt_range[jids,0]+1e-5;hi=m.jnt_range[jids,1]-1e-5
def solve(q,point,rot,start):
 def f(x,jac=False):
  d.qpos[:]=q;d.qpos[qa]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);err=Rotation.from_matrix(rot@d.xmat[h.hand_bids[1]].reshape(3,3).T).as_rotvec();r=np.r_[d.site_xpos[site]-point,-.12*err,1e-5*(x-start)]
  if not jac:return r
  mj.mj_jacSite(m,d,JP,JR,site);return np.r_[JP[:,va],.12*JR[:,va],1e-5*np.eye(7)]
  
 fit=least_squares(f,np.clip(start,lo,hi),jac=lambda x:f(x,True),bounds=(lo,hi),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10);q=q.copy();q[qa]=fit.x;d.qpos[:]=q;mj.mj_forward(m,d);return q,float(np.linalg.norm(d.site_xpos[site]-point)),float(np.linalg.norm(Rotation.from_matrix(rot@d.xmat[h.hand_bids[1]].reshape(3,3).T).as_rotvec()))
def scene_bad(q):
 d.qpos[:]=q;d.qvel[:]=0;mj.mj_forward(m,d);bad=[]
 for c in d.contact[:d.ncon]:
  if c.dist>=-.00001:continue
  gs=list(map(int,c.geom));bs=[int(m.geom_bodyid[g])for g in gs];bn=[m.body(b).name or'world'for b in bs];robot=[n for n in bn if n!='world'and not n.startswith('target_object')]
  if not robot:continue
  if all(any(v in n.lower()for v in ['ankle','foot'])for n in robot)and any('floor'in(m.geom(g).name or'')for g in gs):continue
  bad.append({'bodies':bn,'geoms':[m.geom(g).name for g in gs],'penetration_m':float(-c.dist)})
 return bad
rows=[]
for retreat in [0.,.04,.06]:
 q=base.copy();q[:2]+=retreat*radial
 for i,adr in enumerate(h.object_qadr):q[adr:adr+7]=settled[i]
 q[oq:oq+3]+=delta;project(q);O=Rotation.from_quat(q[oq+3:oq+7][[1,2,3,0]]).as_matrix();palm=q[oq:oq+3]+O@np.asarray(template['palm_offset_in_object_m']);rot=O@Rotation.from_quat(np.asarray(template['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix();grasp,perr,rerr=solve(q,palm,rot,bank[qa]);pre,pe,re=solve(q,palm+np.r_[out[:2]*.08,.10],rot,grasp[qa]);qreset=pre.copy();qreset[59:]=raw[59:];qreset[oq:oq+3]+=delta;bad=scene_bad(qreset)
 table=h.table;hand=h.hand_geoms[1];objg=sum(h.object_geoms,[]);mind=min(float(mj.mj_geomDistance(m,d,g,og,10.,None))for g in hand for og in objg);tablemin=min(float(mj.mj_geomDistance(m,d,g,table,10.,None))for g in hand)
 robotb=[b for b in range(1,m.nbody)if not(m.body(b).name or'').startswith('target_object')];com=np.sum(d.xipos[robotb]*m.body_mass[robotb,None],axis=0)/np.sum(m.body_mass[robotb]);foot=[]
 for g in range(m.ngeom):
  if (m.geom(g).name or'').startswith('collision_foot'):
   c=d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@m.geom_aabb[g,:3];e=abs(d.geom_xmat[g].reshape(3,3))@m.geom_aabb[g,3:];foot.extend([[c[0]+sx*e[0],c[1]+sy*e[1]]for sx in [-1,1]for sy in[-1,1]])
 hull=ConvexHull(foot);margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])));rows.append({'retreat_m':retreat,'initial_qpos':qreset.tolist(),'planning_grasp_qpos':grasp.tolist(),'palm_goal_in_world_m':palm.tolist(),'pregrasp_position_error_m':pe,'pregrasp_orientation_error_rad':re,'grasp_position_error_m':perr,'grasp_orientation_error_rad':rerr,'initial_hand_object_surface_gap_m':mind,'initial_hand_table_surface_gap_m':tablemin,'initial_unexpected_robot_collisions':bad,'robot_COM_world_m':com.tolist(),'projected_COM_foot_AABB_hull_margin_m':margin,'geometry_reset_acceptable':bool(mind>=.05 and tablemin>=.05 and not bad and margin>.005 and pe<.005),'template':template.copy()});print(json.dumps({k:v for k,v in rows[-1].items()if k not in['initial_qpos','planning_grasp_qpos','template']}),flush=True)
bestrow=min(rows,key=lambda x:(not x['geometry_reset_acceptable'],x['grasp_position_error_m']+.1*x['grasp_orientation_error_rad']));bestrow['template']['controlled_digits']=['thumb','index','middle','ring','pinky'];bestrow['template']['required_contact_digits']=['thumb','index','middle'];report={'scope':'Explicit synthetic single-plate tabletop-edge RESET benchmark, not original-video/raw-scene reproduction. Only plate1 reset translation is changed; no object writes after reset allowed. Geometry/COM checks are private planning, not physics proof.','model':str(modelpath),'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'original_hand_colliders_unchanged':True,'object1_reset_translation_world_m':delta.tolist(),'object1_reset_translation_length_m':.05,'robot_initial_pose_rule':'Nominal original validated stand legs/root orientation; candidate waist preset and collision-free pregrasp RH pose. Root retreat declared in best.','best':bestrow,'attempts':rows};path=R/'reports/single_plate_explicit_edge_reset_benchmark_v1.json';path.write_text(json.dumps(report,indent=2)+'\n');print(str(path),flush=True)
