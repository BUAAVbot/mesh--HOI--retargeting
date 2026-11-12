"""Private finite-sampled arm paths from the actual stable edge benchmark."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle

ROOT=Path(__file__).resolve().parents[1];setup=json.loads((ROOT/'reports/single_plate_explicit_edge_reset_benchmark_v3.json').read_text());stand=ROOT/'outputs/fourth_full_sequence/physics_single_plate_groundaware_stand_v1';z=np.load(stand/'settle_trace.npz');actual=z['qpos'][-1].copy();m=mj.MjModel.from_binary_path(str(ROOT/'outputs/fourth_full_sequence/physics_o6_C6_upperbodyhold_v1/effective_model.mjb'));d=mj.MjData(m);ref=dict(np.load(stand/'source_reference.npz'));h=ArmObjectFeedback(m,ref);o=ConvexSeparationOracle(m);c=h.chains[1];qa=c['qa'];va=c['va'];jids=c['joint_ids'];site=m.site('rh_palm_control_site').id;obj=h.object_bids[0];oq=h.object_qadr[0];lo=m.jnt_range[jids,0]+1e-6;hi=m.jnt_range[jids,1]-1e-6;template=setup['best']['template'];OR=Rotation.from_quat(actual[oq+3:oq+7][[1,2,3,0]]).as_matrix();goal=actual[oq:oq+3]+OR@np.asarray(template['palm_offset_in_object_m']);GR=OR@Rotation.from_quat(np.asarray(template['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix();out=np.array([.35216658756772096,.9359373347617386,-.02508]);out/=np.linalg.norm(out);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);oldgrasp=np.asarray(setup['best']['planning_grasp_qpos'])[qa];qinitial=actual[qa].copy()
body_geoms=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith(('rh_','right_shoulder','right_elbow','right_wrist'))];targets=list(dict.fromkeys(h.table_geoms+sum(h.object_geoms,[])))
def fk(q):
 d.qpos[:]=actual;d.qpos[qa]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
 return d.site_xpos[site].copy(),d.xmat[h.hand_bids[1]].reshape(3,3).copy()
def pose(q,point,rot):
 pp,rr=fk(q);return float(np.linalg.norm(pp-point)),float(np.linalg.norm(Rotation.from_matrix(rot@rr.T).as_rotvec()))
def solve(point,rot,seed):
 def evaluate(q,jac=False):
  pp,rr=fk(q);v=np.r_[35*(pp-point),-.25*Rotation.from_matrix(rot@rr.T).as_rotvec(),1e-5*(q-seed)]
  if not jac:return v
  mj.mj_jacSite(m,d,JP,JR,site);return np.r_[35*JP[:,va],.25*JR[:,va],1e-5*np.eye(7)]
 fit=least_squares(evaluate,np.clip(seed,lo,hi),jac=lambda q:evaluate(q,True),bounds=(lo,hi),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 return fit.x,pose(fit.x,point,rot)
def bounds(g):
 rr=d.geom_xmat[g].reshape(3,3);cc=d.geom_xpos[g]+rr@m.geom_aabb[g,:3];ee=abs(rr)@m.geom_aabb[g,3:];return cc-ee,cc+ee
def clearance(q):
 fk(q);bb={g:bounds(g)for g in body_geoms+targets};best=[float('inf'),None,None];bad=[]
 for g in body_geoms:
  gl,gu=bb[g]
  for t in targets:
   tl,tu=bb[t];lower=float(np.linalg.norm(np.maximum(np.maximum(gl-tu,tl-gu),0)))
   if lower>.012:gap=lower
   elif all(m.geom_type[x]in(mj.mjtGeom.mjGEOM_MESH,mj.mjtGeom.mjGEOM_CYLINDER)for x in[g,t]):gap=float(o.query(d,g,t)[0])
   else:gap=float(mj.mj_geomDistance(m,d,g,t,.1,None))
   if gap<best[0]:best=[gap,m.geom(g).name,m.geom(t).name]
   if gap<-.0001:bad.append({'robot_geom':m.geom(g).name,'scene_geom':m.geom(t).name,'gap_m':gap})
 return best,bad
solutions=[]
for label,point in [('grasp',goal),('outside_low',goal+out*.08)]:
 for i,start in enumerate([oldgrasp,qinitial,(oldgrasp+qinitial)*.5]):
  q,e=solve(point,GR,start);solutions.append({'waypoint':label,'seed_index':i,'q':q.tolist(),'position_error_m':e[0],'orientation_error_rad':e[1]})
grasp=min([x for x in solutions if x['waypoint']=='grasp'],key=lambda x:x['position_error_m']+.1*x['orientation_error_rad']);outside=min([x for x in solutions if x['waypoint']=='outside_low'],key=lambda x:x['position_error_m']+.1*x['orientation_error_rad']);paths=[]
for label,waypoints in [('direct_joint',[qinitial,np.asarray(grasp['q'])]),('staged_joint',[qinitial,np.asarray(outside['q']),np.asarray(grasp['q'])])]:
 rows=[]
 for segment,(start,end)in enumerate(zip(waypoints,waypoints[1:])):
  for alpha in np.linspace(0,1,31):
   q=start+alpha*(end-start);best,bad=clearance(q);palm,rot=fk(q);rows.append({'segment':segment,'alpha':float(alpha),'q':q.tolist(),'palm_world_m':palm.tolist(),'orientation_error_to_grasp_rad':float(np.linalg.norm(Rotation.from_matrix(GR@rot.T).as_rotvec())),'minimum_scene_gap_m':best[0],'closest_pair':best[1:],'penetrations':bad})
 result={'path':label,'samples':rows,'minimum_conservative_scene_gap_m':min(x['minimum_scene_gap_m']for x in rows),'sampled_no_penetration':all(not x['penetrations']for x in rows),'max_joint_segment_difference_rad':max(float(np.max(abs(x-y)))for x,y in zip(waypoints,waypoints[1:]))};paths.append(result);print(json.dumps({k:v for k,v in result.items()if k!='samples'}),flush=True)
report={'scope':'Private FK and finite-sampled collision screening only, not executed motion or continuous-path proof. All non-RH-arm coordinates and six actual settled objects held exact from physical two-second ground-aware stand. Far pairs use proven AABB separation; close mesh/cylinder pairs use conservative SAT; capsule pairs native distance.','scene_trial':str(stand),'actual_scene_time_s':float(z['time_s'][-1]),'actual_initial_arm_q':qinitial.tolist(),'original_planning_grasp_arm_q':oldgrasp.tolist(),'initial_to_original_grasp_joint_delta_rad':(oldgrasp-qinitial).tolist(),'hinge_joint_names':[m.joint(j).name for j in jids],'target_palm_world_m':goal.tolist(),'target_R_world':GR.tolist(),'IK_solutions':solutions,'paths':paths};dest=ROOT/'reports/single_plate_benchmark_joint_transfer_v1.json';dest.write_text(json.dumps(report,indent=2)+'\n');print(str(dest),flush=True)
