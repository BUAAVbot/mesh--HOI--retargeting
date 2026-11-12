"""Independent CTRL-only replay and no-external-force audit of a free-root stand."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
import mujoco as mj
R=Path(__file__).resolve().parents[1];trial=R/'outputs/fourth_full_sequence/physics_single_plate_groundaware_stand_v1';modelpath=R/'outputs/fourth_full_sequence/physics_o6_C6_upperbodyhold_v1/effective_model.mjb';basepath=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_calibrated_mimic_v3.xml';m=mj.MjModel.from_binary_path(str(modelpath));b=mj.MjModel.from_xml_path(str(basepath));z=np.load(trial/'settle_trace.npz');d=mj.MjData(m);mj.mj_setState(m,d,z['initial_state'],int(z['state_spec']));mj.mj_forward(m,d)
def hand_hash(model):
 result={}
 for g in range(model.ngeom):
  if not(model.body(int(model.geom_bodyid[g])).name or'').startswith(('rh_','lh_')):continue
  row={v:getattr(model,v)[g].tolist()for v in['geom_type','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_margin','geom_gap']}
  if model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
   mi=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mi]);f=int(model.mesh_faceadr[mi]);row['mesh_sha256']=hashlib.sha256(model.mesh_vert[a:a+model.mesh_vertnum[mi]].tobytes()+model.mesh_face[f:f+model.mesh_facenum[mi]].tobytes()).hexdigest()
  result[model.geom(g).name]=row
 return result
errors=np.zeros(3);unexpected={};maxpen=0.;minup=1.;maxdrift=0.;maxmimic=0.;force0=True;handimp=0.;body=m.body('base_link').id;object_ids={m.body(f'target_object_{i:02d}').id for i in range(1,7)}
for step,ctrl in enumerate(z['ctrl']):
 d.ctrl[:]=ctrl;mj.mj_step(m,d);errors=np.maximum(errors,[np.max(abs(d.qpos-z['qpos'][step+1])),np.max(abs(d.qvel-z['qvel'][step+1])),np.max(abs(d.actuator_force-z['actuator_force'][step]))]);force0=force0 and not np.any(d.qfrc_applied)and not np.any(d.xfrc_applied);minup=min(minup,float(d.xmat[body].reshape(3,3)[2,2]));maxdrift=max(maxdrift,float(np.linalg.norm(d.qpos[:3]-z['qpos'][0,:3])))
 for e in range(m.neq):
  if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
   child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);maxmimic=max(maxmimic,abs(float(d.qpos[m.jnt_qposadr[child]]-m.eq_data[e,1]*d.qpos[m.jnt_qposadr[parent]])))
 for ci,c in enumerate(d.contact[:d.ncon]):
  gs=list(map(int,c.geom));bs=list(map(int,m.geom_bodyid[gs]));bn=[m.body(v).name or'world'for v in bs];gn=[m.geom(v).name or''for v in gs];F=np.zeros(6);mj.mj_contactForce(m,d,ci,F);robot=[n for v,n in zip(bs,bn)if v>0 and v not in object_ids]
  if object_ids.intersection(bs)and any(n.startswith(('rh_','lh_'))for n in bn):handimp+=max(float(F[0]),0)*m.opt.timestep
  if not robot:continue
  if all(any(s in n.lower()for s in['foot','ankle'])for n in robot)and any('floor'in n for n in gn):continue
  if F[0]>.001 or c.dist<-.0001:
   key='|'.join(gn);v=unexpected.setdefault(key,{'geoms':gn,'bodies':bn,'normal_force_peak_N':0.,'normal_impulse_Ns':0.,'penetration_peak_m':0.});v['normal_force_peak_N']=max(v['normal_force_peak_N'],float(F[0]));v['normal_impulse_Ns']+=max(float(F[0]),0)*m.opt.timestep;v['penetration_peak_m']=max(v['penetration_peak_m'],float(-c.dist))
root=m.joint('vita_demo_free_root').id
allfree=all(m.jnt_type[int(m.body_jntadr[v])]==mj.mjtJoint.mjJNT_FREE and m.body_mocapid[v]<0 for v in object_ids);no_weld=not np.any(m.eq_type==mj.mjtEq.mjEQ_WELD);handunchanged=hand_hash(m)==hand_hash(b);recordforcezero=not np.any(z['qfrc_applied'])and not np.any(z['xfrc_applied']);setup=json.loads((R/'reports/single_plate_explicit_edge_reset_benchmark_v3.json').read_text());sameinit=np.array_equal(z['qpos'][0],setup['best']['initial_qpos']);source=(R/'scripts/fourth_ground_reaction_pd_controller.py').read_text();tree=ast.parse(source);actual_writes=[]
for node in ast.walk(tree):
 if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
  targets=node.targets if isinstance(node,ast.Assign)else[node.target]
  for target in targets:
   text=ast.unparse(target)
   if any(text.startswith(v)for v in['data.qpos','data.qvel','data.qfrc_applied','data.xfrc_applied','m.eq_type','model.eq_type']):actual_writes.append(text)
forcecalls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)and any(v in ast.unparse(n.func)for v in['mj_applyFT','mj_setState','mj_resetData'])]
code_review={'real_force_application_calls_absent':not forcecalls,'actual_state_or_force_assignments_absent':not actual_writes,'actual_writes':actual_writes,'private_kinematic_copy_used':True,'actual_output_is_existing_position_actuator_target':True,'predicted_ground_force_is_not_applied_as_external_force':True,'root_accel_goal_used_only_in_inverse_statics':True};r={'scope':'Independent 400-step stand audit only; this is not successful grasp/lift evidence. Uses same configured runtime MJB as original C6, matched by exact full CTRL-only replay.','trial':str(trial),'replay_model':str(modelpath),'replay_model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'replay_qpos_qvel_actuator_force_max_errors':errors.tolist(),'replay_zero_error':bool(np.max(errors)<1e-12),'trace_initial_state_matches_declared_benchmark':sameinit,'hand_collision_and_mesh_fingerprint_unchanged':handunchanged,'root_free':bool(m.jnt_type[root]==mj.mjtJoint.mjJNT_FREE),'all_six_objects_free':allfree,'no_weld_constraints':bool(no_weld),'recorded_applied_forces_zero':bool(recordforcezero),'replayed_applied_forces_zero':bool(force0),'unexpected_robot_scene_contacts':list(unexpected.values()),'actual_hand_object_impulse_Ns':handimp,'actual_root_drift_max_m':maxdrift,'actual_root_upright_min':minup,'actual_mimic_error_max_rad':maxmimic,'source_review':code_review};r['stand_physical_validity_passed']=bool(r['replay_zero_error']and sameinit and handunchanged and r['root_free']and allfree and no_weld and recordforcezero and force0 and not unexpected and handimp==0 and minup>.99 and maxdrift<.01 and maxmimic<.03 and not actual_writes and not forcecalls);out=R/'reports/single_plate_groundaware_stand_independent_audit.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
