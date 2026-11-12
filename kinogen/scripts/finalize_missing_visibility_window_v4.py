"""Schema-only closure for the completed fixed pass; never fits or alters pose."""
import json
from user_visibility_window_v4 import ROOT,sha
def main():
 import numpy as np
 from user_body_kinematics import G1Kinematics
 from hand_geometry import box_signed_distance
 out=ROOT/'outputs/user_reconstruction_v4';bp=out/'stage_window_body.npz';hp=out/'stage_window_hand.npz';parentb=out/'stage_body.npz';parenth=out/'stage_hand.npz';rp=ROOT/'reports/user-v4-missing-window.json';r=json.loads(rp.read_text());assert r['status']=='fixed_pass_completed';B={k:v.copy()for k,v in np.load(bp).items()};H={k:v.copy()for k,v in np.load(hp).items()};B0=np.load(parentb);H0=np.load(parenth);frames=r['frames'];accepted=[x['frame']for x in r['records']if x['accepted']]
 kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');arm=[kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')];geometry=[]
 for v in kin.visuals:
  if '_hand_ro_'in v['link_name']or not kin.ancestor[v['link_index'],arm].any():continue
  p=v['vertices_local'];surface=np.concatenate((np.unique(p,axis=0),p[v['faces']].mean(axis=1)));geometry.append((v['link_name'],v['link_index'],surface))
 box=np.load(ROOT/'outputs/user_reconstruction_v2/box_trajectory.npz');size=box['box_size_xyz_m'];nonhand_audit=[]
 def raw_arm_stats(f,T):
  rows=[]
  for name,i,p in geometry:
   sdf=box_signed_distance(p@T[i,:3,:3].T+T[i,:3,3],box['box_world_transform'][f],size);rows.append({'link':name,'min_sdf_m':float(sdf.min()),'max_penetration_m':float(max(0,-sdf.min())),'fraction_beyond_5mm':float(np.mean(sdf<-.005))})
  return rows
 def revert(item,reason):
  f=item['frame'];B['joint_positions_rad'][f]=B0['joint_positions_rad'][f];B['link_world_transforms'][f]=B0['link_world_transforms'][f];H['body_joint_positions_rad_refined'][f]=B0['joint_positions_rad'][f];H['actuator_positions'][f]=H0['actuator_positions'][f];H['joint_positions'][f]=H0['joint_positions'][f];item['accepted']=False;item['final_audit_rejections']=item.get('final_audit_rejections',[])+[reason]
 for item in r['records']:
  if not item['accepted']:continue
  f=item['frame'];before=raw_arm_stats(f,B0['link_world_transforms'][f]);after=raw_arm_stats(f,B['link_world_transforms'][f]);bad=[b['link']for a,b in zip(before,after)if b['max_penetration_m']>a['max_penetration_m']+1e-6 or b['fraction_beyond_5mm']>a['fraction_beyond_5mm']+1e-7];nonhand_audit.append({'frame':f,'source':before,'candidate':after,'new_penetration_links':bad})
  if bad:revert(item,'new_nonhand_raw_arm_box_penetration')
 while True:
  viol=np.abs(np.diff(B['joint_positions_rad'],axis=0))/np.diff(B['timestamps'])[:,None]>B['joint_velocity_limits_rad_s']+1e-5
  if not viol.any():break
  intervals=np.flatnonzero(viol.any(axis=1));affected=[x for x in r['records']if x['accepted']and any(x['frame']in (i,i+1)for i in intervals)];assert affected,'Source parent unexpectedly violates original body velocity'
  for item in affected:revert(item,'source_velocity_after_audit_fallback')
 accepted=[x['frame']for x in r['records']if x['accepted']]
 wrist=[kin.body_index[str(n)]for n in H['all_wrist_joint_names']];H['all_wrist_joint_positions_rad']=B['joint_positions_rad'][:,wrist];H['wrist_joint_positions_rad']=B['joint_positions_rad'][:,[kin.body_index[str(n)]for n in H['wrist_joint_names']]];H['wrist_yaw_positions_rad']=B['joint_positions_rad'][:,[kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]]
 for item in r['records']:
  f=item['frame'];passed=bool(item['accepted']);B['parent_hand_image_fit_valid'][f,1]=passed;B['parent_hand_side_valid'][f,1]=False;H['image_fit_valid'][f,1]=passed;H['new_hand_candidate_selected'][f,1]=passed;H['refinement_selected_conditional'][f,1]=passed;H['wrist_refinement_accepted'][f,1]=passed
 B.update(source_parent_body_path=np.asarray(str(parentb.relative_to(ROOT))),source_parent_body_sha256=np.asarray(sha(parentb)),finger_actuator_positions=H['actuator_positions'].copy(),finger_actuator_names=H['actuator_names'].copy(),finger_actuator_units=H['actuator_units'].copy(),finger_joint_positions=H['joint_positions'].copy(),finger_joint_names=H['joint_names'].copy(),finger_joint_units=H['joint_units'].copy(),finger_joint_positions_included=np.asarray(True),candidate_is_metric_ground_truth=np.asarray(False));np.savez_compressed(bp,**B);H.update(source_body_path=np.asarray(str(bp.relative_to(ROOT))),source_body_sha256=np.asarray(sha(bp)),body_sha256=np.asarray(sha(bp)),candidate_is_metric_ground_truth=np.asarray(False));H['contact_geometry_uncertain']=~box['valid'].astype(bool)
 for key in ['hand_surface_min_signed_distance_m','hand_surface_fraction_beyond_5mm','contact_nonpenetration_sample_valid']:H.pop(key,None)
 np.savez_compressed(hp,**H)
 unchanged=[i for i in range(121)if i not in accepted]
 for key in ['joint_positions_rad','link_world_transforms']:assert np.array_equal(B[key][unchanged],B0[key][unchanged])
 for key in ['base_xyz_m','base_quaternion_xyzw']:assert np.array_equal(B[key],B0[key])
 assert np.array_equal(H['actuator_positions'][unchanged],H0['actuator_positions'][unchanged]);assert np.array_equal(B['finger_joint_positions'],H['joint_positions']);assert np.array_equal(B['joint_positions_rad'],H['body_joint_positions_rad_refined'])
 from user_rohand_coupling import ROHandCoupling
 coupling=ROHandCoupling(ROOT);candidates=[];cq=[];ca=[];cf=[]
 for item in r['records']:
  trials=item.get('trials',[])
  if not trials:continue
  near=[t for t in trials if set(t['rejections']).issubset({'raw_mesh_penetration'})and t['metrics']['visibility']['visible_mask_iou']>=.2];chosen=min(near,key=lambda t:t['metrics']['max_hand_box_penetration_m'])if near else max(trials,key=lambda t:t['metrics']['visibility']['visible_mask_iou']);f=item['frame'];z=np.asarray(chosen['x']);qq=B0['joint_positions_rad'][f].copy();qq[arm]=z[:7];aa=H0['actuator_positions'][f].copy();lo=np.asarray(coupling.lower[6:]);hi=np.asarray(coupling.upper[6:]);aa[6:]=lo+z[7:]*(hi-lo);cq.append(qq);ca.append(aa);cf.append(f);candidates.append({'frame':f,'phase':chosen['phase'],'rejections':chosen['rejections'],'metrics':chosen['metrics'],'body_contour_metrics':chosen['body_contour_metrics'],'x':chosen['x'],'adopted_in_zero_target_window':f in accepted})
 cp=out/'conditional_front_candidates.npz';np.savez_compressed(cp,frame_indices=np.asarray(cf),timestamps=B0['timestamps'][cf],body_joint_names=B0['joint_names'],body_joint_positions_rad=np.asarray(cq),base_xyz_m=B0['base_xyz_m'][cf],base_quaternion_xyzw=B0['base_quaternion_xyzw'][cf],actuator_names=H0['actuator_names'],actuator_positions=np.asarray(ca),actuator_units=H0['actuator_units'],joint_names=H0['joint_names'],joint_positions=coupling.expand_array(np.asarray(ca)),joint_units=H0['joint_units'],source_body_sha256=np.asarray(sha(parentb)),source_hand_sha256=np.asarray(sha(parenth)),source_body_path=np.asarray(str(parentb.relative_to(ROOT))),source_hand_path=np.asarray(str(parenth.relative_to(ROOT))),observed=np.zeros((len(cf),50),bool),uncertain=np.ones((len(cf),50),bool),strict_valid=np.zeros(len(cf),bool),zero_target_gates_passed=np.asarray([not x['rejections']for x in candidates]),adopted_in_zero_target_window=np.asarray([f in accepted for f in cf]))
 r['outputs']={'body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)},'conditional_candidates':{'path':str(cp.relative_to(ROOT)),'sha256':sha(cp)}};r['best_front_conditional_candidates']=candidates;r['nonhand_raw_arm_source_non_deterioration_audit']=nonhand_audit;r['final_schema_closed']=True;r['accepted_frames']=accepted;r['rejected_frames']=[f for f in frames if f not in accepted];r['rejected_exact_sameframe_parent']=True;r['only_allowed_changes']='Accepted subset of rightarm7 and official coupled6 controls at53,56,57,59,61,62; source54 and all otherframes/base/feet exact frozen stage54 parent.';r['finite_budget']={'global_seconds':570,'per_frame_seconds':86,'front_max_nfev':22,'ray_max_nfev':12,'hard_maxiter':17,'seeds_per_frame':1};r['strict_valid']=False;rp.write_text(json.dumps(r,indent=2));print(json.dumps({'outputs':r['outputs'],'accepted_frames':accepted,'rejected_frames':r['rejected_frames'],'elapsed_seconds':r['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
