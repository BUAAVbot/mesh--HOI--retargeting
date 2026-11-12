"""Read-only selection of saved front candidates; no optimization or interpolation."""
import os,json,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from user_visibility_window_v4 import ROOT,sha
def main():
 import numpy as np
 from user_body_kinematics import G1Kinematics
 from user_rohand_coupling import ROHandCoupling
 from hand_geometry import box_signed_distance
 from scipy.spatial.transform import Rotation
 started=time.monotonic();out=ROOT/'outputs/user_reconstruction_v4';bp=out/'stage_window_body.npz';hp=out/'stage_window_hand.npz';cp=out/'conditional_front_candidates.npz';assert sha(bp)=='348cd53423689764e6c111374adc4e436ed42c0e4c61deffadf9684d996b2e15';assert sha(hp)=='e5ba01877974ada0a56233a84b61cfdb5fb6aa00d117cee9d351ff341c4196ec';assert sha(cp)=='82fa25a4f9e029dd32f5060e88f375cfb251dfff666e5b4f160a3d83c645a332';B0=np.load(bp);H0=np.load(hp);C=np.load(cp);B={k:v.copy()for k,v in B0.items()};H={k:v.copy()for k,v in H0.items()};boxpath=ROOT/'outputs/user_reconstruction_v2/box_trajectory.npz';box=np.load(boxpath);coupling=ROHandCoupling(ROOT);kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');reportpath=ROOT/'reports/user-v4-missing-window.json';parentreport=json.loads(reportpath.read_text());q=B['joint_positions_rad'];act=H['actuator_positions'];Tf=B['link_world_transforms'];arm=[kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')];sourceq=q.copy();sourceact=act.copy();records=[];geometry=[]
 for v in kin.visuals:
  p=v['vertices_local'];geom=np.concatenate((np.unique(p,axis=0),p[v['faces']].mean(axis=1)));geometry.append((v['link_name'],v['link_index'],geom))
 def surfaces(f,T,hand=None):
  rows=[]
  for name,i,p in geometry:
   ishand='_hand_ro_'in name
   if hand is not None and ishand!=hand:continue
   sd=box_signed_distance(p@T[i,:3,:3].T+T[i,:3,3],box['box_world_transform'][f],box['box_size_xyz_m']);rows.append({'link':name,'minimum_sdf_m':float(sd.min()),'max_penetration_m':float(max(0,-sd.min())),'fraction_beyond_5mm':float(np.mean(sd<-.005))})
  return rows
 def fk(f,qq,aa):return kin.forward(B['base_xyz_m'][f],B['base_quaternion_xyzw'][f],qq,coupling.expand_one(aa))
 for f in [56,57,59,61]:
  row=int(np.flatnonzero(C['frame_indices']==f)[0]);saved=next(x for x in parentreport['best_front_conditional_candidates']if x['frame']==f);qq=C['body_joint_positions_rad'][row];aa=C['actuator_positions'][row];T=fk(f,qq,aa);raw=surfaces(f,T);nonhand=[x for x in raw if '_hand_ro_'not in x['link']];hand=[x for x in raw if x['link'].startswith('right_hand_ro_')];maxhand=max(x['max_penetration_m']for x in hand);maxnon=max(x['max_penetration_m']for x in nonhand);reasons=[x for x in saved['rejections']if x!='raw_mesh_penetration'];within=(qq>=kin.body_limits[:,0]-1e-9)&(qq<=kin.body_limits[:,1]+1e-9)
  if not within.all():reasons.append('original_body_position_limits')
  if maxhand>.005:reasons.append('raw_hand_exceeds_original_necessary_5mm')
  if maxnon>.005:reasons.append('raw_nonhand_exceeds_original_necessary_5mm')
  if not reasons:q[f]=qq;act[f]=aa;Tf[f]=T
  records.append({'frame':f,'candidate_phase':saved['phase'],'selected_for_conditional_comparison':not reasons,'strict_zero_target_passed':False,'source_non_deterioration_passed':False,'original_zero_target_rejections':saved['rejections'],'comparison_rejections':reasons,'max_hand_penetration_m':maxhand,'max_nonhand_penetration_m':maxnon,'raw_penetrating_links':[x for x in raw if x['max_penetration_m']>0],'hand_image_metrics':saved['metrics'],'body_contour_metrics':saved['body_contour_metrics']})
 def fallback(r,why):
  f=r['frame'];q[f]=sourceq[f];act[f]=sourceact[f];Tf[f]=B0['link_world_transforms'][f];r['selected_for_conditional_comparison']=False;r['comparison_rejections'].append(why)
 while True:
  speed=np.abs(np.diff(q,axis=0))/np.diff(B['timestamps'])[:,None];bad=np.flatnonzero(np.any(speed>B['joint_velocity_limits_rad_s']+1e-6,axis=1))
  if not len(bad):break
  hits=[r for r in records if r['selected_for_conditional_comparison']and any(r['frame']in(i,i+1)for i in bad)];assert hits,'Frozen parent violates original positive velocity specification'
  for r in hits:fallback(r,'original_velocity_in_actual_combination')
 selected=[r['frame']for r in records if r['selected_for_conditional_comparison']];other=[i for i in range(121)if i not in selected];assert np.array_equal(q[other],sourceq[other]);assert np.array_equal(act[other],sourceact[other]);assert np.array_equal(Tf[other],B0['link_world_transforms'][other]);assert np.array_equal(B['base_xyz_m'],B0['base_xyz_m']);assert np.array_equal(B['base_quaternion_xyzw'],B0['base_quaternion_xyzw'])
 # Original raw non-hand geometry is checked across the complete121-frame
 # combination. Existing F70 source penetration is reported, never erased.
 global_nonhand=[]
 for f in range(121):
  raw=surfaces(f,Tf[f],hand=False);maxpen=max(x['max_penetration_m']for x in raw);global_nonhand.append({'frame':f,'max_penetration_m':maxpen,'links_beyond_5mm':[x for x in raw if x['max_penetration_m']>.005]})
 for f in selected:
  assert global_nonhand[f]['max_penetration_m']<=.005
 for r in records:
  f=r['frame']
  if not r['selected_for_conditional_comparison']:continue
  B['candidate_valid'][f]=False;B['accepted'][f]=False;B['joint_image_supported'][f,arm]=False;B['joint_approx_uncertainty_rad'][f,arm]=np.nan;B['hand_max_penetration_m'][f,1]=r['max_hand_penetration_m'];B['parent_hand_image_fit_valid'][f,1]=r['hand_image_metrics']['legacy_weighted_training_rms_px']<4 and r['hand_image_metrics']['legacy_proxy12_heldout_px']<4.5;B['parent_hand_side_valid'][f,1]=False;H['image_fit_valid'][f,1]=B['parent_hand_image_fit_valid'][f,1];H['side_valid'][f,1]=False;H['valid'][f]=False;H['new_hand_candidate_selected'][f,1]=True;H['new_hand_candidate_accepted'][f,1]=False;H['refinement_selected_conditional'][f,1]=True;H['wrist_refinement_accepted'][f,1]=False
 H['observed'][:]=False;H['uncertain'][:]=True;H['actuator_observed'][:]=False;H['actuator_uncertain'][:]=True;H['actuator_std'][:]=np.nan;H['wrist_observed'][:]=False;H['wrist_uncertain'][:]=True;H['contact_geometry_uncertain']=~box['valid'].astype(bool);H['body_joint_positions_rad_refined']=q.copy();H['joint_positions']=coupling.expand_array(act)
 for key,nkey in [('all_wrist_joint_positions_rad','all_wrist_joint_names'),('wrist_joint_positions_rad','wrist_joint_names')]:H[key]=q[:,[kin.body_index[str(n)]for n in H[nkey]]]
 H['wrist_yaw_positions_rad']=q[:,[kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]];B.update(finger_actuator_positions=act.copy(),finger_joint_positions=H['joint_positions'].copy(),source_parent_body_path=np.asarray(str(bp.relative_to(ROOT))),source_parent_body_sha256=np.asarray(sha(bp)),source_body_sha256=np.asarray(sha(bp)),source_hand_sha256=np.asarray(sha(hp)),source_conditional_candidates_sha256=np.asarray(sha(cp)),candidate_is_metric_ground_truth=np.asarray(False));ob=out/'conditional_window_body.npz';oh=out/'conditional_window_hand.npz';np.savez_compressed(ob,**B);H.update(source_body_path=np.asarray(str(ob.relative_to(ROOT))),source_body_sha256=np.asarray(sha(ob)),body_sha256=np.asarray(sha(ob)),source_hand_sha256=np.asarray(sha(hp)),source_zero_target_body_sha256=np.asarray(sha(bp)),source_zero_target_hand_sha256=np.asarray(sha(hp)),source_conditional_candidates_sha256=np.asarray(sha(cp)),candidate_is_metric_ground_truth=np.asarray(False));np.savez_compressed(oh,**H);vel=np.abs(np.diff(q,axis=0))/np.diff(B['timestamps'])[:,None]
 report={'schema':'kinogen.v4.conditional_visibility.read_only_combination.1','inputs':{'zero_target_body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'zero_target_hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)},'saved_front_candidates':{'path':str(cp.relative_to(ROOT)),'sha256':sha(cp)},'source_candidate_report':{'path':str(reportpath.relative_to(ROOT)),'sha256':sha(reportpath)},'box':{'path':str(boxpath.relative_to(ROOT)),'sha256':sha(boxpath)}},'outputs':{'body':{'path':str(ob.relative_to(ROOT)),'sha256':sha(ob)},'hand':{'path':str(oh.relative_to(ROOT)),'sha256':sha(oh)}},'selected_frames':selected,'records':records,'actual_complete121_original_positive_velocity_passed':bool(np.all(vel<=B['joint_velocity_limits_rad_s']+1e-6)),'max_velocity_rad_s_per_joint':vel.max(axis=0).tolist(),'complete121_raw_nonhand_audit':global_nonhand,'existing_unchanged_frames_beyond_5mm':[x['frame']for x in global_nonhand if x['max_penetration_m']>.005],'only_selected_frames_modified':True,'source_zero_target_stage_unchanged':True,'optimization_performed':False,'interpolation_performed':False,'zero_target_or_source_non_deterioration_passed':False,'strict_valid':False,'all_3d_hand_observations_unknown':True,'warning':'Comparison candidate only: nonzero penetration is retained explicitly; original zero-target and source-non-deterioration are not reclassified. Original necessary5mm condition is separately audited. F70 nonhand source penetration remains unchanged for another owner.','elapsed_seconds':time.monotonic()-started};rp=ROOT/'reports/user-v4-conditional-window.json';rp.write_text(json.dumps(report,indent=2));print(json.dumps({'outputs':report['outputs'],'selected_frames':selected,'records':[{k:r[k]for k in ('frame','selected_for_conditional_comparison','max_hand_penetration_m','max_nonhand_penetration_m','comparison_rejections')}for r in records],'existing_nonhand_beyond5mm':report['existing_unchanged_frames_beyond_5mm'],'elapsed_seconds':report['elapsed_seconds'],'report_sha256':sha(rp)}),flush=True)
if __name__=='__main__':main()
