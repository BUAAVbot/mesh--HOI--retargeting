from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p)as z:return {k:z[k]for k in z.files}
def clean(x):
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,list):return [clean(v)for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float)and not np.isfinite(x):return None
 return x
file=ROOT/'outputs/user_reconstruction_v3/ground_trajectory_retry.npz';reportfile=ROOT/'reports/user-ground-optimization-retry.json'
d=load(file);r=json.loads(reportfile.read_text());h=load(ROOT/'outputs/user_reconstruction_v2/hand_trajectory_contact.npz');c=load(ROOT/'outputs/user_reconstruction_v2/body_trajectory_contact.npz');waist=load(ROOT/'outputs/user_reconstruction_v3/ground_trajectory_retry_waist.npz')
assert sha(file)==r['output']['sha256']
assert np.array_equal(d['finger_actuator_positions'],h['actuator_positions'])
assert np.array_equal(d['finger_joint_positions'],h['joint_positions'])
assert np.array_equal(d['timestamps'],c['timestamps'])
assert np.array_equal(d['joint_positions_rad'],waist['joint_positions_rad'])
assert np.array_equal(d['base_xyz_m'],waist['base_xyz_m'])
assert np.allclose(d['base_quaternion_xyzw'],waist['base_quaternion_xyzw'],atol=1e-14,rtol=0)
q=d['joint_positions_rad'];speed=np.abs(np.diff(q,axis=0))/np.diff(d['timestamps'])[:,None]
assert (speed<=d['joint_velocity_limits_rad_s']+1e-6).all()
assert ((q>=d['joint_limits_rad'][:,0]-1e-8)&(q<=d['joint_limits_rad'][:,1]+1e-8)).all()
changed=d['joint_ground_stage_changed'];assert not d['joint_image_supported'][changed].any();assert np.isnan(d['joint_approx_uncertainty_rad'][changed]).all()
accepted=d['accepted'];images=[im for rec in r['records']for im in rec['after_images']];feet=[f for im in images for f in im['local_foot_contour']]
summary={
 'status':'frozen_ground_candidate_with_one_explicit_failure','canonical':{'path':str(file.relative_to(ROOT)),'sha256':sha(file),'report':str(reportfile.relative_to(ROOT)),'report_sha256':sha(reportfile)},
 'parents':r['inputs'],'frames':len(q),'accepted_frames':int(accepted.sum()),'rejected_frames':np.flatnonzero(~accepted).tolist(),
 'raw_full_foot_min_z_m_all_frames':float(d['raw_foot_min_z_m'].min()),'raw_full_foot_min_z_m_accepted_only':float(d['raw_foot_min_z_m'][accepted].min()),
 'raw_full_foot_min_z_m_by_frame':d['raw_foot_min_z_m'],'sole_only_max_z_m_accepted':max(f['sole_max_z_m']for rec in r['records']if rec['accepted']for f in rec['after_geometry']['feet']),
 'local_foot_heldout_median_px_max':max(f['median_px']for f in feet),'local_foot_heldout_p90_px_max':max(f['p90_px']for f in feet),'local_foot_heldout_p90_px_max_accepted_only':max(f['p90_px']for rec in r['records']if rec['accepted']for im in rec['after_images']for f in im['local_foot_contour']),
 'tap_physical_id_heldout_median_px_max':max(i['tap_heldout_median_px']for i in images),'tap_physical_id_heldout_p90_px_max':max(i['tap_heldout_p90_px']for i in images),
 'source_base_xyz_max_change_m':float(np.linalg.norm(d['base_xyz_m']-c['base_xyz_m'],axis=1).max()),'base_linear_speed_max_m_s':float((np.linalg.norm(np.diff(d['base_xyz_m'],axis=0),axis=1)/np.diff(d['timestamps'])).max()),'base_linear_speed_max_interval':int(np.argmax(np.linalg.norm(np.diff(d['base_xyz_m'],axis=0),axis=1)/np.diff(d['timestamps']))),'base_hardware_velocity_validated':False,
 'body_angular_velocity_max_rad_s':dict(zip(d['joint_names'].tolist(),speed.max(0).tolist())),
 'changed_joint_values_count':int(changed.sum()),'changed_joints_image_support_false_and_uncertainty_unknown':True,
 'all_original_body_limits_and_velocity_satisfied':True,'fixed_hcontact_12_controls_and_50_mixed_states_exact':True,'saved_89_link_tf_uses_declared_fixed_parent_hand':True,
 '35_variable_trial_rejected_and_no_state_adopted':True,'trial_file':'outputs/user_reconstruction_v3/ground_trajectory_retry_waist.npz','trial_sha256':sha(ROOT/'outputs/user_reconstruction_v3/ground_trajectory_retry_waist.npz'),
 'floor_moved':False,'floor_z_m':0.,'strict_metric_or_physical_reconstruction_valid':False,'conditional_camera_and_box_geometry':True,
 'known_remaining_failures':{'frame26':'Ground nonpenetration and unchanged real RGB wrist/hand evidence were not jointly achieved by finite32/35variable searches. Original pose remains invalid, no interpolation.','frame54':'Existing frozenHcontact right hand penetration remains; sibling hand optimization must recompute full FK/collision on this base/body candidate.','left_hand_contact':'No assertion of bilateral contact; prior left hand can remain centimetres from box.'},
 'one_future_frame26_joint_trial_not_executed':{'required_parents':['frozen updated hand trajectory and real gray hand observations train/heldout','this canonical base/body and final frame25/27 body/controls','original camera813f/boxa9c and original sourcevideo/input/SAM/TAP'], 'variables':'base6+body29 includingwaist+12coupledhandcontrols=47, onlyframe26; neighbours fixed','budget':'at most2initializations x4outer x160iterations; CPUapproximately3minutes; no further rounds','constraints':'original q/bodyvelocity; vendor/source control limits and only actually known hand velocity bounds; all original ground/body/foot/TAP/wrist criteria; dense original handmesh SDF; new gray hand train/heldout split no leakage','policy':'If still no compatible solution, keep frame26invalid. Finger shape can change only through officialcoupling; no freehand transforms/mount/scale/box/floor changes.'}
}
p=ROOT/'reports/user-ground-final-summary.json';p.write_text(json.dumps(clean(summary),indent=2)+'\n');print(json.dumps(clean({k:summary[k]for k in ['canonical','frames','accepted_frames','rejected_frames','raw_full_foot_min_z_m_accepted_only','local_foot_heldout_median_px_max','local_foot_heldout_p90_px_max','tap_physical_id_heldout_p90_px_max','source_base_xyz_max_change_m','base_linear_speed_max_m_s']}),indent=2));print('summary_sha',sha(p))
