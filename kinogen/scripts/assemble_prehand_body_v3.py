"""Bind a ground-compatible actual F54 joint solution; never sum pose deltas."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ground-body',type=Path,required=True);ap.add_argument('--frame54-candidate',type=Path,required=True);args=ap.parse_args()
 import numpy as np,xml.etree.ElementTree as ET
 from user_body_kinematics import G1Kinematics
 from user_rohand_coupling import ROHandCoupling
 for key in ('ground_body','frame54_candidate'):
  p=getattr(args,key);setattr(args,key,p if p.is_absolute()else ROOT/p)
 def load(p):
  with np.load(p)as n:return {k:n[k]for k in n.files}
 g=load(args.ground_body);h=load(args.frame54_candidate);original_hand=ROOT/'outputs/user_reconstruction_v2/hand_trajectory_contact.npz';source_hand=load(original_hand);kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');coupling=ROHandCoupling(ROOT)
 if str(h['source_body_sha256'])!=sha(args.ground_body):raise ValueError('F54 candidate was not actually solved against this exact ground parent')
 if not np.array_equal(h['base_xyz_m'],g['base_xyz_m'])or not np.array_equal(h['base_quaternion_xyzw'],g['base_quaternion_xyzw']):raise ValueError('Refuse to mix world/base frames')
 q=h['body_joint_positions_rad_refined'];act=h['actuator_positions'];arm=[kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')];allowed=np.zeros((121,29),bool);allowed[54,arm]=True
 if not np.array_equal(q[~allowed],g['joint_positions_rad'][~allowed]):raise ValueError('F54 candidate changed an unowned body entry')
 allowed_a=np.zeros((121,12),bool);allowed_a[54,6:]=True
 if not np.array_equal(act[~allowed_a],source_hand['actuator_positions'][~allowed_a]):raise ValueError('F54 candidate changed other source hand controls')
 xml=ET.parse(kin.urdf_path).getroot();vs={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in kin.body_index};velocity=np.asarray([vs[n]for n in kin.body_joint_names]);speeds=abs(np.diff(q,axis=0)/np.diff(g['timestamps'])[:,None]);assert np.all(speeds<=velocity[None]+1e-6);assert np.all(q>=kin.body_limits[:,0]-1e-9)and np.all(q<=kin.body_limits[:,1]+1e-9)
 expanded=coupling.expand_array(act);tf=np.stack([kin.forward(g['base_xyz_m'][f],g['base_quaternion_xyzw'][f],q[f],coupling.expand_one(act[f]))for f in range(121)])
 payload=dict(g);payload['joint_positions_rad']=q.copy();payload['link_world_transforms']=tf;payload['link_names']=np.asarray(kin.link_names);payload['finger_actuator_positions']=act.copy();payload['finger_actuator_names']=source_hand['actuator_names'];payload['finger_actuator_units']=source_hand['actuator_units'];payload['finger_joint_positions']=expanded;payload['finger_joint_names']=source_hand['joint_names'];payload['finger_joint_units']=source_hand['joint_units'];payload['source_ground_sha256']=np.asarray(sha(args.ground_body));payload['source_frame54_sha256']=np.asarray(sha(args.frame54_candidate));payload['source_hand_sha256']=np.asarray(sha(args.frame54_candidate))
 if 'joint_image_supported'in payload:payload['joint_image_supported'][54,arm]=False
 if 'joint_approx_uncertainty_rad'in payload:payload['joint_approx_uncertainty_rad'][54,arm]=np.nan
 out=ROOT/'outputs/user_reconstruction_v3';bodypath=out/'prehand_body.npz';np.savez_compressed(bodypath,**payload)
 hp=dict(source_hand);hp.update(actuator_positions=act.copy(),joint_positions=expanded,body_joint_positions_rad_refined=q.copy(),base_xyz_m=g['base_xyz_m'],base_quaternion_xyzw=g['base_quaternion_xyzw'],source_body_sha256=np.asarray(sha(bodypath)),body_sha256=np.asarray(sha(bodypath)),source_ground_sha256=np.asarray(sha(args.ground_body)),source_frame54_sha256=np.asarray(sha(args.frame54_candidate)))
 for key in ('observed','actuator_observed'):
  if key in hp:hp[key][54,25 if key=='observed'else 6:]=False
 hp['uncertain']=~hp['observed'];hp['actuator_uncertain']=~hp['actuator_observed'];hp['image_fit_valid'][54,1]=bool(h['strict_acceptance_passed'])if 'strict_acceptance_passed'in h else False;hp['side_valid'][54,1]=False;hp['valid']=hp['side_valid'].all(1)
 wrist_names=[s+'_wrist_'+a+'_joint'for s in ('left','right')for a in ('roll','pitch','yaw')];wq=q[:,[kin.body_index[n]for n in wrist_names]]
 for key in ('wrist_joint_positions_rad','all_wrist_joint_positions_rad'):hp[key]=wq.copy()
 hp['wrist_yaw_positions_rad']=wq[:,[2,5]];hp['wrist_observed']=np.zeros((121,6),bool);hp['wrist_uncertain']=np.ones((121,6),bool)
 handpath=out/'prehand_hand.npz';np.savez_compressed(handpath,**hp)
 report={'status':'ground_and_actual_joint_solution_bound_without_delta_addition','ground':{'path':str(args.ground_body),'sha256':sha(args.ground_body)},'actual_f54_solution':{'path':str(args.frame54_candidate),'sha256':sha(args.frame54_candidate)},'source_original_hand_sha256':sha(original_hand),'all121_joint_positions_and_original_positive_velocities_valid':True,'base_and_legs_from_ground_exact':True,'other_frames_exact':True,'changed54_support_cleared_unknown':True,'all89_link_fk_recomputed':True,'output_body':{'path':str(bodypath),'sha256':sha(bodypath)},'output_hand':{'path':str(handpath),'sha256':sha(handpath)},'note':'F54 arm was actually optimized in this exact ground world frame. No old-world q delta addition or frame interpolation.'};(ROOT/'reports/user-prehand-binding-v3.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
