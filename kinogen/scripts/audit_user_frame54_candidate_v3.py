#!/usr/bin/env python3
"""Audit a unified full-grid F54 display candidate without accepting it."""
from pathlib import Path
import argparse
import json
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,frame54_image_comparison,CAMERA_SHA,BOX_SHA
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate',type=Path,default=Path('outputs/user_reconstruction_v3/best_near_surface_unaccepted.npz'));ap.add_argument('--output',type=Path,default=Path('reports/user-v3-frame54-independent.json'));ap.add_argument('--hand-training-domain',choices=['sample72','dense'],default='sample72');ap.add_argument('--use-best-trial',action='store_true');args=ap.parse_args()
    p=(ROOT/args.candidate).resolve();out=(ROOT/args.output).resolve();require(p.is_relative_to(ROOT)and out.is_relative_to(ROOT),'Outside project')
    n=load(p);bp=V2/'body_trajectory_contact.npz';hp=V2/'hand_trajectory_contact.npz';b=load(bp);h=load(hp);box=load(V2/'box_trajectory.npz')
    require(str(n['source_body_sha256'])==sha(bp)and str(n['source_hand_sha256'])==sha(hp),'Candidate source changed')
    require(str(n['source_camera_sha256'])==CAMERA_SHA and str(n['source_box_sha256'])==BOX_SHA,'Frozen geometry differs')
    if args.use_best_trial:
        from user_rohand_coupling import ROHandCoupling
        n['body_joint_positions_rad_refined'][54]=n['best_trial_body_joint_positions_rad']
        n['actuator_positions'][54]=n['best_trial_actuator_positions']
        n['joint_positions']=ROHandCoupling(ROOT).expand_array(n['actuator_positions'])
    names=list(map(str,b['joint_names']));q=n['body_joint_positions_rad_refined'][:,[list(map(str,n['body_joint_names'])).index(x)for x in names]]
    sourceq=h['body_joint_positions_rad_refined'][:,[list(map(str,h['body_joint_names'])).index(x)for x in names]]
    require(q.shape==(121,29)and np.array_equal(n['timestamps'],b['timestamps']),'Candidate not full grid')
    allowed=np.zeros((121,29),bool)
    for k,name in enumerate(names):allowed[54,k]=name.startswith('right_')and any(t in name for t in ('shoulder_','elbow_','wrist_'))
    require(allowed.sum()==7 and np.array_equal(q[~allowed],sourceq[~allowed]),'Candidate changed outsideF54 rightarm7')
    for key in ('base_xyz_m','base_quaternion_xyzw'):require(np.array_equal(n[key],b[key]),'Candidate moved base')
    require(np.array_equal(n['actuator_names'],h['actuator_names'])and np.array_equal(n['joint_names'],h['joint_names']),'Candidate changed hand ordering')
    ac=np.zeros((121,12),bool);ac[54,6:]=True
    require(np.array_equal(n['actuator_positions'][~ac],h['actuator_positions'][~ac]),'Candidate changed other actuators')
    body={k:v.copy()for k,v in b.items()};hand={k:v.copy()for k,v in h.items()}
    body['joint_positions_rad']=q.copy();hand['body_joint_positions_rad_refined']=q.copy();hand['body_joint_names']=np.asarray(names)
    for key in ('actuator_positions','joint_positions'):hand[key]=n[key].copy()
    # No promotion: a diagnostic candidate does not create newly observed joints.
    body['joint_image_supported'][54,allowed[54]]=False;body['joint_approx_uncertainty_rad'][54,allowed[54]]=np.nan
    for key in ('observed','actuator_observed'):hand[key][54,25 if key=='observed'else 6:]=False
    hand['uncertain']=~hand['observed'];hand['actuator_uncertain']=~hand['actuator_observed'];hand['image_fit_valid'][54,1]=False;hand['side_valid'][54,1]=False;hand['valid']=hand['side_valid'].all(axis=1)
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());model=XMLModel(ROOT/scene['robot']['urdf'])
    before=model.audit(b,h,box);after=model.audit(body,hand,box);image=frame54_image_comparison(model,b,h,body,hand,args.hand_training_domain)
    report={'status':'conditional_display_candidate_independently_audited','candidate':record(p),'uses_explicit_best_trial_fields':args.use_best_trial,'trial_extraction_policy':'If requested, only supplied actual best_trial_body29 and best_trial_actuator12 replaceF54 in memory; joint50 is deterministically expanded by official coupling. No pose search, interpolation or candidate file mutation.','parents':[record(bp),record(hp)],'only_frame54_rightarm7_and_right_actuators6_changed':True,'all_other_actual_refined_body_and_actuator_states_exact':True,'base_legs_waist_unchanged':True,'fixed_foot_material_projections_identical_by_kinematic_invariance':True,'baseline':before,'candidate_geometry':after,'frame54_fixed_rgb_image':image,'strict_hand_image_gate_passed':image['candidate']['hand_image_necessary_gates_passed'],'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'source_flags_promoted':False,'candidate_acceptance_claimed':False,'floor_still_original_v2':True,'note':'This is a standalone diagnostic view of the supplied unified state; no canonical body/hand files or frozen summaries were changed. It must be reevaluated after a ground/body merge.'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(out),'beforeF54':before['per_frame'][54],'afterF54':after['per_frame'][54],'image':image['candidate'],'positive_body_velocity_passed':after['all_positive_original_body_velocity_limits_passed']},indent=2))

if __name__=='__main__':main()
