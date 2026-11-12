"""Freeze an explicit final body/hand pair, preserving stage parents and uncertainty."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-file', type=Path, required=True)
    parser.add_argument('--hand-file', type=Path, required=True)
    parser.add_argument('--ground-file', type=Path, default=Path('outputs/user_reconstruction_v3/ground_trajectory_retry.npz'))
    parser.add_argument('--prehand-file',type=Path,default=Path('outputs/user_reconstruction_v3/prehand_body.npz'))
    parser.add_argument('--source-report',type=Path,action='append',default=[])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from user_body_kinematics import G1Kinematics
    from user_rohand_coupling import ROHandCoupling
    parents = [p.resolve() if p.is_absolute() else (ROOT/p).resolve() for p in (args.body_file,args.hand_file)]
    if not all(p.is_relative_to(ROOT.resolve()) and p.is_file() for p in parents):
        raise ValueError('Explicit stage parents must exist inside the project')
    out = ROOT/'outputs/user_reconstruction_v4'
    body_path, hand_path = out/'body_trajectory.npz', out/'hand_trajectory.npz'
    if any(p in (body_path,hand_path) for p in parents) or body_path.exists() or hand_path.exists():
        raise FileExistsError('Canonical outputs must be new, with distinct frozen stage parents')
    body, hand = map(load, parents)
    ground_path=args.ground_file.resolve()if args.ground_file.is_absolute()else(ROOT/args.ground_file).resolve()
    prehand_path=args.prehand_file.resolve()if args.prehand_file.is_absolute()else(ROOT/args.prehand_file).resolve()
    reports=[p.resolve()if p.is_absolute()else(ROOT/p).resolve()for p in args.source_report]
    if not all(p.is_relative_to(ROOT.resolve())and p.is_file()for p in [ground_path,prehand_path]+reports):
        raise ValueError('Ground/report sources must exist inside project')
    old_hand_path = ROOT/'outputs/user_reconstruction_v2/hand_trajectory_contact.npz'
    if sha(old_hand_path) != '4f8f7c16df560d785c515fa8462c7de7243e93e1211c351ed6028b20a93ca9a1':
        raise RuntimeError('Frozen uncertainty parent changed')
    old_hand = load(old_hand_path)
    old_body_path=ROOT/'outputs/user_reconstruction_v2/body_trajectory_contact.npz'
    if sha(old_body_path)!='159cabb128e6f2d2f1f182e747c77f51ab6b25c79cfc22df9e49a1de46fb6f66':
        raise RuntimeError('Frozen body uncertainty parent changed')
    old_body=load(old_body_path)
    if str(hand['source_body_sha256']) != sha(parents[0]):
        raise RuntimeError('Hand stage is not bound to supplied body')
    kin = G1Kinematics(ROOT,scene_path=ROOT/'data/video_generation/candidate_scene/scene_rohand.json')
    print('Loaded unchanged model; assembling final explicit states',flush=True)
    coupling = ROHandCoupling(ROOT)
    bn=list(map(str,body['joint_names'])); hbn=list(map(str,hand['body_joint_names']))
    if bn != kin.body_joint_names or not np.array_equal(body['timestamps'],hand['timestamps']):
        raise RuntimeError('Joint order or timeline mismatch')
    q = hand['body_joint_positions_rad_refined'][:,[hbn.index(n) for n in bn]].copy()
    if q.shape != (121,29) or not np.isfinite(q).all():
        raise RuntimeError('Incomplete final body')
    if not np.allclose(body['timestamps'],np.arange(121)/24,atol=1e-10):
        raise RuntimeError('Final timeline differs from the 121 source frames')
    if not np.isfinite(body['base_xyz_m']).all()or not np.allclose(np.linalg.norm(body['base_quaternion_xyzw'],axis=1),1,atol=1e-8):
        raise RuntimeError('Invalid final floating base')
    if np.any(q<kin.body_limits[:,0]-1e-8)or np.any(q>kin.body_limits[:,1]+1e-8):
        raise RuntimeError('Final body violates authored position limits')
    speed=np.abs(np.diff(q,axis=0))/np.diff(body['timestamps'])[:,None]
    if np.any(speed>body['joint_velocity_limits_rad_s'][None]+1e-8):
        raise RuntimeError('Final body violates authored positive velocity limits')
    camera_path=ROOT/'outputs/user_reconstruction_v2/camera_refinement.npz'
    if sha(camera_path)!='813f64ca29c7199cb033c020e6d9b8ada1bc2c77313946baf53585fe50247b5a' or not np.array_equal(body['camera_P_video'],load(camera_path)['P_refined']):
        raise RuntimeError('Frozen cameras changed')
    other = [i for i,n in enumerate(bn) if '_wrist_' not in n]
    if not np.array_equal(q[:,other],body['joint_positions_rad'][:,other]):
        raise RuntimeError('Hand stage modified a nonwrist joint')
    changed=q!=body['joint_positions_rad']
    body['joint_positions_rad']=q
    body['joint_image_supported'][changed]=False
    body['joint_approx_uncertainty_rad'][changed]=np.nan
    body['joint_final_hand_stage_changed']=changed
    # A good 2D contour is not new evidence that a 3D actuator is identifiable.
    old_act_names=list(map(str,old_hand['actuator_names'])); act_names=list(map(str,hand['actuator_names']))
    old_joint_names=list(map(str,old_hand['joint_names'])); joint_names=list(map(str,hand['joint_names']))
    same_act=hand['actuator_positions']==old_hand['actuator_positions'][:,[old_act_names.index(n) for n in act_names]]
    same_joint=hand['joint_positions']==old_hand['joint_positions'][:,[old_joint_names.index(n) for n in joint_names]]
    hand['actuator_observed'] &= old_hand['actuator_observed'][:,[old_act_names.index(n) for n in act_names]] & same_act
    hand['observed'] &= old_hand['observed'][:,[old_joint_names.index(n) for n in joint_names]] & same_joint
    old_bn=list(map(str,old_hand['body_joint_names']))
    for side in ('left','right'):
        indices=[i for i,n in enumerate(bn)if n.startswith(side+'_wrist_')]
        moved=(q[:,indices]!=old_hand['body_joint_positions_rad_refined'][:,[old_bn.index(bn[i])for i in indices]]).any(axis=1)
        for k,n in enumerate(act_names):
            if n.startswith(side+'_'):hand['actuator_observed'][moved,k]=False
        for k,n in enumerate(joint_names):
            if n.startswith(side+'_'):hand['observed'][moved,k]=False
    hand['actuator_uncertain']=~hand['actuator_observed']
    hand['uncertain']=~hand['observed']; hand['prior_only']=~hand['observed']
    hand['actuator_std'][~hand['actuator_observed']]=np.nan
    # Strictly observed grasp states remain unverified even when image residuals pass.
    hand['side_valid'] &= old_hand['side_valid']; hand['valid']=hand['side_valid'].all(axis=1)
    if 'wrist_observed' in hand:
        hand['wrist_observed'][:]=False; hand['wrist_uncertain'][:]=True
    controls=hand['actuator_positions'][:,[act_names.index(n)for n in coupling.actuator_names]]
    expanded=coupling.expand_array(controls)
    if not np.allclose(hand['joint_positions'][:,[joint_names.index(n)for n in coupling.joint_names]],expanded,atol=1e-10,rtol=0):
        raise RuntimeError('Final hand state violates official linkage')
    finger_q=hand['joint_positions'][:,[joint_names.index(n)for n in kin.finger_joint_names]]
    transforms=np.stack([kin.forward(body['base_xyz_m'][f],body['base_quaternion_xyzw'][f],q[f],dict(zip(kin.finger_joint_names,finger_q[f])))for f in range(121)])
    old_q=old_hand['body_joint_positions_rad_refined'][:,[old_bn.index(n)for n in bn]]
    changed_from_v2=q!=old_q
    body['joint_v4_changed_from_v2']=changed_from_v2
    body['joint_image_supported'][changed_from_v2]=False
    body['joint_approx_uncertainty_rad'][changed_from_v2]=np.nan
    old_tf=np.stack([kin.forward(old_body['base_xyz_m'][f],old_body['base_quaternion_xyzw'][f],old_q[f])for f in range(121)])
    for side in ('left','right'):
        wrist=kin.link_index[side+'_wrist_yaw_link']
        moved=np.max(np.abs(transforms[:,wrist]-old_tf[:,wrist]),axis=(1,2))>1e-10
        hand['side_valid'][moved,0 if side=='left'else 1]=False
        for k,n in enumerate(act_names):
            if n.startswith(side+'_'):hand['actuator_observed'][moved,k]=False
        for k,n in enumerate(joint_names):
            if n.startswith(side+'_'):hand['observed'][moved,k]=False
    hand['actuator_uncertain']=~hand['actuator_observed'];hand['uncertain']=~hand['observed'];hand['prior_only']=~hand['observed']
    hand['valid']=hand['side_valid'].all(axis=1)
    hand['actuator_std'][~hand['actuator_observed']]=np.nan
    body['link_world_transforms']=transforms; body['link_names']=np.asarray(kin.link_names)
    for key in ('actuator_positions','actuator_names','actuator_units','joint_positions','joint_names','joint_units'):
        body['finger_'+key]=hand[key].copy()
    body['finger_joint_positions_included']=np.asarray(True)
    body['parent_hand_image_fit_valid']=hand['image_fit_valid'].copy();body['parent_hand_side_valid']=hand['side_valid'].copy()
    box_path=ROOT/'outputs/user_reconstruction_v2/box_trajectory.npz'; box=load(box_path)
    if sha(box_path)!='a9c5b1f274997f86382f24ba09a67abe84ae4231370888f05f254a2d49c415fa':
        raise RuntimeError('Frozen box changed')
    from scipy.spatial.transform import Rotation
    mins=np.full((121,2),np.inf); fractions=np.zeros((121,2)); all_floor=np.full(121,np.inf)
    raw_visuals=[]
    for visual in kin.visuals:
        vertices=np.unique(visual['vertices_local'],axis=0)
        hand_part='_hand_ro_'in visual['link_name']
        centers=visual['vertices_local'][visual['faces']].mean(axis=1)if hand_part else None
        side=0 if visual['link_name'].startswith('left_')else 1
        raw_visuals.append((visual['link_index'],vertices,centers,side))
    print('Cached exact raw vertices and all hand triangle centers',flush=True)
    for f in range(121):
        side_sdf=[[],[]]; rotation=Rotation.from_quat(box['quaternion_xyzw'][f]).as_matrix()
        for link_index,v,centers,side in raw_visuals:
            T=transforms[f,link_index]
            world=v@T[:3,:3].T+T[:3,3];all_floor[f]=min(all_floor[f],float(world[:,2].min()))
            if centers is not None:
                world=np.vstack((world,centers@T[:3,:3].T+T[:3,3]))
                d=np.abs((world-box['center_xyz_m'][f])@rotation)-box['box_size_xyz_m']/2
                sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
                side_sdf[side].append(sdf)
        for side in range(2):
            distances=np.concatenate(side_sdf[side]);mins[f,side]=distances.min();fractions[f,side]=np.mean(distances<-.005)
        if f%30==0:print(f'Final exact raw geometry {f}/120',flush=True)
    body['hand_max_penetration_m']=np.maximum(-mins,0)
    body['raw_all_robot_min_z_m']=all_floor
    body['candidate_valid'] &= (all_floor>=-.005)
    hand['hand_surface_min_signed_distance_m']=mins;hand['hand_surface_fraction_beyond_5mm']=fractions
    hand['contact_nonpenetration_sample_valid']=(mins>=-.005)&box['valid'][:,None]
    body['source_parent_body_sha256']=np.asarray(sha(parents[0]))
    body['source_parent_body_path']=np.asarray(parents[0].relative_to(ROOT).as_posix())
    body['source_prehand_body_sha256']=np.asarray(sha(prehand_path))
    body['source_ground_sha256']=np.asarray(sha(ground_path))
    body['source_ground_path']=np.asarray(ground_path.relative_to(ROOT).as_posix())
    body['source_prehand_sha256']=np.asarray(sha(prehand_path))
    body['source_final_hand_stage_sha256']=np.asarray(sha(parents[1]))
    body['source_stage_hand_sha256']=np.asarray(sha(parents[1]))
    body['source_final_hand_stage_path']=np.asarray(parents[1].relative_to(ROOT).as_posix())
    hand['source_prehand_body_sha256']=np.asarray(sha(prehand_path));hand['source_hand_stage_sha256']=np.asarray(sha(parents[1]))
    hand['source_prehand_body_path']=np.asarray(prehand_path.relative_to(ROOT).as_posix())
    hand['source_hand_stage_path']=np.asarray(parents[1].relative_to(ROOT).as_posix())
    hand['source_stage_hand_sha256']=np.asarray(sha(parents[1]))
    hand['stage_source_body_sha256']=hand['source_body_sha256'].copy()
    hand['stage_source_hand_sha256']=hand.get('source_hand_sha256',np.asarray('')).copy()
    # Body references the immutable stage hand, avoiding a canonical-body/hand hash cycle.
    body['source_hand_sha256']=np.asarray(sha(parents[1]))
    np.savez_compressed(body_path,**body)
    hand['body_sha256']=hand['source_body_sha256']=np.asarray(sha(body_path))
    hand['body_joint_names']=np.asarray(bn);hand['body_joint_positions_rad_refined']=q
    hand['stage_wrist_refinement_accepted']=hand['wrist_refinement_accepted'].copy()
    hand['wrist_refinement_accepted'][:]=False
    hand['canonical_wrist_refinement_absorbed_into_body']=np.asarray(True)
    np.savez_compressed(hand_path,**hand)
    inputs=parents+[ground_path,prehand_path,old_hand_path,old_body_path,box_path,camera_path,ROOT/'scripts/canonicalize_user_reconstruction_v4.py']+reports
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'canonical_pair_frozen',
            'inputs':{p.relative_to(ROOT).as_posix():sha(p)for p in inputs},
            'outputs':{p.relative_to(ROOT).as_posix():sha(p)for p in (body_path,hand_path)},
            'body_wrist_entries_absorbed':int(changed.sum()),'all_89_link_transforms_recomputed':True,
            'uncertainty_policy':'No promotion of 3D support from a 2D fit. Changed body support=false/std=NaN; hand support intersects frozen V2 support and exact unchanged parameter; wrist support unknown.',
            'body_candidate_valid_frames':int(body['candidate_valid'].sum()),'hand_image_fit_valid_counts':hand['image_fit_valid'].sum(0).tolist(),
            'ground_failed_frames_gt5mm':np.flatnonzero(all_floor<-.005).tolist(),
            'hand_failed_frames_gt5mm':np.flatnonzero(mins.min(axis=1)<-.005).tolist(),
            'max_hand_sample_penetration_m':float(max(0,-mins.min())),
            'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'gpu_used':False,
            'limitations':['Conditional authored floor/camera/box and approximate RGB observations remain.','Raw mesh vertices and triangle centers are sampled checks, not exact continuous collision or grasp proof.','Original stage image residuals remain bound to immutable parents; canonical geometry was recomputed.']}
    report_path=ROOT/'reports/user-v4-canonical-provenance.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
