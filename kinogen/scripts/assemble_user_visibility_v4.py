"""Compose disjoint, explicitly declared visibility stages without interpolation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',nargs=3,action='append',required=True,metavar=('BODY','HAND','FRAMES'))
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from user_body_kinematics import G1Kinematics
    from user_rohand_coupling import ROHandCoupling
    def path(value):
        p=Path(value);p=p.resolve()if p.is_absolute()else(ROOT/p).resolve()
        if not p.is_relative_to(ROOT.resolve())or not p.is_file():raise ValueError('Stage outside project or missing')
        return p
    def load(p):
        with np.load(p,allow_pickle=False)as d:return {k:d[k].copy()for k in d.files}
    source_body=path('outputs/user_reconstruction_v3/body_trajectory.npz');source_hand=path('outputs/user_reconstruction_v3/hand_trajectory.npz')
    if sha(source_body)!='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'or sha(source_hand)!='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1':raise RuntimeError('Frozen V3 source changed')
    body,hand=load(source_body),load(source_hand)
    kin=G1Kinematics(ROOT,scene_path=ROOT/'data/video_generation/candidate_scene/scene_rohand.json');coupling=ROHandCoupling(ROOT)
    bn=kin.body_joint_names; q=body['joint_positions_rad'].copy();act=hand['actuator_positions'].copy();old_q=q.copy();old_act=act.copy()
    allowed_arm=[i for i,n in enumerate(bn)if n.startswith('right_')and any(part in n for part in ('shoulder_','elbow_','wrist_'))]
    protected=[i for i in range(29)if i not in allowed_arm]
    rows=[];used=set();inputs={source_body.relative_to(ROOT).as_posix():sha(source_body),source_hand.relative_to(ROOT).as_posix():sha(source_hand)}
    for body_name,hand_name,frame_text in args.stage:
        bp,hp=path(body_name),path(hand_name);b,h=load(bp),load(hp)
        frames=[int(s)for s in frame_text.split(',')]
        if not frames or any(f<0 or f>=121 or f in used for f in frames)or len(set(frames))!=len(frames):raise ValueError('Stages must declare disjoint source frame IDs')
        used.update(frames)
        if str(h['source_body_sha256'])!=sha(bp):raise RuntimeError('Stage hand has stale body binding')
        if list(map(str,b['joint_names']))!=bn or list(map(str,h['actuator_names']))!=coupling.actuator_names:raise RuntimeError('Stage joint/control order differs')
        actual=h['body_joint_positions_rad_refined'][:,[list(map(str,h['body_joint_names'])).index(n)for n in bn]]
        if not np.array_equal(actual,b['joint_positions_rad']):raise RuntimeError('Stage body has not absorbed its actual angles')
        if not np.array_equal(b['base_xyz_m'],body['base_xyz_m'])or not np.array_equal(b['base_quaternion_xyzw'],body['base_quaternion_xyzw']):raise RuntimeError('Visibility stage moved base')
        if not np.array_equal(actual[:,protected],body['joint_positions_rad'][:,protected]):raise RuntimeError('Visibility stage changed protected body chain')
        if not np.array_equal(h['actuator_positions'][:,:6],hand['actuator_positions'][:,:6]):raise RuntimeError('Right-hand stage changed left controls')
        if not np.array_equal(b['timestamps'],body['timestamps'])or not np.array_equal(h['timestamps'],hand['timestamps']):raise RuntimeError('Stage timeline differs')
        q[frames]=actual[frames];act[frames]=h['actuator_positions'][frames]
        for field in ('candidate_valid','ground_image_valid','ground_nonpenetration_valid'):
            if field in b:body[field][frames]=b[field][frames]
        for field in ('image_fit_valid','side_valid','valid'):
            if field in h:hand[field][frames]=h[field][frames]
        inputs[bp.relative_to(ROOT).as_posix()]=sha(bp);inputs[hp.relative_to(ROOT).as_posix()]=sha(hp)
        rows.append({'body':bp.relative_to(ROOT).as_posix(),'hand':hp.relative_to(ROOT).as_posix(),'frames_copied_exactly':frames})
    velocity=body['joint_velocity_limits_rad_s'];speed=np.abs(np.diff(q,axis=0))/np.diff(body['timestamps'])[:,None]
    if np.any(speed>velocity+1e-8)or np.any(q<kin.body_limits[:,0]-1e-8)or np.any(q>kin.body_limits[:,1]+1e-8):raise RuntimeError('Composed stages violate original G1 q/velocity')
    expanded=coupling.expand_array(act);body['joint_positions_rad']=q
    body['finger_actuator_positions']=act;body['finger_actuator_names']=np.asarray(coupling.actuator_names);body['finger_actuator_units']=np.asarray(coupling.actuator_units)
    body['finger_joint_positions']=expanded;body['finger_joint_names']=np.asarray(coupling.joint_names);body['finger_joint_units']=np.asarray(coupling.joint_units)
    body['link_world_transforms']=np.stack([kin.forward(body['base_xyz_m'][f],body['base_quaternion_xyzw'][f],q[f],dict(zip(coupling.joint_names,expanded[f])))for f in range(121)])
    changes=q!=old_q;body['joint_image_supported'][changes]=False;body['joint_approx_uncertainty_rad'][changes]=np.nan
    for key in list(hand):
        if any(token in key for token in ('contact_tip','hand_surface','contact_nonpenetration','wrist_keypoint_error','wrist_origin_world_shift','dense_collision','heldout_edge','legacy_observation','v3_proxy16')):del hand[key]
    hand.update(actuator_positions=act,joint_positions=expanded,body_joint_positions_rad_refined=q,
                observed=np.zeros((121,50),bool),uncertain=np.ones((121,50),bool),prior_only=np.ones((121,50),bool),
                actuator_observed=np.zeros((121,12),bool),actuator_uncertain=np.ones((121,12),bool),actuator_std=np.full((121,12),np.nan),
                wrist_observed=np.zeros((121,6),bool),wrist_uncertain=np.ones((121,6),bool))
    hand['side_valid'][:]=False;hand['valid'][:]=False
    wrist_names=[s+'_wrist_'+a+'_joint'for s in ('left','right')for a in ('roll','pitch','yaw')]
    for field in ('wrist_joint_positions_rad','all_wrist_joint_positions_rad'):hand[field]=q[:,[bn.index(n)for n in wrist_names]]
    hand['wrist_yaw_positions_rad']=q[:,[bn.index(s+'_wrist_yaw_joint')for s in ('left','right')]]
    out=ROOT/'outputs/user_reconstruction_v4';out.mkdir(exist_ok=True)
    bp,hp=out/'visibility_body_stage.npz',out/'visibility_hand_stage.npz'
    if bp.exists()or hp.exists():raise FileExistsError('Composed stages are immutable; choose a fresh stage')
    body['source_body_sha256']=np.asarray(sha(source_body));body['source_hand_sha256']=np.asarray(sha(source_hand));np.savez_compressed(bp,**body)
    hand['source_body_sha256']=hand['body_sha256']=np.asarray(sha(bp));hand['source_hand_sha256']=np.asarray(sha(source_hand));np.savez_compressed(hp,**hand)
    rp=ROOT/'reports/user-v4-composed-stages.json';inputs['scripts/assemble_user_visibility_v4.py']=sha(Path(__file__))
    report={'status':'explicit_disjoint_stage_composition','inputs':inputs,'stages':rows,'declared_stage_frame_ids':sorted(used),
            'actual_changed_frame_ids':np.flatnonzero(changes.any(axis=1)|(act!=old_act).any(axis=1)).tolist(),
            'output':{'path':bp.relative_to(ROOT).as_posix(),'sha256':sha(bp)},'hand_output':{'path':hp.relative_to(ROOT).as_posix(),'sha256':sha(hp)},
            'original_joint_position_and_positive_velocity_bounds_passed':True,'body_candidate_valid_frames':int(body['candidate_valid'].sum()),
            'source_pixels_or_poses_interpolated':False,'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False}
    rp.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
