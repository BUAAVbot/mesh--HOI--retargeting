"""Correct shoe canonical axes and infer continuous wear/removal references.

The original source arrays and full 398-frame robot IK are left intact.
Contact-consistent shoe targets are saved alongside the raw source, never as
new observations. Inferred welds are a task assumption for physical replay.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v2_full.npz')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v3_pca_shoe.npz')
    p.add_argument('--release-time',type=float,default=10.85)
    p.add_argument('--wear-forward',type=float,default=.004)
    p.add_argument('--wear-sole-z',type=float,default=-.074)
    p.add_argument('--floor-clearance',type=float,default=.001)
    p.add_argument('--asset',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_wear_v2.xml')
    p.add_argument('--proxy',default='outputs/rgb_near_pipeline/assets/vita_single_proxy.xml')
    p.add_argument('--wear-local',type=float,nargs=3)
    a=p.parse_args()
    generator_bytes=Path(__file__).read_bytes();generator_sha256=hashlib.sha256(generator_bytes).hexdigest()
    import numpy as np
    import mujoco as mj
    root=a.root
    inp=root/a.input;out=root/a.output
    z=np.load(inp,allow_pickle=False);payload={k:z[k]for k in z.files}
    asset=root/a.asset
    sm=mj.MjModel.from_xml_path(str(asset));sd=mj.MjData(sm)
    proxy=root/a.proxy
    m=mj.MjModel.from_xml_path(str(proxy));d=mj.MjData(m)
    sole=mj.mj_name2id(sm,mj.mjtObj.mjOBJ_GEOM,'shoe_sole_proxy')
    shoe_body=mj.mj_name2id(sm,mj.mjtObj.mjOBJ_BODY,'shoe')
    sq=mj.mj_name2id(sm,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sqa=int(sm.jnt_qposadr[sq])
    foot=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'left_ankle_roll_link')
    hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link')
    Q=np.empty(9);mj.mju_quat2Mat(Q,sm.geom_quat[sole]);Q=Q.reshape(3,3)
    sole_local=sm.geom_pos[sole].copy()
    shoe_geoms=[g for g in range(sm.ngeom)if sm.geom_bodyid[g]==shoe_body and sm.geom_contype[g]]
    t=z['time_s'];q=z['qpos_p1'];switch=min(int(np.searchsorted(t,a.release_time)),len(t)-1)
    poses=[];foot_poses=[];hand_world=[];foot_world=[];floor_shift=0.
    wear_local=None;hand_rel_pos=None;hand_rel_rot=None;initial_floor_min=None
    for f in range(len(t)):
        d.qpos[:]=q[f];mj.mj_kinematics(m,d)
        Rf=d.xmat[foot].reshape(3,3);Rh=d.xmat[hand].reshape(3,3)
        Rshoe=Rf@Q.T
        if wear_local is None:
            pos=(d.xpos[foot]+Rf@np.asarray(a.wear_local)) if a.wear_local is not None else (d.xpos[foot]+Rf@np.array([a.wear_forward,0,a.wear_sole_z])-Rshoe@sole_local)
            quat=np.empty(4);mj.mju_mat2Quat(quat,Rshoe.ravel())
            sd.qpos[sqa:sqa+7]=np.r_[pos,quat];mj.mj_kinematics(sm,sd)
            # Exact box support extents for all shoe shell collision primitives.
            bottom=min(sd.geom_xpos[g,2]-np.dot(np.abs(sd.geom_xmat[g].reshape(3,3)[2]),sm.geom_size[g])for g in shoe_geoms)
            initial_floor_min=float(bottom)
            floor_shift=max(a.floor_clearance-float(bottom),0.)
            if a.wear_local is not None and floor_shift>.0001:
                raise ValueError('Fixed wear transform requires a higher reference foot; do not push shoe into foot to clear ground')
            pos[2]+=floor_shift
            wear_local=Rf.T@(pos-d.xpos[foot])
        pos=d.xpos[foot]+Rf@wear_local
        foot_world.append(pos.copy());hand_world.append(d.xpos[hand].copy())
        fq=np.empty(4);mj.mju_mat2Quat(fq,Rshoe.ravel());foot_poses.append(np.r_[pos,fq])
        if f==switch:
            hand_rel_pos=Rh.T@(pos-d.xpos[hand]);hand_rel_rot=Rh.T@Rshoe
        if f>=switch:
            pos=d.xpos[hand]+Rh@hand_rel_pos;Rshoe=Rh@hand_rel_rot
        quat=np.empty(4);mj.mju_mat2Quat(quat,Rshoe.ravel());poses.append(np.r_[pos,quat])
    poses=np.asarray(poses);raw=z['shoe_raw_source_pose']
    inferred=np.zeros((len(t),2,2,1),dtype=bool)
    inferred[:,0,1,0]=t>=9.7
    payload.update(object_target_pose=poses,contact_optimized_shoe_pose=poses,
        shoe_attachment_name=np.where(np.arange(len(t))<switch,'shoe_wear_left','shoe_grasp_right'),
        shoe_contact_phase=np.where(np.arange(len(t))<switch,0,2),
        contact_candidate_observed=np.zeros_like(inferred),contact_candidate_inferred=inferred,
        contact_candidate_confidence=np.where(inferred,.5,0.),
        shoe_canonical_axes_columns=Q,shoe_wear_position_local_to_left_ankle=wear_local,
        shoe_wear_rotation_local_to_left_ankle=Q.T,
        shoe_hand_position_local_to_right_hand=hand_rel_pos,shoe_hand_rotation_local_to_right_hand=hand_rel_rot,
        shoe_wear_floor_shift_m=np.asarray(floor_shift),
        shoe_release_frame=np.asarray(switch),shoe_release_time_s=np.asarray(t[switch]),
        shoe_wear_only_reference_pose=np.asarray(foot_poses),
        shoe_source_position_correction_m=np.linalg.norm(poses[:,:3]-raw[:,:3],axis=1),
        inferred_contact_not_observed=np.asarray(True),
        adaptation_description=np.asarray('Full398x2 sourcebody IK retained; original shoe pose retained independently. Shoe shell PCA axes corrected to wear foot; exact collider floor correction applied once to fixed wear transform. At10.87s a fixed continuous right-hand grip transform is inferred from the aligned foot-shoe pose. Both attachments are declared hypotheses; no hand/finger contact was measured.'))
    np.savez_compressed(out,**payload)
    report={'schema':'rgb_near_contact_reference.v1','reference':str(out),'reference_sha256':sha(out),
        'script_sha256':generator_sha256,
        'input_reference_sha256':sha(inp),'asset_sha256':sha(asset),'frames':len(t),'actors':2,
        'wear_floor_shift_m':floor_shift,'wear_position_local':wear_local.tolist(),
        'initial_shoe_proxy_floor_min_z_m':initial_floor_min,
        'release_frame':switch,'release_time_s':float(t[switch]),'grasp_position_local':hand_rel_pos.tolist(),
        'grasp_body_origin_distance_m':float(np.linalg.norm(hand_rel_pos)),
        'max_frame_position_delta_m':float(np.max(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1))),
        'raw_source_position_correction_max_m':float(np.max(payload['shoe_source_position_correction_m'])),
        'raw_source_position_correction_median_m':float(np.median(payload['shoe_source_position_correction_m'])),
        'source_hand_valid':False,'contact_constraints_inferred':True,'dynamics_not_run_by_this_script':True}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    out.with_suffix('.generator.py').write_bytes(generator_bytes)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
