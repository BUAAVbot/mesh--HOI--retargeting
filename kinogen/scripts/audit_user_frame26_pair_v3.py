#!/usr/bin/env python3
"""Independent real F26 joint-pair audit before canonical scene export."""
import json
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,check_provenance,declared_body_fk_audit,foot_track_comparison,frame54_image_comparison
    out=ROOT/'outputs/user_reconstruction_v3';bp=out/'body_frame26_joint.npz';hp=out/'hand_frame26_joint.npz';parent_hp=out/'hand_trajectory_image_v3.npz';parent_bp=out/'prehand_body.npz';rp=ROOT/'reports/user-frame26-joint.json'
    b,h,pb,ph=map(load,(bp,hp,parent_bp,parent_hp));provenance=check_provenance(rp)
    require(str(h['source_body_sha256'])==sha(bp)and str(h['source_hand_sha256'])==sha(parent_hp),'F26 source pair differs')
    bn=list(map(str,b['joint_names']));pn=list(map(str,ph['body_joint_names']));sourceq=ph['body_joint_positions_rad_refined'][:,[pn.index(n)for n in bn]];hq=h['body_joint_positions_rad_refined'][:,[list(map(str,h['body_joint_names'])).index(n)for n in bn]]
    require(np.array_equal(hq,b['joint_positions_rad']),'F26 actual hand/body angles differ')
    others=np.arange(121)!=26
    require(np.array_equal(b['joint_positions_rad'][others],sourceq[others]),'Changed another frame actual stageH body angles')
    for key in ('base_xyz_m','base_quaternion_xyzw'):require(np.array_equal(b[key][others],ph[key][others])and np.array_equal(h[key],b[key]),'Base context changed')
    require(np.array_equal(h['actuator_positions'][others],ph['actuator_positions'][others]),'Changed another frame controls')
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());model=XMLModel(ROOT/scene['robot']['urdf']);box=load(V2/'box_trajectory.npz')
    before=model.audit(pb,ph,box);after=model.audit(b,h,box);saved=declared_body_fk_audit(model,b,scene)
    foot=foot_track_comparison(model,scene,pb,ph,b,h)
    images=[frame54_image_comparison(model,pb,ph,b,h,'dense',26,side)for side in ('left','right')]
    L=load(V2/'body_landmark_observations.npz');P=load(V2/'camera_refinement.npz')['P_refined'];q=dict(zip(bn,b['joint_positions_rad'][26]));q.update(dict(zip(map(str,h['joint_names']),h['joint_positions'][26])))
    tf=model.fk(b['base_xyz_m'][26],b['base_quaternion_xyzw'][26],q);wrists=[]
    for side,view,k in [('left',1,6),('right',0,2)]:
        X=np.r_[tf[side+'_wrist_yaw_link'][:3,3],1]@P[view].T;uv=X[:2]/X[2]
        wrists.append({'side':side,'view':view,'source_manual_LK_valid':bool(L['valid'][26,view,k]),'error_px':float(np.linalg.norm(uv-L['positions_xy'][26,view,k])),'approx_manual_uncertainty_px':8})
    report={'status':'frame26_actual_pair_independently_audited','inputs':[record(p)for p in (bp,hp,parent_bp,parent_hp)],'provenance':provenance,'other120_actual_refined_body_base_and_controls_exact':True,'frame26_saved_body_fk':saved,'candidate':after,'before_frame26':before['per_frame'][26],'after_frame26':after['per_frame'][26],'fixed_material_foot_frame26':foot['per_frame'][26],'actual_wrist_keypoint_errors_frame26':wrists,'fixed_rgb_hand_image_frame26':images,'producer_image_gates':json.loads(rp.read_text())['after'],'producer_metric_scope':'SHA-bound finite joint-search image/body/foot acceptance diagnostics; independent XML FK, raw geometry, fixed material feet and fixed RGB hand metrics are separate fields.','strict_metric_reconstruction_valid':False,'physical_hardware_validated':False}
    p=ROOT/'reports/user-v3-frame26-independent.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(p),'other120_exact':True,'saved_fk_error':saved['max_abs_error'],'raw_floor_failed_frames':after['ground_failed_frames_gt5mm'],'raw_hand_failed_frames':after['hand_failed_frames_gt5mm'],'original_body_velocity_passed':after['all_positive_original_body_velocity_limits_passed'],'frame26':after['per_frame'][26],'wrists':wrists,'foot_heldout':[v['candidate_fixed_heldout_ids']for v in foot['per_frame'][26]['views']]},indent=2))

if __name__=='__main__':main()
