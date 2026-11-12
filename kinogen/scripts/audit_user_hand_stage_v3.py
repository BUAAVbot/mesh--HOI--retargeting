#!/usr/bin/env python3
"""Independent immutable hand-stage fallback, authority and dense-geometry audit."""
import argparse
import json
import os
from pathlib import Path

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,check_provenance
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--hand-file',type=Path,default=Path('outputs/user_reconstruction_v3/hand_trajectory_image_v3.npz'));ap.add_argument('--stage-report',type=Path,default=Path('reports/user-hand-optimization_image_v3.json'));ap.add_argument('--output',type=Path,default=Path('reports/user-v3-hand-stage-independent.json'));args=ap.parse_args()
    hp=(ROOT/args.hand_file).resolve();rp=(ROOT/args.stage_report).resolve();op=(ROOT/args.output).resolve();require(all(p.is_relative_to(ROOT)for p in (hp,rp,op)),'Outside project')
    report=json.loads(rp.read_text());provenance=check_provenance(rp);h=load(hp)
    ip=Path(report['initial_hand_parent']['path']);ip=ROOT/ip if not ip.is_absolute()else ip
    candidates=[ROOT/p for p in report['files']if p.endswith('.npz')]
    bp=next(p for p in candidates if sha(p)==str(h['source_body_sha256']))
    require(sha(ip)==str(h['source_hand_sha256']),'Stage initial hand SHA mismatch')
    b=load(bp);initial=load(ip);bn=list(map(str,b['joint_names']));hbn=list(map(str,h['body_joint_names']));q=h['body_joint_positions_rad_refined'][:,[hbn.index(n)for n in bn]]
    nonwrist=[k for k,n in enumerate(bn)if '_wrist_'not in n]
    require(len(nonwrist)==23 and np.array_equal(q[:,nonwrist],b['joint_positions_rad'][:,nonwrist]),'Non-wrist body changed')
    for key in ('base_xyz_m','base_quaternion_xyzw'):require(np.array_equal(h[key],b[key]),'Stage altered base')
    selected=h['new_hand_candidate_selected'];strict=h['new_hand_candidate_accepted'];conditional=h['refinement_selected_conditional']
    require(selected.shape==strict.shape==conditional.shape==(121,2),'Selection shape')
    require(np.array_equal(conditional,selected&~strict)and not(strict&~selected).any(),'Strict/conditional selection conflated')
    changed=[];joint_changed={}
    for sideindex,side in enumerate(('left','right')):
        rejected=~selected[:,sideindex];cols=[k for k,n in enumerate(map(str,h['actuator_names']))if n.startswith(side+'_')];oldcols=[list(map(str,initial['actuator_names'])).index(str(h['actuator_names'][k]))for k in cols]
        require(np.array_equal(h['actuator_positions'][rejected][:,cols],initial['actuator_positions'][rejected][:,oldcols]),'Rejected side controls not exact same-frame source')
        wrists=[k for k,n in enumerate(bn)if n.startswith(side+'_wrist_')]
        require(np.array_equal(q[rejected][:,wrists],b['joint_positions_rad'][rejected][:,wrists]),'Rejected side wrists changed')
        delta=q[:,wrists]!=b['joint_positions_rad'][:,wrists];changed.append(int(delta.any(axis=1).sum()))
        for k in wrists:joint_changed[bn[k]]=int((q[:,k]!=b['joint_positions_rad'][:,k]).sum())
    before_dense=h['dense_collision_metrics_before'];after_dense=h['dense_collision_metrics_after'];tol=np.array([1e-6,1e-7,1e-7])
    require(before_dense.shape==after_dense.shape==(121,2,3)and np.isfinite(before_dense).all()and np.isfinite(after_dense).all(),'Dense metrics incomplete')
    require(np.all(after_dense<=before_dense+tol+1e-12),'Declared dense metric regressed')
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());model=XMLModel(ROOT/scene['robot']['urdf']);box=load(V2/'box_trajectory.npz')
    before=model.audit(b,initial,box);after=model.audit(b,h,box)
    def dense(rows):return np.array([[[max(0,-r['side_minimum_signed_distance_m'][s]),r['side_all_sample_fraction_gt5mm'][s],r['side_mean_depth_beyond5mm_m'][s]]for s in range(2)]for r in rows])
    old=dense(before['per_frame']);new=dense(after['per_frame']);require(np.all(new<=old+tol+1e-12),'Independent52 raw visual metrics regressed')
    require(after['all_positive_original_body_velocity_limits_passed'],'Stage original body velocity failed')
    out={'status':'immutable_hand_stage_independent_checks_passed','hand':record(hp),'body':record(bp),'initial_hand':record(ip),'provenance':provenance,'selected_sides':int(selected.sum()),'strict_selected_sides':int(strict.sum()),'conditional_selected_sides':int(conditional.sum()),'actual_wrist_changed_sides':changed,'actual_wrist_changed_joint_entries':sum(joint_changed.values()),'wrist_changed_entries_by_joint':joint_changed,'reported_wrist_refinements':int(h['wrist_refinement_accepted'].sum()),'sameframe_rejected_controls_and_three_wrist_axes_exact':True,'other23_body_angles_and_base_exact':True,'independent52_raw_visual_nonregression_passed':True,'independent_dense_metric_max_increase':np.max(new-old,axis=(0,1)).tolist(),'raw_geometry':after,'image_fit_valid_sides':h['image_fit_valid'].sum(0).tolist(),'strict_side_valid_sides':h['side_valid'].sum(0).tolist(),'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'note':'Selection includes conditional improvements; strict-only mask must not be used for fallback. All 3D support flags still require final conservative canonical intersection and clearing for changed world wrist poses.'}
    op.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items()if k not in ('provenance','raw_geometry')},indent=2))

if __name__=='__main__':main()
