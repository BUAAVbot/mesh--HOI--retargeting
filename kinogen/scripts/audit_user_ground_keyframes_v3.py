#!/usr/bin/env python3
"""Independent five-keyframe audit in immutable original-neighbor context."""
from pathlib import Path
import argparse
import json
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,foot_track_comparison,CAMERA_SHA,BOX_SHA
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate',type=Path,default=Path('outputs/user_reconstruction_v3/ground_keyframes.npz'));ap.add_argument('--output',type=Path,default=Path('reports/user-v3-ground-keyframes-independent.json'));args=ap.parse_args()
    p=(ROOT/args.candidate).resolve();out=(ROOT/args.output).resolve();require(p.is_relative_to(ROOT)and out.is_relative_to(ROOT),'Outside project');g=load(p)
    bp=V2/'body_trajectory_contact.npz';hp=V2/'hand_trajectory_contact.npz'
    require(str(g['source_body_sha256'])==sha(bp)and str(g['source_hand_sha256'])==sha(hp),'Keyframes parent changed')
    require(str(g['camera_sha256'])==CAMERA_SHA and str(g['source_box_sha256'])==BOX_SHA,'Frozen geometry changed')
    baseline=load(bp);parent=load(hp);box=load(V2/'box_trajectory.npz');scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());model=XMLModel(ROOT/scene['robot']['urdf'])
    indices=g['frame_indices'].astype(int);require(len(set(indices))==len(indices),'Repeated keyframe')
    require(np.array_equal(g['timestamps'],baseline['timestamps'][indices]),'Keyframe timestamps differ')
    body={k:v.copy()for k,v in baseline.items()};hand={k:v.copy()for k,v in parent.items()}
    # In-memory context only. No new poses, interpolation, attachments or output trajectory.
    for k in ('base_xyz_m','base_quaternion_xyzw','joint_positions_rad','joint_image_supported','joint_approx_uncertainty_rad'):
        body[k][indices]=g[k]
    hand['body_joint_positions_rad_refined'][indices]=g['joint_positions_rad']
    require(np.array_equal(g['finger_actuator_positions'],parent['actuator_positions'][indices]),'Ground changed parent controls')
    require(np.array_equal(g['finger_joint_positions'],parent['joint_positions'][indices]),'Ground changed official coupled states')
    before=model.audit(baseline,parent,box);after=model.audit(body,hand,box)
    fixed=foot_track_comparison(model,scene,baseline,parent,body,hand)
    max_fk=0.
    for row,f in enumerate(indices):
        q=dict(zip(map(str,g['joint_names']),g['joint_positions_rad'][row]));q.update(dict(zip(map(str,parent['joint_names']),parent['joint_positions'][f])))
        tf=model.fk(g['base_xyz_m'][row],g['base_quaternion_xyzw'][row],q)
        max_fk=max(max_fk,float(np.abs(np.stack([tf[str(n)]for n in g['link_names']])-g['link_world_transforms'][row]).max()))
    require(max_fk<1e-7,'Ground saved FK differs from independent XML')
    rows=[{'frame':int(f),'before':before['per_frame'][f],'after':after['per_frame'][f],'fixed_material_track_projection':fixed['per_frame'][f]}for f in indices]
    report={'status':'independent_ground_candidate_audit_completed','source_candidate':record(p),'parents':[record(bp),record(hp)],'frames_actually_audited':indices.tolist(),'frame_count':len(indices),'all121_evaluated':len(indices)==121,'candidate_accepted_frames':int(g['accepted'].sum()),'candidate_declared_ground_failed_frames':indices[~g['ground_nonpenetration_valid']].tolist(),'in_memory_context_policy':'Only provided frames are replaced for evaluating exact raw geometry and adjacent original-neighbor velocity. No new trajectory is written and no untested frame is labeled repaired.','independent_saved_keyframe_xml_fk_max_error':max_fk,'all_positive_original_body_velocity_limits_passed_in_source_context':after['all_positive_original_body_velocity_limits_passed'],'observed_foot_points_not_sole_height':True,'fixed_material_attachment_policy':fixed['policy'],'fixed_local_xyz_m':fixed['fixed_local_xyz_m'],'point_link_names':fixed['point_link_names'],'heldout_ids':fixed['fixed_heldout_ids'],'raw_geometry_method':'Original full mesh unique vertices+hand triangle centers; normalized-axis independent XML FK, fixed box/P/floor','records':rows,'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'gpu_used':False}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shown=rows if len(rows)<=5 else[x for x in rows if x['frame']in(0,26,54,60,120)]
    print(json.dumps({'output':record(out),'max_fk_error':max_fk,'evaluated_frames':len(indices),'declared_ground_failed_frames':report['candidate_declared_ground_failed_frames'],'selected_diagnostics':[{'frame':x['frame'],'before_min_z_m':x['before']['robot_visual_min_z_m'],'after_min_z_m':x['after']['robot_visual_min_z_m'],'after_sole_min_z_m':x['after']['sole_min_z_m'],'after_hand_depth_m':max(x['after']['hand_vertex_depth_m'],x['after']['hand_trianglecenter_depth_m']),'foot_heldout_views':[v['candidate_fixed_heldout_ids']for v in x['fixed_material_track_projection']['views']]}for x in shown]},indent=2))

if __name__=='__main__':main()
