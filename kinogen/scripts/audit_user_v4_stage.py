#!/usr/bin/env python3
"""Limited read-only V4 F54 comparison against immutable canonical V3."""
import argparse
import json
import os
from pathlib import Path

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,declared_body_fk_audit,frame54_image_comparison,check_provenance,CAMERA_SHA,BOX_SHA,URDF_SHA,VIDEO_SHA
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--body-file',type=Path,default=Path('outputs/user_reconstruction_v4/stage_body.npz'));ap.add_argument('--hand-file',type=Path,default=Path('outputs/user_reconstruction_v4/stage_hand.npz'));ap.add_argument('--allowed-frames',default='54',help='Explicit comma-separated right-arm/right-control frame ownership');ap.add_argument('--scene-report',type=Path);ap.add_argument('--provenance-report',type=Path,action='append',default=[]);ap.add_argument('--output',type=Path,default=Path('reports/user-v4-stage-independent.json'));args=ap.parse_args()
    frames=sorted(set(map(int,args.allowed_frames.split(','))));require(frames and all(0<=f<121 for f in frames),'Invalid allowed frames')
    for k in ('body_file','hand_file','scene_report','output'):
        p=getattr(args,k)
        if p is not None:
            p=(ROOT/p).resolve();require(p.is_relative_to(ROOT),'Outside project');setattr(args,k,p)
    oldb=ROOT/'outputs/user_reconstruction_v3/body_trajectory.npz';oldh=ROOT/'outputs/user_reconstruction_v3/hand_trajectory.npz'
    require(sha(oldb)=='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'and sha(oldh)=='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1','Frozen V3 parent changed')
    b,h,pb,ph=map(load,(args.body_file,args.hand_file,oldb,oldh))
    require(sha(V2/'camera_refinement.npz')==CAMERA_SHA and sha(V2/'box_trajectory.npz')==BOX_SHA,'Frozen geometry changed')
    require(str(b['camera_sha256'])==str(h['camera_sha256'])==CAMERA_SHA and str(h['box_sha256'])==BOX_SHA,'Candidate used another camera/box')
    require(str(b['source_video_sha256'])==str(h['source_video_sha256'])==VIDEO_SHA,'Candidate source video changed')
    require(np.array_equal(b['camera_P_video'],pb['camera_P_video']),'Candidate changed projection')
    require(str(h['source_body_sha256'])==sha(args.body_file),'Stage hand not bound to body')
    require(str(h['source_hand_sha256'])==sha(oldh),'Stage hand initializer not frozen V3')
    bn=list(map(str,b['joint_names']));oldbn=list(map(str,pb['joint_names']));q=b['joint_positions_rad'];oldq=pb['joint_positions_rad'][:,[oldbn.index(n)for n in bn]]
    allowed=np.zeros((121,29),bool)
    for k,n in enumerate(bn):allowed[frames,k]=n.startswith('right_')and any(s in n for s in ('shoulder_','elbow_','wrist_'))
    require(allowed.sum()==7*len(frames) and np.array_equal(q[~allowed],oldq[~allowed]),'Changed outside explicit owned rightarm7 frames')
    for k in ('base_xyz_m','base_quaternion_xyzw'):require(np.array_equal(b[k],pb[k]),'Stage moved base')
    names=list(map(str,h['body_joint_names']));require(np.array_equal(h['body_joint_positions_rad_refined'][:,[names.index(n)for n in bn]],q),'Stage hand/body actual angles differ')
    require(np.array_equal(h['actuator_names'],ph['actuator_names']),'Actuator order changed')
    a=np.zeros((121,12),bool);a[frames,6:]=True
    require(np.array_equal(h['actuator_positions'][~a],ph['actuator_positions'][~a]),'Changed another side/frame controls')
    require(bool(b['finger_joint_positions_included']),'Final stage body must declare actual coupled fingers')
    for key in ('actuator_names','actuator_positions','joint_names','joint_positions','joint_units'):
        require(np.array_equal(b['finger_'+key],h[key]),'Final stage body finger metadata differs from actual hand: '+key)
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());require(sha(ROOT/scene['robot']['urdf'])==URDF_SHA,'Robot model changed');model=XMLModel(ROOT/scene['robot']['urdf']);box=load(V2/'box_trajectory.npz')
    before=model.audit(pb,ph,box);after=model.audit(b,h,box);saved=declared_body_fk_audit(model,b,scene)
    images={str(f):frame54_image_comparison(model,pb,ph,b,h,'dense',frame=f)for f in frames}
    landmark_path=V2/'body_landmark_observations.npz';ground_path=ROOT/'outputs/user_reconstruction_v3/ground_trajectory_retry.npz'
    landmarks=load(landmark_path);ground=load(ground_path)
    point_links=[s+'_'+n+'_link' for s in ('right','left') for n in ('shoulder_roll','elbow','wrist_yaw','wrist_yaw')]
    def image_landmarks(state,frame):
        qmap=dict(zip(map(str,state['joint_names']),state['joint_positions_rad'][frame]));qmap.update(dict(zip(map(str,h['joint_names']),h['joint_positions'][frame])))
        tf=model.fk(state['base_xyz_m'][frame],state['base_quaternion_xyzw'][frame],qmap);world=np.array([tf[n][:3,3] for n in point_links]);rows=[]
        for view,P in enumerate(b['camera_P_video']):
            uv=np.column_stack((world,np.ones(8)))@P.T;uv=uv[:,:2]/uv[:,2:];good=landmarks['valid'][frame,view].copy();good[[3,7]]=False;errors=np.linalg.norm(uv[good]-landmarks['positions_xy'][frame,view,good],axis=1);wi=2+4*view
            rows.append({'view':view,'observed_points':int(good.sum()),'median_px':float(np.median(errors)),'p90_px':float(np.percentile(errors,90)),'wrist_error_px':float(np.linalg.norm(uv[wi]-landmarks['positions_xy'][frame,view,wi])) if landmarks['valid'][frame,view,wi] else None})
        return rows
    landmark_results={}
    for f in frames:
        ground_landmarks=image_landmarks(ground,f);candidate_landmarks=image_landmarks(b,f);checks=[]
        for old,new in zip(ground_landmarks,candidate_landmarks):
            limits={'median_px':min(8,old['median_px']+2),'p90_px':min(16,old['p90_px']+3),'wrist_error_px':min(8,old['wrist_error_px']+3) if old['wrist_error_px'] is not None else None}
            checks.append({'view':new['view'],'limits_px':limits,'all_available_necessary_gates_passed':all(new[k] is not None and new[k]<=v+1e-6 for k,v in limits.items() if v is not None)})
        landmark_results[str(f)]={'frozen_v3':image_landmarks(pb,f),'ground_anchor':ground_landmarks,'candidate':candidate_landmarks,'checks':checks}
    report={'status':'v4_limited_raw_audit_visibility_pending','inputs':[record(p)for p in (args.body_file,args.hand_file,oldb,oldh)],'verification_script':record(Path(__file__)),'allowed_frames':frames,'unchanged_frame_count':121-len(frames),'unowned_actual_body_and_controls_exact':True,'only_owned_rightarm7_and_right_controls6_mutable':True,'base_legs_waist_exact':'Verified by parameter identity; fixed foot material projections therefore identical','body_declares_exact_actual_hand_controls_and_coupled_joints':True,'declared_body_fk':saved,'baseline_frozen_v3':before,'candidate':after,'fixed_rgb_proxy_by_frame':images,'actual_complete_scene_visibility_verified':False,'overall_reconstruction_success':False,'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False}
    report['fixed_body_landmarks']={'sources':[record(landmark_path),record(ground_path)],'by_frame':landmark_results,'policy':'Same fixed source RGB landmark IDs and unchanged per-view ground-relative median/P90/wrist budgets. No candidate-selected or reattached observations.'}
    report['provenance_reports']=[check_provenance((ROOT/p).resolve()) for p in args.provenance_report]
    if args.scene_report:
        sr=json.loads(args.scene_report.read_text())
        for key,p in [('body_trajectory',args.body_file),('hand_trajectory',args.hand_file)]:require(sr['inputs'][key]['sha256']==sha(p),'Renderer used another candidate')
        for x in sr['outputs']:require(sha(ROOT/x['path'])==x['sha256'],'Scene output changed')
        actual={}
        for f in frames:
            rows=[r for r in sr['per_frame_mesh_audit']if r['frame']==f];require(len(rows)==1,'No actual owned-frame render')
            vr=[x for x in rows[0]['visible_hand_geometry_comparison']if x is not None and x['side']=='right'];require(len(vr)==1,'Missing owned-frame observed right hand')
            visible=vr[0]['candidate'];actual[str(f)]={'comparison':vr[0],'observed_hand_presence_passed':bool(not visible['missing_visible_model_contour']and visible['visible_hand_pixels_in_domain']>0)}
        report['actual_visibility_source']=record(args.scene_report);report['actual_visible_hand_by_frame']=actual;report['actual_complete_scene_visibility_verified']=True
        report['status']='v4_limited_candidate_visible_hand_present'if all(x['observed_hand_presence_passed']for x in actual.values())else'v4_limited_candidate_visible_hand_failed'
        report['visible_hand_comparison_summary']=sr.get('visible_hand_comparison_summary')
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(args.output),'status':report['status'],'raw_floor_failed':after['ground_failed_frames_gt5mm'],'raw_hand_failed':after['hand_failed_frames_gt5mm'],'original_body_velocity_passed':after['all_positive_original_body_velocity_limits_passed'],'image_by_frame':{f:x['candidate']for f,x in images.items()},'actual_visibility':report.get('actual_visible_hand_by_frame')},indent=2))

if __name__=='__main__':main()
