#!/usr/bin/env python3
"""Read-only non-hand original-visual versus frozen-box sampled SDF diagnostic."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from verify_user_reconstruction_v3 import ROOT,V2,XMLModel,load,sha,record,require,BOX_SHA,URDF_SHA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--body-file',type=Path,default=Path('outputs/user_reconstruction_v4/stage_body.npz'))
    ap.add_argument('--hand-file',type=Path,default=Path('outputs/user_reconstruction_v4/stage_hand.npz'))
    ap.add_argument('--output',type=Path,default=Path('reports/user-v4-arm-box-diagnostic.json'))
    args=ap.parse_args()
    for key in ('body_file','hand_file','output'):
        p=(ROOT/getattr(args,key)).resolve();require(p.is_relative_to(ROOT),'Path outside project');setattr(args,key,p)
    oldb=ROOT/'outputs/user_reconstruction_v3/body_trajectory.npz';oldh=ROOT/'outputs/user_reconstruction_v3/hand_trajectory.npz'
    require(sha(oldb)=='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'and sha(oldh)=='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1','Frozen V3 changed')
    boxpath=V2/'box_trajectory.npz';require(sha(boxpath)==BOX_SHA,'Frozen box changed');box=load(boxpath)
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());urdf=ROOT/scene['robot']['urdf'];require(sha(urdf)==URDF_SHA,'URDF changed');model=XMLModel(urdf)
    parts=[p for p in model.parts if not p[3]];require(parts,'No non-hand mesh visuals')

    def audit(bp,hp):
        B,H=load(bp),load(hp);require(str(H['source_body_sha256'])==sha(bp),'Actual hand not bound to supplied body')
        bn=list(map(str,B['joint_names']));hn=list(map(str,H['joint_names']));hbn=list(map(str,H['body_joint_names']))
        bodyq=H['body_joint_positions_rad_refined'][:,[hbn.index(n)for n in bn]]
        require(np.array_equal(bodyq,B['joint_positions_rad']),'Actual body/hand states differ')
        summary={};frames=[]
        for f in range(121):
            q=dict(zip(bn,bodyq[f]));q.update(zip(hn,H['joint_positions'][f]));tf=model.fk(B['base_xyz_m'][f],B['base_quaternion_xyzw'][f],q)
            rot=model.Rotation.from_quat(box['quaternion_xyzw'][f]).as_matrix();center=box['center_xyz_m'][f];half=box['box_size_xyz_m']/2
            row={'frame':f,'max_nonhand_box_sample_penetration_m':0.,'links_deeper_than_5mm':[],'per_link':{}}
            for link,points,nvertices,_ in parts:
                T=tf[link];world=points@T[:3,:3].T+T[:3,3];d=np.abs((world-center)@rot)-half
                sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0);depth=float(max(0,-sdf.min()));deep=sdf<-.005
                value={'minimum_signed_distance_m':float(sdf.min()),'max_penetration_m':depth,'sample_count':len(sdf),'unique_vertices':nvertices,'triangle_centers':len(sdf)-nvertices,'vertices_deeper_than_5mm':int(deep[:nvertices].sum()),'triangle_centers_deeper_than_5mm':int(deep[nvertices:].sum()),'fraction_deeper_than_5mm':float(deep.mean())}
                # This model has one visual per non-hand link; reject an unhandled aggregation.
                require(link not in row['per_link'],'Multiple visuals for non-hand link require explicit aggregation')
                row['per_link'][link]=value;row['max_nonhand_box_sample_penetration_m']=max(row['max_nonhand_box_sample_penetration_m'],depth)
                if depth>.005:row['links_deeper_than_5mm'].append(link)
                if link not in summary:summary[link]={'max_penetration_m':0.,'worst_frame':None,'frames_deeper_than_5mm':[],'sample_count':len(sdf)}
                s=summary[link]
                if depth>s['max_penetration_m']:s['max_penetration_m']=depth;s['worst_frame']=f
                if depth>.005:s['frames_deeper_than_5mm'].append(f)
            frames.append(row)
        return {'source_body':record(bp),'source_hand':record(hp),'max_penetration_m':max(x['max_nonhand_box_sample_penetration_m']for x in frames),'frames_deeper_than_5mm':[x['frame']for x in frames if x['links_deeper_than_5mm']],'per_link':summary,'per_frame':frames}

    baseline=audit(oldb,oldh);candidate=audit(args.body_file,args.hand_file)
    report={'status':'independent_nonhand_box_sample_diagnostic','verification_script':record(Path(__file__)),'inputs':[record(p)for p in (oldb,oldh,args.body_file,args.hand_file,boxpath,urdf)],'method':'Independent normalized-axis XML FK; every original non-hand visual unique vertex and every triangle center with actual visual scale/origin. Frozen box pose/size and camera are not optimized. This is a new diagnostic, not an altered acceptance threshold or exact continuous collision test.','nonhand_visual_count':len(parts),'hand52_excluded':True,'baseline_frozen_v3':baseline,'stage_candidate':candidate,'frame54':{'baseline':baseline['per_frame'][54],'candidate':candidate['per_frame'][54]},'nonhand_no_worse_per_frame':all(c['max_nonhand_box_sample_penetration_m']<=b['max_nonhand_box_sample_penetration_m']+1e-6 for b,c in zip(baseline['per_frame'],candidate['per_frame'])),'physical_hardware_validated':False,'self_collision_verified':False,'strict_metric_reconstruction_valid':False}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(args.output),'baseline_max_m':baseline['max_penetration_m'],'candidate_max_m':candidate['max_penetration_m'],'baseline_failed_frames':baseline['frames_deeper_than_5mm'],'candidate_failed_frames':candidate['frames_deeper_than_5mm'],'frame54_candidate_deep_links':candidate['per_frame'][54]['links_deeper_than_5mm'],'candidate_deep_links':{k:v for k,v in candidate['per_link'].items()if v['max_penetration_m']>.005}},indent=2))


if __name__=='__main__':main()
