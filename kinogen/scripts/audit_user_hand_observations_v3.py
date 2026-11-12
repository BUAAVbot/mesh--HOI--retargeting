#!/usr/bin/env python3
"""Read-only audit of legacy hand observation censorship; counts are not finger truth."""
from pathlib import Path
import json
import hashlib
import os

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import cv2
    from optimize_user_hands import observation
    paths=[ROOT/'outputs/user_reconstruction/input.npz',ROOT/'outputs/user_reconstruction/segmentation.npz',ROOT/'outputs/user_reconstruction_v2/body_landmark_observations.npz',ROOT/'scripts/optimize_user_hands.py']
    with np.load(paths[0])as z:rgb=z['rgb']
    with np.load(paths[1])as z:robot=z['robot'];box=z['box'];valid=z['observation_valid']
    with np.load(paths[2])as z:palm=z['positions_xy'];pv=z['valid']
    rows=[]
    for f in range(50,59):
        for side,v,d,k in [('left',1,-1,7),('right',0,1,3)]:
            row={'frame':f,'side':side,'view':v,'legacy_palm_xy':palm[f,v,k].tolist(),'palm_track_valid':bool(pv[f,v,k])}
            if not pv[f,v,k]:rows.append(row);continue
            o=observation(rgb[f,v],robot[f,v],box[f,v],valid[f,v],palm[f,v,k],d)
            if o is None:row['legacy_observation_available']=False;rows.append(row);continue
            row['legacy_observation_available']=True
            raw=cv2.Canny(cv2.cvtColor(rgb[f,v],cv2.COLOR_RGB2GRAY),40,110)>0
            border=cv2.dilate((~o['roi']).astype(np.uint8),np.ones((5,5),np.uint8))>0
            domain=o['roi']&~border&valid[f,v]
            excluded=cv2.dilate(box[f,v].astype(np.uint8),np.ones((3,3),np.uint8))>0
            candidate=raw&robot[f,v]&domain;removed=candidate&excluded
            hsv=cv2.cvtColor(rgb[f,v],cv2.COLOR_RGB2HSV)
            neutral=(hsv[:,:,1]<55)&(hsv[:,:,2]>45)&(hsv[:,:,2]<245)
            row.update(robot_box_overlap_pixels=int((robot[f,v]&box[f,v]&domain).sum()),actual_rgb_robot_edge_pixels_before_box_cut=int(candidate.sum()),actual_rgb_robot_edge_pixels_removed_by_box_cut=int(removed.sum()),removed_fraction=float(removed.sum()/max(candidate.sum(),1)),removed_low_saturation_midvalue_edge_pixels=int((removed&neutral).sum()),legacy_training_sample_count=len(o['points']),legacy_heldout_count=len(o['heldout_points']))
            rows.append(row)
    report={'status':'legacy_hand_observation_policy_audited_read_only','inputs':{str(p.relative_to(ROOT)):sha(p)for p in paths},'source_segmentation_unchanged':True,'frames':list(range(50,59)),'records':rows,'source_logic':{'deletion':'observed=(robot_mask_gradient | (RGB_Canny & robot)) & roi & ~border & ~dilate(box,3x3)','heldout':'Every fifth spatial6px block is split only after deletion; train/heldout share semantic censoring.','projection_weights':'image_residual marks box & !robot as occluded; projected robot&box points keep weight but target Canny edges within box were deleted.'},'interpretation':'Removed edges are actual RGB robot-mask edges, not independently labeled fingers. Overlap may include SAM errors, tape or box texture. Neutral color counts are descriptive only, never new finger labels or metric truth.','recommendations':['Fix observation domain from candidate-independent actual RGB/ROI evidence before comparing candidates; preserve ambiguous overlap.','Do not simply include all box Canny edges as fingers; distinguish gray robot surfaces, brown box and tape with reviewed evidence and uncertainty.','Keep fixed source pixel coordinates/regions for train and holdout. Recompute both old/new poses against the same new observation domain.','Spatial holdout inside one censored class mask cannot validate omitted fingers; report cross-domain and independent neighboring-frame evidence.','Do not retroactively change frozenv2 observation/valid counts.'], 'gpu_used':False,'training':False}
    path=ROOT/'reports/user-v3-hand-observation-audit.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(path),'sha256':sha(path),'records':rows},indent=2))

if __name__=='__main__':main()
