#!/usr/bin/env python3
"""Independent five-finger appearance prior for the generated robot-hand pixels.

WiLoR predictions are model-derived candidates, never measured robot joints.
The two designated views expose opposite physical hands; the far hand is not
invented from the box. Metric placement inherits MANO human-hand scale/cameras.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def roi(rgb, robot, palm_uv, direction):
    import cv2
    import numpy as np
    x, y = palm_uv
    x0, x1 = max(0, int(x)-85), min(640, int(x)+85)
    y0, y1 = max(0, int(y)-70), min(720, int(y)+65)
    region = robot[y0:y1, x0:x1]
    yy, xx = np.nonzero(region)
    if len(xx) < 40:
        return None
    distal = (xx.max()+x0) if direction > 0 else (xx.min()+x0)
    a, b = (max(0, distal-73), min(640, distal+4)) if direction > 0 else (max(0, distal-4), min(640, distal+73))
    ys, xs = np.nonzero(robot[y0:y1, a:b])
    if len(ys) < 30:
        return None
    c, d = max(0, int(ys.min()+y0)-5), min(720, int(ys.max()+y0)+6)
    if d-c > 120:
        c, d = max(0, int(y)-50), min(720, int(y)+50)
    return np.asarray([a, c, b, d], dtype=float)


def score_mesh(image, observed, vertices_xy, faces, box, direction):
    import cv2
    import numpy as np
    x0, y0, x1, y1 = np.round(box).astype(int)
    predicted = np.zeros(observed.shape, np.uint8)
    for face in faces:
        p = np.clip(vertices_xy[face], -10000, 10000).astype(np.int32)
        cv2.fillConvexPoly(predicted, p, 1)
    valid = np.zeros_like(predicted, bool)
    valid[y0+2:y1-2, x0+2:x1-2] = True
    # The detector-free ROI includes a wrist entry. Exclude its proximal strip
    # from the hand-only mask comparison for both prediction and observation.
    if direction > 0:
        valid[:, :x0+int((x1-x0)*.2)] = False
    else:
        valid[:, x1-int((x1-x0)*.2):] = False
    p, m = (predicted > 0) & valid, observed & valid
    intersection = int((p&m).sum()); union = int((p|m).sum())
    edge_p = cv2.morphologyEx(p.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
    edge_m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)) > 0
    border = cv2.dilate((~valid).astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    edge_p &= ~border; edge_m &= ~border
    if not edge_p.any() or not edge_m.any():
        error = float("inf")
    else:
        dp = cv2.distanceTransform((~edge_p).astype(np.uint8), cv2.DIST_L2, 3)
        dm = cv2.distanceTransform((~edge_m).astype(np.uint8), cv2.DIST_L2, 3)
        error = float((dp[edge_m].mean()+dm[edge_p].mean())/2)
    overlay = image.copy()
    overlay[p] = (.5*overlay[p]+.5*np.array([80,230,90])).astype(np.uint8)
    for face in faces[::4]:
        cv2.polylines(overlay, [np.clip(vertices_xy[face], -10000, 10000).astype(np.int32)], True, (40,190,40), 1)
    return {"iou": intersection/max(union,1), "symmetric_edge_mean_px": error,
            "observed_pixels": int(m.sum()), "predicted_pixels": int(p.sum())}, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--manual-keyframes", action="store_true", help="Use independently inspected body-agent palm points at frames 0/60/120; no interpolation")
    parser.add_argument("--landmark-rois", action="store_true", help="Use actual image palm landmarks with bidirectional/multi-anchor LK validity; skip missing observations")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("Only physical GPU6 is assigned for this experiment")
    import cv2
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    for name, value in {"bool":bool,"int":int,"float":float,"complex":complex,"object":object,"str":str,"unicode":str}.items():
        if name not in np.__dict__:setattr(np,name,value)
    if not hasattr(inspect,"getargspec"):inspect.getargspec=inspect.getfullargspec
    sys.path.insert(0,str(ROOT/'dependencies/wilor'))
    from wilor.configs import get_config
    from wilor.models import WiLoR
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    cv2.setNumThreads(3)
    source = ROOT/'outputs/user_reconstruction'
    output = ROOT/'outputs/user_reconstruction_v2';output.mkdir(parents=True,exist_ok=True)
    data = np.load(source/'input.npz'); rgb=data['rgb']; timestamps=data['timestamps']
    seg = np.load(source/'segmentation.npz');robot=seg['robot']
    camera_path=output/'camera_refinement.npz';camera=np.load(camera_path);P=camera['P_refined']
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene.json').read_text())
    selected = np.asarray([0,60,120]) if args.manual_keyframes else (np.arange(121) if args.all_frames else np.asarray([0,30,60,90,120]))
    manual_palms={0:{'right':[228,328],'left':[404,328]},60:{'right':[299,323],'left':[343,326]},120:{'right':[299,291],'left':[342,292]}}
    landmark_path=output/'body_landmark_observations.npz'
    landmarks=np.load(landmark_path,allow_pickle=False) if args.landmark_rois else None
    pairs=[('left',1,-1),('right',0,1)]
    wt=ROOT/'weights/wilor';checkpoint=wt/'pretrained_models/wilor_final.ckpt'
    cfg=get_config(str(wt/'pretrained_models/model_config.yaml'));cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256];cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    cfg.MANO.DATA_DIR=str(wt/'mano_data');cfg.MANO.MODEL_PATH=str(wt/'mano_data');cfg.MANO.MEAN_PARAMS=str(wt/'mano_data/mano_mean_params.npz');cfg.freeze()
    model=WiLoR(cfg,init_renderer=False);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    incompatible=model.load_state_dict(payload['state_dict'],strict=False)
    missing=set(incompatible.missing_keys)&set(dict(model.named_parameters()))
    if missing:raise RuntimeError(f'Missing learned parameters: {missing}')
    del payload;model=model.cuda().eval()
    faces=np.asarray(model.mano.faces)
    world=np.full((121,2,778,3),np.nan,np.float32);pixels=np.full((121,2,778,2),np.nan,np.float32)
    local_vertices=np.full_like(world,np.nan);mano_poses=np.full((121,2,15,3,3),np.nan,np.float32)
    shapes=np.full((121,2,10),np.nan,np.float32);rois=np.full((121,2,4),np.nan,np.float32)
    valid=np.zeros((121,2),bool);inferred=np.zeros_like(valid);metrics=[];tiles=[]
    start=time.monotonic()
    for frame in selected:
        for hand,(side,view,direction) in enumerate(pairs):
            palm=np.asarray(scene['robot']['link_world_transforms'][side+'_hand_palm_link'])[:3,3]
            uvw=P[view]@np.r_[palm,1];box=roi(rgb[frame,view],robot[frame,view],uvw[:2]/uvw[2],direction)
            if args.manual_keyframes:
                x,y=manual_palms[int(frame)][side]
                box=np.asarray([x-30 if direction>0 else x-40,y-30,x+40 if direction>0 else x+30,y+27],float)
            if args.landmark_rois:
                point=3 if side=='right' else 7
                if not bool(landmarks['valid'][frame,view,point]):
                    metrics.append({'frame':int(frame),'side':side,'view':view,'valid':False,'reason':'Actual palm landmark invalid; no reference/interpolation fallback'});continue
                x,y=landmarks['positions_xy'][frame,view,point]
                if not np.isfinite([x,y]).all():continue
                box=np.asarray([x-30 if direction>0 else x-40,y-30,x+40 if direction>0 else x+30,y+27],float)
                box[[0,2]]=np.clip(box[[0,2]],0,640);box[[1,3]]=np.clip(box[[1,3]],0,720)
            if box is None:
                metrics.append({'frame':int(frame),'side':side,'view':view,'valid':False,'reason':'No image-derived robot-hand ROI'});continue
            image=cv2.cvtColor(rgb[frame,view],cv2.COLOR_RGB2BGR)
            ds=ViTDetDataset(cfg,image,box[None],np.asarray([side=='right'],float),rescale_factor=1.5)
            batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=1,num_workers=0)))
            batch=recursive_to(batch,torch.device('cuda'))
            with torch.inference_mode():pred=model(batch)
            cam=pred['pred_cam'].clone();cam[:,1]*=2*batch['right']-1
            focal=cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*batch['img_size'].max()
            translation=cam_crop_to_full(cam,batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),focal).cpu().numpy()[0]
            vertices=pred['pred_vertices'][0].cpu().numpy().copy();vertices[:,0]*=1 if side=='right' else -1
            projected=vertices+translation;uv=projected[:,:2]/projected[:,2:]*float(focal)+np.array([320,360])
            result,overlay=score_mesh(image,robot[frame,view],uv,faces,box,direction)
            # A placement candidate from human MANO scale and the supplied camera.
            # The target 2D points are model predictions, not measured landmarks.
            K,R,C,*_=cv2.decomposeProjectionMatrix(P[view]);K/=K[2,2];C=(C[:3]/C[3]).ravel()
            camera_t=translation.copy();camera_t[2]*=float(K[0,0])/float(focal)
            camera_t[0]=((float(focal)*translation[0]/translation[2]+320-K[0,2])/K[0,0])*camera_t[2]
            camera_t[1]=((float(focal)*translation[1]/translation[2]+360-K[1,2])/K[1,1])*camera_t[2]
            ok,rvec,tvec=cv2.solvePnP(vertices.astype(np.float64),uv.astype(np.float64),K,np.zeros(4),np.zeros((3,1)),camera_t.reshape(3,1).astype(float),True,flags=cv2.SOLVEPNP_ITERATIVE)
            camera_vertices=vertices@Rotation.from_rotvec(rvec.ravel()).as_matrix().T+tvec.ravel()
            world[frame,hand]=camera_vertices@R+C
            pixels[frame,hand]=uv;inferred[frame,hand]=True
            local_vertices[frame,hand]=vertices;rois[frame,hand]=box
            mano_poses[frame,hand]=pred['pred_mano_params']['hand_pose'][0].detach().cpu().numpy().reshape(15,3,3)
            shapes[frame,hand]=pred['pred_mano_params']['betas'][0].detach().cpu().numpy()
            accepted=bool(result['iou']>=.5 and result['symmetric_edge_mean_px']<=4. and ok)
            valid[frame,hand]=accepted
            row={'frame':int(frame),'side':side,'view':view,'roi_xyxy':box.tolist(),**result,'valid':accepted,
                 'measurement_ground_truth':False,'metric_placement_validated':False}
            metrics.append(row)
            if frame in (0,30,60,90,120):
                x0,y0,x1,y1=np.round(box).astype(int);crop=overlay[max(0,y0-8):min(720,y1+8),max(0,x0-8):min(640,x1+8)]
                crop=cv2.resize(crop,(320,240));cv2.putText(crop,f'{frame} {side}: IoU{result["iou"]:.2f}, {result["symmetric_edge_mean_px"]:.1f}px',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1);tiles.append(crop)
            print(json.dumps(row),flush=True)
    suffix='_manual_roi' if args.manual_keyframes else ''
    destination=output/('mano_hand_candidate.npz' if args.all_frames else 'mano_hand_keyframes'+suffix+'.npz')
    np.savez_compressed(destination,vertices_world_m=world,vertices_pixels_xy=pixels,faces=faces,valid=valid,inferred=inferred,
                        vertices_mano_local_m=local_vertices,mano_hand_pose_rotation_matrices=mano_poses,betas=shapes,roi_xyxy=rois,
                        metric_valid=np.zeros_like(valid),side_names=np.asarray(['left','right']),source_views=np.asarray([1,0]),timestamps=timestamps,
                        source_video_sha256=data['video_sha256'],camera_sha256=np.asarray(sha(camera_path)),checkpoint_sha256=np.asarray(sha(checkpoint)))
    if tiles:
        while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
        sheet=np.concatenate([np.concatenate(tiles[i:i+2],axis=1) for i in range(0,len(tiles),2)],axis=0)
        cv2.imwrite(str(ROOT/('outputs/previews/user_mano_hand_keyframes'+suffix+'.jpg')),sheet)
    report={'status':'appearance_candidates_evaluated','method':'WiLoR + licensed MANO on real semantic robot-hand ROIs; detector bypassed',
            'robot_hardware':'Original G1 stock three fingers changed by user to public five-finger ROHand; MANO remains a distinct human appearance prior','appearance_prior':'Human MANO five fingers; independent from robot hardware',
            'original_stock_topology_conflict_retained':True,'predictions_are_measurements':False,'metric_world_placement_validated':False,
            'world_placement_policy':'Conditional on human MANO scale and supplied camera; solvePnP targets are WiLoR-derived pixels, not independently detected landmarks',
            'frames_inferred':int(inferred.any(1).sum()),'hands_inferred':int(inferred.sum()),'appearance_valid_count':int(valid.sum()),
            'all_121_frames':args.all_frames,'selected_frames':selected.tolist(),'elapsed_seconds':time.monotonic()-start,
            'roi_source':'Body-agent actual-image manual palm observations at frames 0/60/120, approximate uncertainty 6-8 pixels' if args.manual_keyframes else 'Semantic robot pixels around reference-projected palm; not reference-pose ground truth',
            'output':{'path':str(destination.relative_to(ROOT)),'sha256':sha(destination)},'metrics':metrics}
    if args.landmark_rois:
        report['roi_source']='Actual manual palm anchors plus bidirectional/multi-anchor-consistent image LK; missing points skipped, never reference-projected fallback'
        report['landmark_sha256']=sha(landmark_path)
    (ROOT/('reports/user-mano-hand-prior'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
