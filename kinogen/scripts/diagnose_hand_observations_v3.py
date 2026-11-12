from pathlib import Path
import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def main():
 import numpy as np,cv2
 from optimize_user_hands import observation
 from user_hand_observations_v3 import make_observation
 p=ROOT/'outputs/user_reconstruction';v=np.load(p/'input.npz');s=np.load(p/'segmentation.npz');l=np.load(ROOT/'outputs/user_reconstruction_v2/body_landmark_observations.npz');out=ROOT/'outputs/user_reconstruction_v3';out.mkdir(exist_ok=True)
 rows=[];records=[];masks=[];edges=[]
 for f in (50,54,58):
  image=v['rgb'][f,0];robot=s['robot'][f,0];box=s['box'][f,0];valid=s['observation_valid'][f,0];palm=l['positions_xy'][f,0,3];wrist=l['positions_xy'][f,0,2]
  old=observation(image,robot,box,valid,palm,1);new=make_observation(image,robot,box,valid,palm,wrist);base=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);tiles=[]
  for mode in range(4):
   x=base.copy()
   if mode==1:
    x[robot]=x[robot]//2+np.asarray([0,100,0],np.uint8);x[box]=x[box]//2+np.asarray([0,50,100],np.uint8);x[robot&box]=(255,0,255)
   if mode==2:
    for xy in old['points']:cv2.circle(x,tuple(np.round(xy).astype(int)),1,(0,255,255),-1)
    for xy in old['heldout_points']:cv2.circle(x,tuple(np.round(xy).astype(int)),1,(255,50,50),-1)
   if mode==3:
    x[new['hand_mask']]=x[new['hand_mask']]//2+np.asarray([90,0,0],np.uint8)
    for xy in new['points']:cv2.circle(x,tuple(np.round(xy).astype(int)),1,(0,255,255),-1)
    for xy in new['heldout_points']:cv2.circle(x,tuple(np.round(xy).astype(int)),1,(255,50,50),-1)
   crop=cv2.resize(x[285:365,250:345],(380,320),interpolation=cv2.INTER_NEAREST);cv2.putText(crop,['RGB','SAM green=robot brown=box','OLD train yellow held blue','V3 gray-metal visible hand'][mode],(4,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);tiles.append(crop)
  rows.append(np.concatenate(tiles,axis=1));masks.append(new['hand_mask']);edges.append(new['edge_mask'])
  roi=new['roi'];oldraw=cv2.distanceTransform((old['distance']>0).astype(np.uint8),cv2.DIST_L2,3)==0
  records.append({'frame':f,'robot_box_overlap_in_roi_px':int((robot&box&roi).sum()),'v3_visible_hand_px':int(new['hand_mask'].sum()),'v3_training_edges':len(new['points']),'v3_heldout_edges':len(new['heldout_points']),'old_training_points':len(old['points']),'old_heldout_points':len(old['heldout_points']),'old_box_dilation_excluded_robot_pixels':int((roi&robot&(cv2.dilate(box.astype(np.uint8),np.ones((3,3),np.uint8))>0)).sum())})
 cv2.imwrite(str(out/'hand_observation_v3_50_54_58.jpg'),np.concatenate(rows,axis=0));np.savez_compressed(out/'hand_observation_v3_keyframes.npz',frame_indices=np.asarray([50,54,58]),view=np.asarray(0),hand_mask=np.asarray(masks),edge_mask=np.asarray(edges))
 report={'status':'image_only_observation_diagnostic','source_masks_unchanged':True,'uses_model_projection':False,'rules':'Existing palm/wrist manual+LK ROI; saturation<65, V45..245 gray metal connected to source robot; source robot ownership wins over neighboring box. Exclude artificial ROI/wrist-cut edges. Fixed six-pixel spatial blocks mod5 held out.','manual_wrist_palm_uncertainty_px':8,'records':records};(ROOT/'reports/user-hand-observation-v3.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
