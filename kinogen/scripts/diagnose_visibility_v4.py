"""Read-only F54 complete raw-mesh visibility diagnosis, immutable V3 source."""
from pathlib import Path
import json,time
from user_visibility_v4 import Visibility54,ROOT,sha

def main():
 import numpy as np,cv2
 started=time.monotonic();v=Visibility54();loaded=time.monotonic();m,pred=v.metrics(v.x0);state,act,T,g,tf=v.decode(v.x0);hand,arm=v.ray_depths(T)
 localcam=(v.camera-v.boxT[:3,3])@v.boxT[:3,:3];surface=v.full_surface(tf);local=(surface-v.boxT[:3,3])@v.boxT[:3,:3];gap=np.abs(local)-v.boxsize/2;face=np.argmax(gap,axis=1);sign=np.sign(local[np.arange(len(local)),face]);closest=np.argmin(np.maximum(gap,0).sum(axis=1));boxhit=np.isfinite(v.box_depth);hits=v.camera+v.direction[boxhit]*v.box_depth[boxhit,None];hl=(hits-v.boxT[:3,3])@v.boxT[:3,:3];hf=np.argmax(np.abs(hl)/(v.boxsize/2),axis=1);hs=np.sign(hl[np.arange(len(hl)),hf]);grayhit=v.gray&np.isfinite(hand);before=(grayhit&boxhit)&(hand<v.box_depth);after=(grayhit&boxhit)&~before
 faces=lambda a,s:{('xyz'[i]+('+'if z>0 else '-')):int(((a==i)&(s==z)).sum())for i in range(3)for z in (-1,1)}
 m.update({'camera_in_box_local_m':localcam.tolist(),'hand_surface_nearest_face_counts':faces(face,sign),'visible_box_ray_entry_faces':faces(hf,hs),'source_gray_hand_hit_box_front':int(before.sum()),'source_gray_hand_hit_box_behind':int(after.sum()),'source_gray_no_hand_hit':int((v.gray&~np.isfinite(hand)).sum()),'arm_names':[v.kin.body_joint_names[i]for i in v.arm],'arm_neighbor_bounded_lower':v.lower[:7].tolist(),'arm_neighbor_bounded_upper':v.upper[:7].tolist(),'arm_source_position':v.x0[:7].tolist(),'arm_slack_to_lower_rad':(v.x0[:7]-v.lower[:7]).tolist(),'arm_slack_to_upper_rad':(v.upper[:7]-v.x0[:7]).tolist(),'ground_landmarks':v.ground_errors,'ground_anchored_wrist_budget_px':v.wrist_budget})
 y,x=np.nonzero(v.obs['domain']);x0,x1=x.min()-8,x.max()+9;y0,y1=y.min()-8,y.max()+9
 tiles=[]
 for label,mask,color in [('source gray hand',v.obs['hand_mask'],(0,255,255)),('hand alone',np.zeros(pred.shape,bool),(0,255,0)),('actual visible',pred,(255,255,0)),('hand behind box',np.zeros(pred.shape,bool),(0,0,255))]:
  if label=='hand alone':mask[v.yy,v.xx]=np.isfinite(hand)
  if label=='hand behind box':mask[v.yy,v.xx]=after
  im=cv2.cvtColor(v.rgb[0],cv2.COLOR_RGB2BGR);im[mask]=im[mask]*.45+np.asarray(color)*.55;im=im[y0:y1,x0:x1];im=cv2.resize(im,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST);cv2.putText(im,label,(8,20),0,.55,(255,255,255),1);tiles.append(im)
 out=ROOT/'outputs/user_reconstruction_v4';out.mkdir(exist_ok=True);cv2.imwrite(str(out/'source_visibility_054.jpg'),np.concatenate(tiles,axis=1));report={'schema':'kinogen.v4.visibility.diagnostic.1','frame':54,'inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()},'observation_module_sha256':sha(ROOT/'scripts/user_hand_observations_v3.py'),'metrics':m,'elapsed_seconds':time.monotonic()-started,'load_seconds':loaded-started,'strict':False,'source_unchanged':True};path=ROOT/'reports/user-v4-visibility-source.json';path.write_text(json.dumps(report,indent=2,allow_nan=True));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
