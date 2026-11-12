#!/usr/bin/env python3
"""Read-only v3 inspection of the actual RGB/observations and full hand meshes."""
from pathlib import Path
import hashlib,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 import numpy as np,cv2
 from user_body_kinematics import G1Kinematics,project_points
 from hand_geometry import box_signed_distance
 from user_rohand_coupling import ROHandCoupling
 out=ROOT/'outputs/user_reconstruction_v3';out.mkdir(exist_ok=True)
 parents=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
 paths={k:parents/v for k,v in {'body':'body_trajectory_contact.npz','hand':'hand_trajectory_contact.npz','camera':'camera_refinement.npz','box':'box_trajectory.npz','landmarks':'body_landmark_observations.npz'}.items()}
 d={k:np.load(p) for k,p in paths.items()};V=np.load(source/'input.npz');S=np.load(source/'segmentation.npz');rgb=V['rgb'];q=d['hand']['body_joint_positions_rad_refined'];P=d['camera']['P_refined'];kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');coupling=ROHandCoupling(ROOT)
 visuals=[]
 for v in kin.visuals:
  if '_hand_ro_' not in v['link_name']:continue
  vertices=v['vertices_local'];points=np.concatenate((np.unique(np.round(vertices,10),axis=0),vertices[v['faces']].mean(1)))
  visuals.append((v['link_name'],v['link_index'],points))
 records=[];tiles=[]
 for f in range(50,59):
  tf=kin.forward(d['body']['base_xyz_m'][f],d['body']['base_quaternion_xyzw'][f],q[f],coupling.expand_one(d['hand']['actuator_positions'][f]));links=[]
  canvas=cv2.cvtColor(rgb[f,0],cv2.COLOR_RGB2BGR)
  contour=cv2.findContours(S['box'][f,0].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0];cv2.drawContours(canvas,contour,-1,(0,170,255),1)
  for name,idx,pts in visuals:
   if not name.startswith('right_'):continue
   world=pts@tf[idx,:3,:3].T+tf[idx,:3,3];sdf=box_signed_distance(world,d['box']['box_world_transform'][f],d['box']['box_size_xyz_m']);links.append({'link':name,'min_sdf_m':float(sdf.min()),'fraction_beyond_5mm':float(np.mean(sdf<-.005)),'surface_count':len(sdf)})
   uv=project_points(P[0],world);take=np.linspace(0,len(uv)-1,min(150,len(uv))).astype(int)
   for xy,dd in zip(uv[take],sdf[take]):
    x,y=np.round(xy).astype(int)
    if 0<=x<640 and 0<=y<720:cv2.circle(canvas,(x,y),1,(40,40,245)if dd<-.005 else (60,210,80),-1)
  observations=[]
  for idx,color in ((2,(255,80,80)),(3,(255,0,255))):
   xy=d['landmarks']['positions_xy'][f,0,idx];ok=bool(d['landmarks']['valid'][f,0,idx]);observations.append({'name':str(d['landmarks']['point_names'][idx]),'xy':xy.tolist(),'valid':ok})
   if ok:
    x,y=np.round(xy).astype(int);cv2.drawMarker(canvas,(x,y),color,cv2.MARKER_CROSS,12,1);cv2.putText(canvas,str(idx),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
  wrist=tf[kin.link_index['right_wrist_yaw_link'],:3,3];wristuv=project_points(P[0],wrist[None])[0]
  records.append({'frame':f,'link_penetration':sorted(links,key=lambda a:a['min_sdf_m']),'actual_landmarks':observations,'wrist_projection_xy':wristuv.tolist(),'wrist_error_px':float(np.linalg.norm(wristuv-d['landmarks']['positions_xy'][f,0,2])),'box_strict_valid':bool(d['box']['valid'][f])})
  crop=canvas[230:420,170:410];crop=cv2.resize(crop,(600,475));cv2.putText(crop,f'FRAME {f} | red=inside box >5mm',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);tiles.append(crop)
 montage=np.concatenate([np.concatenate(tiles[i:i+3],axis=1)for i in (0,3,6)],axis=0);cv2.imwrite(str(out/'frame54_neighbors_mesh.jpg'),montage)
 for mode in ('rgb','mask'):
  tiles=[]
  for view in range(2):
   image=cv2.cvtColor(rgb[54,view],cv2.COLOR_RGB2BGR)
   if mode=='mask':
    image=np.where(S['robot'][54,view,:, :,None],image//2+np.asarray([0,100,0],np.uint8),image//3);image=np.where(S['box'][54,view,:,:,None],image//2+np.asarray([0,60,120],np.uint8),image).astype(np.uint8)
   for idx in (2,3,6,7):
    if d['landmarks']['valid'][54,view,idx]:
     x,y=np.round(d['landmarks']['positions_xy'][54,view,idx]).astype(int);cv2.drawMarker(image,(x,y),(255,0,255),cv2.MARKER_CROSS,12,1);cv2.putText(image,str(idx),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),1)
   tiles.append(image)
  cv2.imwrite(str(out/('frame54_'+mode+'.jpg')),np.concatenate(tiles,axis=1))
 report={'status':'read_only_actual_image_and_full_surface_diagnostic','source_sha256':{k:sha(p)for k,p in paths.items()},'camera_video_refined_max_abs_difference':float(np.max(abs(d['camera']['P_video']-P))),'records':records,'color_policy':'Red actual mesh surface points lie >5mm inside fixed candidate box; magenta crosses are actual manual/LK palm indices3/7, blue wrist2; never projection-derived targets.'}
 (ROOT/'reports/user-frame54-diagnostic-v3.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'worst_links_f54':records[4]['link_penetration'][:8],'observations_f54':records[4]['actual_landmarks'],'wrist_error':records[4]['wrist_error_px']}),flush=True)

if __name__=='__main__':main()
