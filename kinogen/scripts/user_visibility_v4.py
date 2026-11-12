"""Fixed RGB evidence and complete raw-mesh ray ordering for the F54 pilot."""
from pathlib import Path
import hashlib,os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Visibility54:
 def __init__(self):
  import numpy as np,cv2,open3d as o3d,trimesh,xml.etree.ElementTree as ET
  from scipy.spatial.transform import Rotation
  from user_body_kinematics import G1Kinematics,project_points
  from user_rohand_coupling import ROHandCoupling
  from hand_geometry import ROHandGeometry
  from user_hand_observations_v3 import make_observation
  self.np=np;self.cv2=cv2;self.o3d=o3d;self.project=project_points;self.f=54
  root=ROOT;v3=root/'outputs/user_reconstruction_v3';v2=root/'outputs/user_reconstruction_v2'
  self.paths={'body':v3/'body_trajectory.npz','hand':v3/'hand_trajectory.npz','camera':v2/'camera_refinement.npz','box':v2/'box_trajectory.npz','landmarks':v2/'body_landmark_observations.npz','ground':v3/'ground_trajectory_retry.npz','input':root/'outputs/user_reconstruction/input.npz','segmentation':root/'outputs/user_reconstruction/segmentation.npz'}
  assert sha(self.paths['body'])=='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'
  assert sha(self.paths['hand'])=='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1'
  self.d={k:np.load(p)for k,p in self.paths.items()if k not in ('input','segmentation')};B=self.d['body'];H=self.d['hand'];C=self.d['camera'];L=self.d['landmarks'];V=np.load(self.paths['input']);S=np.load(self.paths['segmentation'])
  self.rgb=V['rgb'][54].copy();self.valid=S['observation_valid'][54].copy();self.robot=S['robot'][54].copy();self.boxmask=S['box'][54].copy();self.video_sha=str(V['video_sha256']);self.P=C['P_refined'];self.times=H['timestamps'];self.coupling=ROHandCoupling(root);self.kin=G1Kinematics(root,scene_path='data/video_generation/candidate_scene/scene_rohand.json');self.model=ROHandGeometry(root,'right')
  self.q=B['joint_positions_rad'].copy();self.act=H['actuator_positions'].copy();self.arm=np.asarray([self.kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]);self.state=np.r_[B['base_xyz_m'][54],Rotation.from_quat(B['base_quaternion_xyzw'][54]).as_rotvec(),self.q[54]]
  self.wrist=self.kin.link_index['right_wrist_yaw_link'];ref=self.kin.scene['robot']['link_world_transforms'];self.mount=np.linalg.inv(np.asarray(ref['right_wrist_yaw_link']))@np.asarray(ref['right_hand_ro_base_link']);self.boxT=self.d['box']['box_world_transform'][54];self.boxsize=self.d['box']['box_size_xyz_m']
  self.obs=make_observation(self.rgb[0],self.robot[0],self.boxmask[0],self.valid[0],L['positions_xy'][54,0,3],L['positions_xy'][54,0,2]);assert self.obs is not None;self.obs['points']=self.obs['training_all_points']
  self.yy,self.xx=np.nonzero(self.obs['domain']);self.xy=np.column_stack((self.xx,self.yy));M=self.P[0,:,:3];self.camera=-np.linalg.solve(M,self.P[0,:,3]);direction=np.column_stack((self.xy+.5,np.ones(len(self.xy))))@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=1,keepdims=True);self.direction=direction;self.rays=o3d.core.Tensor(np.column_stack((np.broadcast_to(self.camera,direction.shape),direction)).astype(np.float32));self.gray=self.obs['hand_mask'][self.yy,self.xx];self.held=(self.xx//6+self.yy//6)%5==0;self.train_gray=np.flatnonzero(self.gray&~self.held)[::3]
  srcT=self.kin.fk(self.state,self.coupling.expand_one(self.act[54]));self.hand_visuals=[];self.arm_visuals=[];fixed=[]
  for visual in self.kin.visuals:
   item=(visual['link_name'],visual['link_index'],visual['vertices_local'],visual['faces'])
   if visual['link_name'].startswith('right_hand_ro_'):self.hand_visuals.append(item)
   elif self.kin.ancestor[visual['link_index'],self.arm].any():self.arm_visuals.append(item)
   else:
    T=srcT[visual['link_index']];fixed.append((visual['vertices_local']@T[:3,:3].T+T[:3,3],visual['faces']))
  table=self.kin.scene['table']
  for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
   m=trimesh.creation.box(extents=size);m.apply_translation(center);fixed.append((np.asarray(m.vertices),np.asarray(m.faces)))
  box=trimesh.creation.box(extents=self.boxsize);bv=np.asarray(box.vertices)@self.boxT[:3,:3].T+self.boxT[:3,3];fixed.append((bv,np.asarray(box.faces)));self.static_scene=self.scene(fixed);self.static_depth=self.cast(self.static_scene);self.box_scene=self.scene([(bv,np.asarray(box.faces))]);self.box_depth=self.cast(self.box_scene)
  self.point_links=np.asarray([self.kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);self.zero=np.zeros((8,3));G=self.d['ground'];gs=np.r_[G['base_xyz_m'][54],Rotation.from_quat(G['base_quaternion_xyzw'][54]).as_rotvec(),G['joint_positions_rad'][54]];self.ground_errors=self.landmarks(gs);self.wrist_budget=min(8,self.ground_errors[0]['wrist_error_px']+3)
  xml=ET.parse(self.kin.urdf_path).getroot();vs={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in self.kin.body_index};self.vel=np.asarray([vs[n]for n in self.kin.body_joint_names]);v=self.vel[self.arm];lo=np.maximum.reduce([self.kin.body_limits[self.arm,0],self.q[53,self.arm]-v*(self.times[54]-self.times[53]),self.q[55,self.arm]-v*(self.times[55]-self.times[54])]);hi=np.minimum.reduce([self.kin.body_limits[self.arm,1],self.q[53,self.arm]+v*(self.times[54]-self.times[53]),self.q[55,self.arm]+v*(self.times[55]-self.times[54])]);self.lower=np.r_[lo,np.zeros(6)];self.upper=np.r_[hi,np.ones(6)];self.x0=np.r_[self.q[54,self.arm],(self.act[54,6:]-self.model.lower)/(self.model.upper-self.model.lower)]
 def scene(self,parts):
  np=self.np;o3d=self.o3d;vertices=[];faces=[];offset=0
  for v,f in parts:vertices.append(v);faces.append(f+offset);offset+=len(v)
  scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.core.Tensor(np.concatenate(vertices).astype(np.float32)),o3d.core.Tensor(np.concatenate(faces).astype(np.uint32)));return scene
 def cast(self,scene):return scene.cast_rays(self.rays,nthreads=4)['t_hit'].numpy().astype(float)
 def decode(self,x):
  state=self.state.copy();state[6+self.arm]=x[:7];act=self.act[54].copy();act[6:]=self.model.lower+x[7:]*(self.model.upper-self.model.lower);T=self.kin.fk(state,self.coupling.expand_one(act));g,tips,tf=self.model.world_geometry(act[6:],T[self.wrist]@self.mount);return state,act,T,g,tf
 def landmarks(self,state):
  np=self.np;L=self.d['landmarks'];world=self.kin.points(state,self.point_links,self.zero);answer=[]
  for view in range(2):
   good=L['valid'][54,view].copy();good[[3,7]]=False;uv=self.project(self.P[view],world);err=np.linalg.norm(uv[good]-L['positions_xy'][54,view,good],axis=1);wi=2+4*view;answer.append({'median_px':float(np.median(err)),'p90_px':float(np.percentile(err,90)),'wrist_error_px':float(np.linalg.norm(uv[wi]-L['positions_xy'][54,view,wi]))if L['valid'][54,view,wi]else None})
  return answer
 def boundary(self,g,count=16):
  from hand_geometry import polygon_samples
  return self.np.concatenate([polygon_samples(self.project(self.P[0],p),count)for p in g.values()])
 def full_surface(self,tf):return self.model.contact_surface(tf,dense=True)
 def raw_parts(self,T,visuals):return [(v@T[i,:3,:3].T+T[i,:3,3],f)for _,i,v,f in visuals]
 def ray_depths(self,T):
  return self.cast(self.scene(self.raw_parts(T,self.hand_visuals))),self.cast(self.scene(self.raw_parts(T,self.arm_visuals)))
 def visibility(self,T):
  np=self.np;cv2=self.cv2;hand,arm=self.ray_depths(T);visible=np.isfinite(hand)&(hand<np.minimum(self.static_depth,arm)-1e-5);pred=np.zeros((720,640),bool);pred[self.yy,self.xx]=visible;truth=self.obs['hand_mask'];domain=self.obs['domain'];inter=(pred&truth&domain).sum();union=((pred|truth)&domain).sum()
  def boundary(m):
   c=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0];e=np.zeros(m.shape,np.uint8);cv2.drawContours(e,[x for x in c if len(x)>=5],-1,1,1);return e.astype(bool)
  interior=cv2.erode(domain.astype(np.uint8),np.ones((5,5),np.uint8))>0;edge=boundary(truth)&interior;y,x=np.nonzero(edge);held=(x//6+y//6)%5==0;src=np.column_stack((x[held],y[held]));py,px=np.nonzero(boundary(pred)&interior)
  from scipy.spatial import cKDTree
  dist=cKDTree(np.column_stack((px,py))).query(src)[0]if len(px)and len(src)else np.asarray([np.inf]);gray_hand_hit=self.gray&np.isfinite(hand);viol=self.gray&np.isfinite(hand)&np.isfinite(self.box_depth)&(hand>self.box_depth+1e-4)
  return {'visible_mask_iou':float(inter/max(union,1)),'visible_gray_recall':float(inter/max(truth.sum(),1)),'visible_hand_pixels':int(visible.sum()),'visible_gray_pixels':int(inter),'source_gray_pixels':int(truth.sum()),'external_heldout_mean_px':float(dist.mean()),'external_heldout_p90_px':float(np.percentile(dist,90)),'hand_alone_gray_coverage':float(gray_hand_hit.sum()/max(self.gray.sum(),1)),'gray_rays_hand_behind_box_count':int(viol.sum()),'gray_rays_hand_box_depth_excess_mean_m':float(np.mean(hand[viol]-self.box_depth[viol]))if viol.any()else 0.},pred,hand,arm
 def metrics(self,x):
  from optimize_user_hands import image_residual,heldout_error
  from hand_geometry import box_signed_distance
  np=self.np;state,act,T,g,tf=self.decode(x);sdf=box_signed_distance(self.full_surface(tf),self.boxT,self.boxsize);visibility,pred,hand,arm=self.visibility(T);u=x[7:];base=T[self.wrist]@self.mount;res=image_residual(self.model,u,base,self.P,[self.obs,None]);return {'visibility':visibility,'max_hand_box_penetration_m':float(max(0,-sdf.min())),'min_hand_box_sdf_m':float(sdf.min()),'legacy_weighted_training_rms_px':float(np.sqrt(np.mean(res**2))*3),'legacy_proxy12_heldout_px':heldout_error(self.model,u,base,self.P,[self.obs,None]),'landmarks':self.landmarks(state)},pred
