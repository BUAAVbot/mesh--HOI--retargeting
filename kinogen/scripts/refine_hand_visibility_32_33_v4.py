"""Independent frames32/33 RGB evidence and raw mesh ray-ordering; immutable V3."""
from pathlib import Path
import hashlib,os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class VisibilityFrame:
 def __init__(self,frame,q_override=None,act_override=None):
  import numpy as np,cv2,open3d as o3d,trimesh,xml.etree.ElementTree as ET
  from scipy.spatial.transform import Rotation
  from user_body_kinematics import G1Kinematics,project_points
  from user_rohand_coupling import ROHandCoupling
  from hand_geometry import ROHandGeometry
  from user_hand_observations_v3 import make_observation
  self.np=np;self.cv2=cv2;self.o3d=o3d;self.project=project_points;self.f=int(frame)
  root=ROOT;v3=root/'outputs/user_reconstruction_v3';v2=root/'outputs/user_reconstruction_v2'
  self.paths={'body':v3/'body_trajectory.npz','hand':v3/'hand_trajectory.npz','camera':v2/'camera_refinement.npz','box':v2/'box_trajectory.npz','landmarks':v2/'body_landmark_observations.npz','ground':v3/'ground_trajectory_retry.npz','input':root/'outputs/user_reconstruction/input.npz','segmentation':root/'outputs/user_reconstruction/segmentation.npz'}
  assert sha(self.paths['body'])=='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'
  assert sha(self.paths['hand'])=='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1'
  self.d={k:np.load(p)for k,p in self.paths.items()if k not in ('input','segmentation')};B=self.d['body'];H=self.d['hand'];C=self.d['camera'];L=self.d['landmarks'];V=np.load(self.paths['input']);S=np.load(self.paths['segmentation'])
  self.rgb=V['rgb'][self.f].copy();self.valid=S['observation_valid'][self.f].copy();self.robot=S['robot'][self.f].copy();self.boxmask=S['box'][self.f].copy();self.video_sha=str(V['video_sha256']);self.P=C['P_refined'];self.times=H['timestamps'];self.coupling=ROHandCoupling(root);self.kin=G1Kinematics(root,scene_path='data/video_generation/candidate_scene/scene_rohand.json');self.model=ROHandGeometry(root,'right')
  self.q=B['joint_positions_rad'].copy()if q_override is None else q_override.copy();self.act=H['actuator_positions'].copy()if act_override is None else act_override.copy();self.arm=np.asarray([self.kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]);self.state=np.r_[B['base_xyz_m'][self.f],Rotation.from_quat(B['base_quaternion_xyzw'][self.f]).as_rotvec(),self.q[self.f]]
  self.wrist=self.kin.link_index['right_wrist_yaw_link'];ref=self.kin.scene['robot']['link_world_transforms'];self.mount=np.linalg.inv(np.asarray(ref['right_wrist_yaw_link']))@np.asarray(ref['right_hand_ro_base_link']);self.boxT=self.d['box']['box_world_transform'][self.f];self.boxsize=self.d['box']['box_size_xyz_m']
  self.obs=make_observation(self.rgb[0],self.robot[0],self.boxmask[0],self.valid[0],L['positions_xy'][self.f,0,3],L['positions_xy'][self.f,0,2]);assert self.obs is not None,'No valid actual RGB hand observation';self.obs['points']=self.obs['training_all_points']
  self.yy,self.xx=np.nonzero(self.obs['domain']);self.xy=np.column_stack((self.xx,self.yy));M=self.P[0,:,:3];self.camera=-np.linalg.solve(M,self.P[0,:,3]);direction=np.column_stack((self.xy+.5,np.ones(len(self.xy))))@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=1,keepdims=True);self.direction=direction;self.rays=o3d.core.Tensor(np.column_stack((np.broadcast_to(self.camera,direction.shape),direction)).astype(np.float32));self.gray=self.obs['hand_mask'][self.yy,self.xx];self.held=(self.xx//6+self.yy//6)%5==0;self.train_gray=np.flatnonzero(self.gray&~self.held)[::3]
  srcT=self.kin.fk(self.state,self.coupling.expand_one(self.act[self.f]));self.hand_visuals=[];self.arm_visuals=[];fixed=[]
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
  self.point_links=np.asarray([self.kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);self.zero=np.zeros((8,3));G=self.d['ground'];gs=np.r_[G['base_xyz_m'][self.f],Rotation.from_quat(G['base_quaternion_xyzw'][self.f]).as_rotvec(),G['joint_positions_rad'][self.f]];self.ground_errors=self.landmarks(gs);self.wrist_budget=min(8,self.ground_errors[0]['wrist_error_px']+3)
  xml=ET.parse(self.kin.urdf_path).getroot();vs={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in self.kin.body_index};self.vel=np.asarray([vs[n]for n in self.kin.body_joint_names]);v=self.vel[self.arm];lo=np.maximum.reduce([self.kin.body_limits[self.arm,0],self.q[self.f-1,self.arm]-v*(self.times[self.f]-self.times[self.f-1]),self.q[self.f+1,self.arm]-v*(self.times[self.f+1]-self.times[self.f])]);hi=np.minimum.reduce([self.kin.body_limits[self.arm,1],self.q[self.f-1,self.arm]+v*(self.times[self.f]-self.times[self.f-1]),self.q[self.f+1,self.arm]+v*(self.times[self.f+1]-self.times[self.f])]);self.lower=np.r_[lo,np.zeros(6)];self.upper=np.r_[hi,np.ones(6)];self.x0=np.r_[self.q[self.f,self.arm],(self.act[self.f,6:]-self.model.lower)/(self.model.upper-self.model.lower)]
 def scene(self,parts):
  np=self.np;o3d=self.o3d;vertices=[];faces=[];offset=0
  for v,f in parts:vertices.append(v);faces.append(f+offset);offset+=len(v)
  scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.core.Tensor(np.concatenate(vertices).astype(np.float32)),o3d.core.Tensor(np.concatenate(faces).astype(np.uint32)));return scene
 def cast(self,scene):return scene.cast_rays(self.rays,nthreads=4)['t_hit'].numpy().astype(float)
 def decode(self,x):
  state=self.state.copy();state[6+self.arm]=x[:7];act=self.act[self.f].copy();act[6:]=self.model.lower+x[7:]*(self.model.upper-self.model.lower);T=self.kin.fk(state,self.coupling.expand_one(act));g,tips,tf=self.model.world_geometry(act[6:],T[self.wrist]@self.mount);return state,act,T,g,tf
 def landmarks(self,state):
  np=self.np;L=self.d['landmarks'];world=self.kin.points(state,self.point_links,self.zero);answer=[]
  for view in range(2):
   good=L['valid'][self.f,view].copy();good[[3,7]]=False;uv=self.project(self.P[view],world);err=np.linalg.norm(uv[good]-L['positions_xy'][self.f,view,good],axis=1);wi=2+4*view;answer.append({'median_px':float(np.median(err)),'p90_px':float(np.percentile(err,90)),'wrist_error_px':float(np.linalg.norm(uv[wi]-L['positions_xy'][self.f,view,wi]))if L['valid'][self.f,view,wi]else None})
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

def clean(v):
 import math
 if isinstance(v,dict):return {str(k):clean(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x)for x in v]
 if hasattr(v,'tolist'):return clean(v.tolist())
 return None if isinstance(v,float)and not math.isfinite(v)else v

def body_metrics(v,state):
 """Actual SAM body holdout; every hand is excluded from body mesh/domain."""
 import numpy as np,cv2,trimesh
 from scipy.spatial import cKDTree
 hh,ww=240,213;xx,yy=np.meshgrid((np.arange(ww)+.5)*640/ww,(np.arange(hh)+.5)*720/hh);T=v.kin.fk(state);parts=[]
 for visual in v.kin.visuals:
  if '_hand_ro_'in visual['link_name']:continue
  t=T[visual['link_index']];parts.append((visual['vertices_local']@t[:3,:3].T+t[:3,3],visual['faces']))
 scene=v.scene(parts);table=v.kin.scene['table'];occluders=[]
 for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
  mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);occluders.append((np.asarray(mesh.vertices),np.asarray(mesh.faces)))
 mesh=trimesh.creation.box(extents=v.boxsize);occluders.append((np.asarray(mesh.vertices)@v.boxT[:3,:3].T+v.boxT[:3,3],np.asarray(mesh.faces)));occ=v.scene(occluders);L=v.d['landmarks'];answer=[]
 for view in range(2):
  M=v.P[view,:,:3];camera=-np.linalg.solve(M,v.P[view,:,3]);directions=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;directions/=np.linalg.norm(directions,axis=-1,keepdims=True);rays=v.o3d.core.Tensor(np.concatenate((np.broadcast_to(camera,directions.shape),directions),axis=-1).astype(np.float32));depth=scene.cast_rays(rays,nthreads=4)['t_hit'].numpy();occlusion=occ.cast_rays(rays,nthreads=4)['t_hit'].numpy();mask=np.isfinite(depth)&(depth<occlusion-1e-5)
  def domain(points):
   keep=np.ones(len(points),bool)
   for k,radius in ((2+4*view,55),(3+4*view,35)):
    if L['valid'][v.f,view,k]:keep&=np.linalg.norm(points-L['positions_xy'][v.f,view,k],axis=1)>radius
   return keep
  valid=domain(np.column_stack((xx.ravel(),yy.ravel()))).reshape(hh,ww)&(cv2.resize(v.valid[view].astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)>0);truth=v.robot[view]&~v.boxmask[view]&v.valid[view];contours=cv2.findContours(truth.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0];pixels=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;pixels=pixels[domain(pixels)];held=((pixels[:,0]//16).astype(int)+64*(pixels[:,1]//16).astype(int))%5==0;pixels=pixels[held];boundary=mask&~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);iy,ix=np.nonzero(boundary);model=np.column_stack(((ix+.5)*640/ww,(iy+.5)*720/hh));model=model[domain(model)];distance=cKDTree(model).query(pixels)[0];modelmask=mask&valid;target=(cv2.resize(truth.astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)>0)&valid
  answer.append({'view':view,'body_mask_iou':float((modelmask&target).sum()/max((modelmask|target).sum(),1)),'body_heldout_median_px':float(np.median(distance)),'body_heldout_p90_px':float(np.percentile(distance,90)),'heldout_pixels':len(pixels)})
 return answer

def main():
 import argparse,time,json,numpy as np,cv2
 from scipy.optimize import least_squares
 from scipy.spatial import cKDTree
 from hand_geometry import box_signed_distance
 from optimize_user_hands import sample
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--budget-seconds',type=float,default=480);ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--front-nfev',type=int,default=45);ap.add_argument('--ray-nfev',type=int,default=35);a=ap.parse_args();started=time.monotonic();deadline=started+min(600,a.budget_seconds);out=ROOT/'outputs/user_reconstruction_v4';out.mkdir(exist_ok=True)
 q=None;acts=None;sourceB=None;sourceH=None;records=[];selected={};inputs={};seed_file=out/'candidate_2_raw_rays_body.npz';seed_hand=out/'candidate_2_raw_rays_hand.npz'
 class BudgetEnded(Exception):pass
 for frame in (32,33):
  v=VisibilityFrame(frame,q,acts)
  if q is None:
   sourceB={k:v.d['body'][k].copy()for k in v.d['body'].files};sourceH={k:v.d['hand'][k].copy()for k in v.d['hand'].files};q=v.q.copy();acts=v.act.copy();inputs={k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()};inputs['observation_module']={'path':'scripts/user_hand_observations_v3.py','sha256':sha(ROOT/'scripts/user_hand_observations_v3.py')}
  baseline,_=v.metrics(v.x0);baseline_body=body_metrics(v,v.state);L=v.d['landmarks'];fixed=v.train_gray;fixed=fixed[np.isfinite(v.box_depth[fixed])];target=v.camera+v.direction[fixed]*(v.box_depth[fixed]-.002)[:,None];trials=[];source_ground=v.ground_errors
  def surface(tf,local):return np.concatenate([points@tf['right_hand_ro_'+name][:3,:3].T+tf['right_hand_ro_'+name][:3,3]for name,points in local.items()])
  def assess(x):
   m,pred=v.metrics(x);state,*_=v.decode(x);body=body_metrics(v,state);m['body']=body;why=[]
   if m['max_hand_box_penetration_m']>1e-6:why.append('raw_mesh_penetration')
   if m['min_hand_box_sdf_m']>.004:why.append('hand_not_near_box')
   for view in range(2):
    for key,absolute,extra in [('wrist_error_px',8,3),('median_px',8,2),('p90_px',16,3)]:
     if m['landmarks'][view][key]is not None and m['landmarks'][view][key]>min(absolute,source_ground[view][key]+extra)+1e-6:why.append(f'view{view}_ground_{key}')
    for key,absolute,extra in [('body_heldout_median_px',6,2),('body_heldout_p90_px',15,3)]:
     if body[view][key]>min(absolute,baseline_body[view][key]+extra)+1e-6:why.append(f'view{view}_{key}')
    if body[view]['body_mask_iou']<max(.55,baseline_body[view]['body_mask_iou']-.04):why.append(f'view{view}_body_iou')
   if m['visibility']['visible_gray_pixels']<100:why.append('missing_actual_raw_visible_hand')
   strict=m['legacy_weighted_training_rms_px']<4 and m['legacy_proxy12_heldout_px']<4.5
   if m['legacy_weighted_training_rms_px']>baseline['legacy_weighted_training_rms_px']+.03:why.append('training_worse_than_parent')
   if m['legacy_proxy12_heldout_px']>baseline['legacy_proxy12_heldout_px']+.25:why.append('proxy12_worse_than_parent')
   return m,pred,why,strict
  seeds=[v.x0.copy()]
  if seed_file.exists()and seed_hand.exists():
   with np.load(seed_file)as sb,np.load(seed_hand)as sh:
    seed=v.x0.copy();seed[:7]=sb['joint_positions_rad'][54,v.arm];seed[7:]=(sh['actuator_positions'][54,6:]-v.model.lower)/(v.model.upper-v.model.lower);seeds.append(seed)
   inputs['front_branch_initial_body']={'path':str(seed_file.relative_to(ROOT)),'sha256':sha(seed_file)};inputs['front_branch_initial_hand']={'path':str(seed_hand.relative_to(ROOT)),'sha256':sha(seed_hand)}
  else:
   seed=v.x0.copy();seed[:7]=q[frame-1,v.arm];seed[7:12]=.12;seed[12]=.45;seeds.append(seed)
  for si,seed in enumerate(seeds[:min(2,a.seeds)]):
   if time.monotonic()>deadline:break
   x=np.clip(seed,v.lower+1e-7,v.upper-1e-7);history=[]
   for phase,nfev in [('front_init',min(60,a.front_nfev)),('raw_rays_wrist',min(50,a.ray_nfev))]:
    for outer in range(2):
     if time.monotonic()>deadline:break
     _,_,_,_,tf=v.decode(x);active={}
     for name,points in v.model.dense_contact_vertices.items():
      t=tf['right_hand_ro_'+name];sdf=box_signed_distance(points@t[:3,:3].T+t[:3,3],v.boxT,v.boxsize);deep=np.argpartition(sdf,min(31,len(points)-1))[:min(32,len(points))];active[name]=np.concatenate((v.model.contact_vertices[name],points[deep]))
     best_local=[np.inf,x.copy()]
     def residual(z):
      if time.monotonic()>deadline:raise BudgetEnded
      state,act,T,g,tf=v.decode(z);surf=surface(tf,v.model.contact_vertices);nearest=cKDTree(surf).query(target)[1];delta=surf[nearest]-target;boundary=v.boundary(g,12);back=cKDTree(boundary).query(v.obs['training_all_points'])[0];forward=sample(v.obs['distance'],boundary,cval=40);outside=sample(v.obs['outside'],boundary,cval=40);result=[delta.ravel()/.008*np.sqrt(250/max(1,len(target))),forward*.25/3,outside*.2/3,back*.7/3];sd=box_signed_distance(surface(tf,active),v.boxT,v.boxsize);pen=np.maximum(.001-sd,0)/.003;result.extend((pen*np.sqrt(1500/len(pen)),np.asarray([pen.max()*70]),np.asarray([max(float(sd.min())-.003,0)/.005*20])))
      world=v.kin.points(state,v.point_links,v.zero)
      for view in range(2):
       good=L['valid'][frame,view].copy();good[[3,7]]=False;uv=v.project(v.P[view],world);diff=uv[good]-L['positions_xy'][frame,view,good];weight=np.asarray([1,1.5,5,0,1,1.5,5,0]);result.append((diff*weight[good,None]/3).ravel());err=np.linalg.norm(diff,axis=1);wi=2+4*view;werr=np.linalg.norm(uv[wi]-L['positions_xy'][frame,view,wi]);budgets=np.asarray([min(8,source_ground[view]['wrist_error_px']+3)-.1,min(8,source_ground[view]['median_px']+2)-.1,min(16,source_ground[view]['p90_px']+3)-.1]);result.append(np.maximum(np.asarray([werr,np.median(err),np.percentile(err,90)])-budgets,0)*100)
      if phase=='raw_rays_wrist':
       hand,arm=v.ray_depths(T);depth=hand[fixed];occluder=np.minimum(v.static_depth[fixed],arm[fixed]);hit=np.isfinite(depth);gap=np.zeros(len(depth));gap[hit]=np.maximum(depth[hit]-occluder[hit]+.0005,0);gap[~hit]=np.minimum(np.linalg.norm(delta[~hit],axis=1),.3);result.append(gap/.006*np.sqrt(150/max(1,len(gap))))
      result.extend(((z[:7]-v.x0[:7])*.02,(z[7:]-v.x0[7:])*.01));r=np.concatenate(result);cost=float(r@r)
      if cost<best_local[0]:best_local[:]=[cost,z.copy()]
      return r
     try:
      fit=least_squares(residual,x,bounds=(v.lower,v.upper),diff_step=.002,max_nfev=nfev,loss='linear'if phase=='raw_rays_wrist'else 'soft_l1',f_scale=3,ftol=3e-5,xtol=3e-5,gtol=3e-4);x=fit.x;history.append({'phase':phase,'outer':outer,'nfev':fit.nfev,'cost':float(fit.cost)})
     except BudgetEnded:x=best_local[1];history.append({'phase':phase,'outer':outer,'budget_terminated':True});break
    m,pred,why,strict=assess(x);trials.append({'seed':si,'phase':phase,'x':x.copy(),'metrics':m,'rejections':why,'strict_image_fit':strict,'history':history.copy()});print(json.dumps(clean({'frame':frame,'seed':si,'phase':phase,'rejections':why,'strict_image_fit':strict,'visible_gray':m['visibility']['visible_gray_pixels'],'wrist_errors':[t['wrist_error_px']for t in m['landmarks']],'hand_gap':m['min_hand_box_sdf_m'],'elapsed_seconds':time.monotonic()-started})),flush=True)
    if not why:break
   if trials and not trials[-1]['rejections']:break
  passing=[t for t in trials if not t['rejections']];chosen=max(passing,key=lambda t:t['metrics']['visibility']['visible_gray_pixels'])if passing else None
  if chosen is not None:
   state,act,T,_,_=v.decode(chosen['x']);q[frame]=state[6:];acts[frame]=act;selected[frame]=(chosen,T)
  records.append({'frame':frame,'baseline':baseline,'baseline_body':baseline_body,'ground_landmark_anchors':source_ground,'selected':chosen is not None,'selected_conditional':bool(chosen is not None and not chosen['strict_image_fit']),'trials':trials});report={'status':'finite_search_running','inputs':inputs,'records':records,'budget_seconds':a.budget_seconds,'elapsed_seconds':time.monotonic()-started,'strict_metric_valid':False};(ROOT/'reports/user-v4-hand32-33-visibility.json').write_text(json.dumps(clean(report),indent=2)+'\n')
  if chosen is not None:
   _,pred=v.metrics(chosen['x']);canvas=cv2.cvtColor(v.rgb[0],cv2.COLOR_RGB2BGR);edge=pred&~cv2.erode(pred.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);canvas[edge]=(255,255,0);cv2.imwrite(str(out/f'frame_{frame:03d}_visibility_selected.jpg'),canvas)
 B={k:a.copy()for k,a in sourceB.items()};H={k:a.copy()for k,a in sourceH.items()};B['joint_positions_rad']=q;B['finger_actuator_positions']=acts;B['finger_joint_positions']=v.coupling.expand_array(acts);B['link_world_transforms']=sourceB['link_world_transforms'].copy();B.update(source_body_sha256=np.asarray(inputs['body']['sha256']),source_hand_sha256=np.asarray(inputs['hand']['sha256']),source_canonical_v3_body_sha256=np.asarray(inputs['body']['sha256']),source_canonical_v3_hand_sha256=np.asarray(inputs['hand']['sha256']),candidate_is_metric_ground_truth=np.asarray(False))
 for frame,(chosen,T)in selected.items():
  B['link_world_transforms'][frame]=T;changed=np.abs(q[frame]-sourceB['joint_positions_rad'][frame])>1e-7
  if 'joint_image_supported'in B:B['joint_image_supported'][frame,changed]=False
  if 'joint_approx_uncertainty_rad'in B:B['joint_approx_uncertainty_rad'][frame,changed]=np.nan
  if not chosen['strict_image_fit']:
   for key in ('candidate_valid','ground_image_valid','image_fit_valid'):
    if key in B:B[key][frame]=False
 stale=('contact_tip','hand_surface','contact_nonpenetration','wrist_keypoint_error','wrist_origin_world_shift','dense_collision','heldout_edge','legacy_observation','v3_proxy16');H={k:a.copy()for k,a in H.items()if not any(token in k for token in stale)};H.update(actuator_positions=acts,joint_positions=B['finger_joint_positions'],body_joint_positions_rad_refined=q,base_xyz_m=B['base_xyz_m'].copy(),base_quaternion_xyzw=B['base_quaternion_xyzw'].copy(),source_hand_sha256=np.asarray(inputs['hand']['sha256']),source_canonical_v3_body_sha256=np.asarray(inputs['body']['sha256']),source_canonical_v3_hand_sha256=np.asarray(inputs['hand']['sha256']))
 for frame,(chosen,_)in selected.items():
  for key in ('actuator_observed','observed','wrist_observed'):
   if key in H:H[key][frame]=False
  for key in ('actuator_uncertain','uncertain','prior_only','wrist_uncertain'):
   if key in H:H[key][frame]=True
  if 'actuator_std'in H:H['actuator_std'][frame]=np.nan
  if 'image_fit_valid'in H:H['image_fit_valid'][frame,1]=chosen['strict_image_fit']
  for key in ('valid','side_valid'):
   if key in H:H[key][frame]=False
 if 'wrist_joint_names'in H:
  wq=q[:,[v.kin.body_index[n]for n in H['wrist_joint_names']]];H['wrist_joint_positions_rad']=wq;H['all_wrist_joint_positions_rad']=wq
 if 'wrist_yaw_positions_rad'in H:H['wrist_yaw_positions_rad']=q[:,[v.kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]]
 for f in range(121):
  if f not in selected:assert np.array_equal(q[f],sourceB['joint_positions_rad'][f])and np.array_equal(acts[f],sourceH['actuator_positions'][f])
 assert np.array_equal(B['base_xyz_m'],sourceB['base_xyz_m'])and np.array_equal(B['base_quaternion_xyzw'],sourceB['base_quaternion_xyzw'])
 assert np.max(np.abs(np.diff(q,axis=0))/np.diff(v.times)[:,None]-v.vel)<1e-5
 bp=out/'stage_32_33_body.npz';np.savez_compressed(bp,**B);H['source_body_sha256']=np.asarray(sha(bp));H['body_sha256']=np.asarray(sha(bp));hp=out/'stage_32_33_hand.npz';np.savez_compressed(hp,**H);report.update(status='finite_search_completed',selected_frames=sorted(selected),outputs={'body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)}},all_original_body_positive_velocity_and_q_bounds_satisfied=True,other119_frames_exact_and_base_feet_all_fixed=True,stale_hand_derived_fields_removed_matching_tokens=stale,source_gray_is_RGB_only_and_fixed_spatial_heldout=True,front_box_targets_are_conditional_initialization_not_measured3D=True,elapsed_seconds=time.monotonic()-started);(ROOT/'reports/user-v4-hand32-33-visibility.json').write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps(clean({k:report[k]for k in ('status','selected_frames','outputs','elapsed_seconds')}),indent=2),flush=True)

if __name__=='__main__':main()
